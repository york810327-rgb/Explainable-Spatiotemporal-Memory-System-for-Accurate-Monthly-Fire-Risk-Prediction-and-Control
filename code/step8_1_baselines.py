#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step8_1 Baselines (final, paper-rigorous)
-----------------------------------------
Baselines:
- Climatology: p(y | BIOME, month) with Beta-Binomial smoothing + hierarchical fallback
- Memory baseline: Logistic on memory-only features
- Climate-only baseline: Logistic-L2 on climate/water-stress features
- Veg/Energy-only baseline: Logistic-L2 on NDVI/EVI/LST (+ rolling)

Key guarantees:
1) Training is STRICTLY protocol x fold-wise
2) y is taken ONLY from split parquet
3) P2' uses two-stage semantics:
   - inner model: fit on inner_train, predict inner_train / inner_val
   - outer model: fit on outer_train = inner_train ∪ inner_val, predict val / test
4) Final safety checks:
   - predicted rows exactly match split rows per (protocol, fold)
   - manifest records n_rows_train_fit per fold
   - all probability columns have no NaN and are within [0,1]

Notes:
- For P1 / P2 / P3 / P4: one fold-specific model per baseline ("single" stage)
- For P2': two model stages per fold ("inner" and "outer")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Utils
# ----------------------------

def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def file_hash(path: str, algo: str = "sha256", chunk_size: int = 2**20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def month_from_date(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.month.astype("int8")


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return float("nan")

    order = np.argsort(-y_prob)
    y_true = y_true[order]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    pos = tp[-1] if len(tp) else 0
    if pos == 0:
        return 0.0

    precision = tp / np.maximum(tp + fp, 1)
    ap = precision[y_true == 1].sum() / pos
    return float(ap)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


@dataclass
class FeatureSets:
    climo: List[str]
    memory: List[str]
    climate: List[str]
    vegenergy: List[str]


# ----------------------------
# Feature selection
# ----------------------------

def pick_feature_sets(df: pd.DataFrame, veg_drop_cov: bool) -> FeatureSets:
    cols = set(df.columns)

    climo = ["BIOME", "month"]

    # remove duplicate alias fire_tslf; keep months_since_last_fire only
    memory = [c for c in ["y_lag1", "fire_count_12m", "months_since_last_fire"] if c in cols]

    climate_candidates = [
        "P_sum_mon_lag1",
        "PET_sum_mon_lag1",
        "TP_sum_mon_lag1",
        "Tmax_mon_lag1",
        "Tmean_mon_lag1",
        "RH_mean_mon_lag1",
        "RH_min_mon_lag1",
        "SM1_mean_mon_lag1",
        "SM2_mean_mon_lag1",
        "VPD_mean_mon_lag1",
        "VPD_max_mon_lag1",
        "WS_mean_mon_lag1",
        "WS_max_mon_lag1",
        "WS_strong_frac_lag1",
        "WD_u_mon_lag1",
        "WD_v_mon_lag1",
        "WD_R_mon_lag1",
        "P_sum_mon_lag1_roll3m_sum",
        "PET_sum_mon_lag1_roll3m_sum",
        "VPD_mean_mon_lag1_roll3m_mean",
    ]
    climate = [c for c in climate_candidates if c in cols]

    for base in [
        "P_sum_mon", "PET_sum_mon", "TP_sum_mon",
        "Tmax_mon", "Tmean_mon",
        "RH_mean_mon", "RH_min_mon",
        "SM1_mean_mon", "SM2_mean_mon",
        "VPD_mean_mon", "VPD_max_mon",
        "WS_mean_mon", "WS_max_mon", "WS_strong_frac",
        "WD_u_mon", "WD_v_mon", "WD_R_mon",
    ]:
        cov = f"{base}_cov_lag1"
        mis = f"{base}_missing_lag1"
        if cov in cols:
            climate.append(cov)
        if mis in cols:
            climate.append(mis)

    veg_candidates = [
        "NDVI_mean_mon_lag1",
        "EVI_mean_mon_lag1",
        "LST_day_mon_lag1",
        "LST_night_mon_lag1",
        "NDVI_mean_mon_lag1_roll3m_mean",
    ]
    vegenergy = [c for c in veg_candidates if c in cols]

    # per your prior decision: drop *_missing_lag1 from veg baseline
    veg_bases = ["NDVI_mean_mon", "EVI_mean_mon", "LST_day_mon", "LST_night_mon"]
    for base in veg_bases:
        cov = f"{base}_cov_lag1"
        if (not veg_drop_cov) and cov in cols:
            vegenergy.append(cov)

    def dedup(lst: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in lst:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return FeatureSets(
        climo=dedup(climo),
        memory=dedup(memory),
        climate=dedup(climate),
        vegenergy=dedup(vegenergy),
    )


# ----------------------------
# Climatology
# ----------------------------

def fit_climo_with_fallback(
    train_df: pd.DataFrame,
    alpha: float,
    beta: float,
    biome_col: str = "BIOME",
    month_col: str = "month",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    g = train_df.groupby([biome_col, month_col], dropna=False)["y"]
    agg = g.agg(pos="sum", n="count").reset_index()
    agg["p"] = (agg["pos"] + alpha) / (agg["n"] + alpha + beta)

    gb = train_df.groupby([biome_col], dropna=False)["y"].agg(pos="sum", n="count").reset_index()
    gb["p"] = (gb["pos"] + alpha) / (gb["n"] + alpha + beta)

    pos = float(train_df["y"].sum())
    n = float(len(train_df))
    global_p = (pos + alpha) / (n + alpha + beta)

    fallback = {
        "global_p": float(global_p),
        "biome_level_p": {str(r[biome_col]): float(r["p"]) for _, r in gb.iterrows()},
    }
    return agg, fallback


def predict_climo_with_fallback(
    df: pd.DataFrame,
    climo_table: pd.DataFrame,
    fallback: Dict[str, float],
    biome_col: str = "BIOME",
    month_col: str = "month",
) -> np.ndarray:
    key = climo_table[[biome_col, month_col]].copy()
    key["p"] = climo_table["p"].values

    tmp = df[[biome_col, month_col]].merge(key, on=[biome_col, month_col], how="left")
    p = tmp["p"].to_numpy(dtype=float)

    biome_map = fallback.get("biome_level_p", {})
    global_p = fallback.get("global_p", 0.5)

    miss = np.isnan(p)
    if miss.any():
        biomes = df.loc[miss, biome_col].astype(str).tolist()
        p[miss] = np.array([biome_map.get(b, global_p) for b in biomes], dtype=float)

    miss2 = np.isnan(p)
    if miss2.any():
        p[miss2] = global_p

    return p


# ----------------------------
# Logistic baseline
# ----------------------------

def fit_logit_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    features: List[str],
    C: float,
    solver: str,
    max_iter: int,
    class_weight: Optional[str],
    standardize: bool = True,
) -> np.ndarray:
    X_train = train_df[features]
    y_train = train_df["y"].astype(int).to_numpy()
    X_pred = pred_df[features]

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append((
        "logit",
        LogisticRegression(
            penalty="l2",
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
        )
    ))
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_pred)[:, 1].astype(float)


# ----------------------------
# Validation helpers
# ----------------------------

def normalize_class_weight(s: str | None) -> Optional[str]:
    if s in [None, "", "none", "null", "None"]:
        return None
    if s not in ["balanced"]:
        raise ValueError(f"Unsupported class_weight={s}. Use 'none' or 'balanced'.")
    return s


def assert_required_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] missing required columns: {miss}")


def same_values_or_take_feature(
    split: pd.DataFrame,
    feat: pd.DataFrame,
    col: str,
) -> pd.Series:
    split_col = f"{col}_split"
    feat_col = col
    has_split = split_col in split.columns
    has_feat = feat_col in split.columns

    if has_split and has_feat:
        a = split[split_col]
        b = split[feat_col]
        ok = (a.isna() & b.isna()) | (a == b)
        if not ok.all():
            raise ValueError(f"Column mismatch between split and featureset for '{col}'.")
        return a.where(~a.isna(), b)
    if has_split:
        return split[split_col]
    if has_feat:
        return split[feat_col]
    raise ValueError(f"Column '{col}' not found after merge.")


def check_prob_columns(out: pd.DataFrame, prob_cols: List[str]) -> None:
    for c in prob_cols:
        if out[c].isna().any():
            n_bad = int(out[c].isna().sum())
            raise ValueError(f"Probability column {c} contains NaN: n_bad={n_bad}")
        if ((out[c] < 0) | (out[c] > 1)).any():
            bad = out.loc[(out[c] < 0) | (out[c] > 1), c].head(10).tolist()
            raise ValueError(f"Probability column {c} has values outside [0,1]. examples={bad}")


def check_pred_rows_match_split(out: pd.DataFrame, split_df: pd.DataFrame) -> None:
    grp_split = split_df.groupby(["protocol", "fold_id"], dropna=False).size().reset_index(name="n_split")
    grp_out = out.groupby(["protocol", "fold_id"], dropna=False).size().reset_index(name="n_pred")
    chk = grp_split.merge(grp_out, on=["protocol", "fold_id"], how="outer")

    if chk["n_split"].isna().any() or chk["n_pred"].isna().any():
        raise ValueError("Prediction coverage mismatch: some (protocol, fold_id) missing in split or preds.")

    bad = chk[chk["n_split"].astype(int) != chk["n_pred"].astype(int)]
    if len(bad) > 0:
        raise ValueError(
            "Prediction coverage mismatch per (protocol, fold_id): "
            + bad.to_dict(orient="records").__repr__()
        )


def rows_to_records(df: pd.DataFrame) -> List[Dict]:
    return [dict(r) for r in df.to_dict(orient="records")]


# ----------------------------
# Fold-wise prediction engines
# ----------------------------

def apply_single_stage_baselines(
    dff: pd.DataFrame,
    fs: FeatureSets,
    biome_col: str,
    enable_climo: bool,
    enable_memory_logit: bool,
    enable_climate_logit: bool,
    enable_vegenergy_logit: bool,
    climo_alpha: float,
    climo_beta: float,
    logit_C: float,
    logit_solver: str,
    logit_max_iter: int,
    logit_class_weight: Optional[str],
) -> Tuple[pd.DataFrame, Dict]:
    train_df = dff[dff["split_role"] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("Single-stage protocol requires split_role='train' rows.")

    pred_df = dff.copy()
    pred_df["fit_stage"] = "single"

    meta = {
        "fit_semantics": "single",
        "n_rows_train_fit": int(len(train_df)),
        "train_roles_used": ["train"],
    }

    if enable_climo:
        climo_table, fallback = fit_climo_with_fallback(
            train_df=train_df,
            alpha=climo_alpha,
            beta=climo_beta,
            biome_col=biome_col,
            month_col="month",
        )
        pred_df["p_climo_biome_month"] = predict_climo_with_fallback(
            df=pred_df,
            climo_table=climo_table,
            fallback=fallback,
            biome_col=biome_col,
            month_col="month",
        )
        meta["climo"] = {
            "table_rows": int(len(climo_table)),
            "global_p": float(fallback["global_p"]),
        }

    if enable_memory_logit:
        if len(fs.memory) == 0:
            raise ValueError("memory feature set is empty.")
        pred_df["p_memory_logit"] = fit_logit_predict(
            train_df=train_df,
            pred_df=pred_df,
            features=fs.memory,
            C=logit_C,
            solver=logit_solver,
            max_iter=logit_max_iter,
            class_weight=logit_class_weight,
            standardize=True,
        )

    if enable_climate_logit:
        if len(fs.climate) == 0:
            raise ValueError("climate feature set is empty.")
        pred_df["p_climate_logit"] = fit_logit_predict(
            train_df=train_df,
            pred_df=pred_df,
            features=fs.climate,
            C=logit_C,
            solver=logit_solver,
            max_iter=logit_max_iter,
            class_weight=logit_class_weight,
            standardize=True,
        )

    if enable_vegenergy_logit:
        if len(fs.vegenergy) == 0:
            raise ValueError("vegenergy feature set is empty.")
        pred_df["p_vegenergy_logit"] = fit_logit_predict(
            train_df=train_df,
            pred_df=pred_df,
            features=fs.vegenergy,
            C=logit_C,
            solver=logit_solver,
            max_iter=logit_max_iter,
            class_weight=logit_class_weight,
            standardize=True,
        )

    return pred_df, meta


def apply_p2prime_two_stage_baselines(
    dff: pd.DataFrame,
    fs: FeatureSets,
    biome_col: str,
    enable_climo: bool,
    enable_memory_logit: bool,
    enable_climate_logit: bool,
    enable_vegenergy_logit: bool,
    climo_alpha: float,
    climo_beta: float,
    logit_C: float,
    logit_solver: str,
    logit_max_iter: int,
    logit_class_weight: Optional[str],
) -> Tuple[pd.DataFrame, Dict]:
    inner_train = dff[dff["split_role"] == "inner_train"].copy()
    inner_val = dff[dff["split_role"] == "inner_val"].copy()
    outer_val = dff[dff["split_role"] == "val"].copy()
    outer_test = dff[dff["split_role"] == "test"].copy()

    if len(inner_train) == 0:
        raise ValueError("P2' requires inner_train rows.")
    if len(inner_val) == 0:
        raise ValueError("P2' requires inner_val rows.")
    if len(outer_val) == 0:
        raise ValueError("P2' requires val rows.")
    if len(outer_test) == 0:
        raise ValueError("P2' requires test rows.")

    # inner stage
    pred_inner = pd.concat([inner_train, inner_val], axis=0, ignore_index=True).copy()
    pred_inner["fit_stage"] = "inner"

    # outer stage: retrain on outer_train = inner_train ∪ inner_val
    outer_train = pd.concat([inner_train, inner_val], axis=0, ignore_index=True).copy()
    pred_outer = pd.concat([outer_val, outer_test], axis=0, ignore_index=True).copy()
    pred_outer["fit_stage"] = "outer"

    meta = {
        "fit_semantics": "p2prime_two_stage",
        "n_rows_train_fit_inner": int(len(inner_train)),
        "n_rows_train_fit_outer": int(len(outer_train)),
        "train_roles_used_inner": ["inner_train"],
        "train_roles_used_outer": ["inner_train", "inner_val"],
    }

    # ---- climo ----
    if enable_climo:
        climo_inner, fallback_inner = fit_climo_with_fallback(
            train_df=inner_train,
            alpha=climo_alpha,
            beta=climo_beta,
            biome_col=biome_col,
            month_col="month",
        )
        pred_inner["p_climo_biome_month"] = predict_climo_with_fallback(
            df=pred_inner,
            climo_table=climo_inner,
            fallback=fallback_inner,
            biome_col=biome_col,
            month_col="month",
        )

        climo_outer, fallback_outer = fit_climo_with_fallback(
            train_df=outer_train,
            alpha=climo_alpha,
            beta=climo_beta,
            biome_col=biome_col,
            month_col="month",
        )
        pred_outer["p_climo_biome_month"] = predict_climo_with_fallback(
            df=pred_outer,
            climo_table=climo_outer,
            fallback=fallback_outer,
            biome_col=biome_col,
            month_col="month",
        )

        meta["climo"] = {
            "inner_table_rows": int(len(climo_inner)),
            "outer_table_rows": int(len(climo_outer)),
            "inner_global_p": float(fallback_inner["global_p"]),
            "outer_global_p": float(fallback_outer["global_p"]),
        }

    # ---- memory ----
    if enable_memory_logit:
        if len(fs.memory) == 0:
            raise ValueError("memory feature set is empty.")
        pred_inner["p_memory_logit"] = fit_logit_predict(
            train_df=inner_train,
            pred_df=pred_inner,
            features=fs.memory,
            C=logit_C,
            solver=logit_solver,
            max_iter=logit_max_iter,
            class_weight=logit_class_weight,
            standardize=True,
        )
        pred_outer["p_memory_logit"] = fit_logit_predict(
            train_df=outer_train,
            pred_df=pred_outer,
            features=fs.memory,
            C=logit_C,
            solver=logit_solver,
            max_iter=logit_max_iter,
            class_weight=logit_class_weight,
            standardize=True,
        )

    # ---- climate ----
    if enable_climate_logit:
        if len(fs.climate) == 0:
            raise ValueError("climate feature set is empty.")
        pred_inner["p_climate_logit"] = fit_logit_predict(
            train_df=inner_train,
            pred_df=pred_inner,
            features=fs.climate,
            C=logit_C,
            solver=logit_solver,
            max_iter=logit_max_iter,
            class_weight=logit_class_weight,
            standardize=True,
        )
        pred_outer["p_climate_logit"] = fit_logit_predict(
            train_df=outer_train,
            pred_df=pred_outer,
            features=fs.climate,
            C=logit_C,
            solver=logit_solver,
            max_iter=logit_max_iter,
            class_weight=logit_class_weight,
            standardize=True,
        )

    # ---- veg/energy ----
    if enable_vegenergy_logit:
        if len(fs.vegenergy) == 0:
            raise ValueError("vegenergy feature set is empty.")
        pred_inner["p_vegenergy_logit"] = fit_logit_predict(
            train_df=inner_train,
            pred_df=pred_inner,
            features=fs.vegenergy,
            C=logit_C,
            solver=logit_solver,
            max_iter=logit_max_iter,
            class_weight=logit_class_weight,
            standardize=True,
        )
        pred_outer["p_vegenergy_logit"] = fit_logit_predict(
            train_df=outer_train,
            pred_df=pred_outer,
            features=fs.vegenergy,
            C=logit_C,
            solver=logit_solver,
            max_iter=logit_max_iter,
            class_weight=logit_class_weight,
            standardize=True,
        )

    pred_df = pd.concat([pred_inner, pred_outer], axis=0, ignore_index=True)
    return pred_df, meta


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--featureset_in", required=True)
    ap.add_argument("--split_in", required=True)
    ap.add_argument("--biome_col", default="BIOME")

    ap.add_argument("--enable_climo", action="store_true")
    ap.add_argument("--enable_memory_logit", action="store_true")
    ap.add_argument("--enable_climate_logit", action="store_true")
    ap.add_argument("--enable_vegenergy_logit", action="store_true")

    ap.add_argument("--climo_alpha", type=float, default=1.0)
    ap.add_argument("--climo_beta", type=float, default=1.0)

    ap.add_argument("--logit_C", type=float, default=0.1)
    ap.add_argument("--logit_solver", default="lbfgs")
    ap.add_argument("--logit_max_iter", type=int, default=2000)
    ap.add_argument("--logit_class_weight", default="none")

    ap.add_argument("--veg_drop_cov", action="store_true")
    ap.add_argument("--print_feature_sets", action="store_true")

    ap.add_argument("--out_pred_parquet", required=True)
    ap.add_argument("--out_manifest_json", required=True)
    ap.add_argument("--out_qc_csv", required=True)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_id = utc_run_id()
    logit_class_weight = normalize_class_weight(args.logit_class_weight)

    feat = pd.read_parquet(args.featureset_in)
    split = pd.read_parquet(args.split_in)

    assert_required_columns(feat, ["cell_id", "date"], "featureset")
    assert_required_columns(split, ["cell_id", "date", "y", "protocol", "fold_id", "split_role"], "split")

    feat["date"] = pd.to_datetime(feat["date"])
    split["date"] = pd.to_datetime(split["date"])

    # merge: y comes ONLY from split
    feat_keep = feat.drop(columns=[c for c in ["y"] if c in feat.columns]).copy()
    df = split.merge(feat_keep, on=["cell_id", "date"], how="left", suffixes=("_split", ""))

    # biome handling: allow BIOME from split or feature
    biome_col = args.biome_col
    split_biome = f"{biome_col}_split"
    feat_biome = biome_col

    if "month" not in df.columns:
        df["month"] = month_from_date(df["date"])

    # force y source = split only
    if "y" not in df.columns:
        raise ValueError("After merge, split y column missing.")
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype("int8")

    # biome recovery
    if biome_col not in df.columns:
        if split_biome in df.columns:
            df[biome_col] = df[split_biome]
        else:
            raise ValueError(f"biome_col={biome_col} not found in merged dataframe")

    if df[biome_col].isna().any():
        n_bad = int(df[biome_col].isna().sum())
        raise ValueError(f"biome_col={biome_col} contains NaN after merge. n_bad={n_bad}")

    fs = pick_feature_sets(df, veg_drop_cov=args.veg_drop_cov)

    if args.print_feature_sets:
        print("\n=== Step8_1 Feature Sets (final) ===")
        print(f"[climo]      n={len(fs.climo)}: {fs.climo}")
        print(f"[memory]     n={len(fs.memory)}: {fs.memory}")
        print(f"[climate]    n={len(fs.climate)}: {fs.climate}")
        print(f"[vegenergy]  n={len(fs.vegenergy)}: {fs.vegenergy}")
        print("===================================\n")

    out_parts = []
    fold_fit_manifest = []

    group_cols = ["protocol", "fold_id"]
    for (protocol, fold_id), dff in df.groupby(group_cols, dropna=False):
        dff = dff.copy().reset_index(drop=True)

        is_p2prime = str(protocol).startswith("P2_prime_nested_")

        if is_p2prime:
            pred_fold, fold_meta = apply_p2prime_two_stage_baselines(
                dff=dff,
                fs=fs,
                biome_col=biome_col,
                enable_climo=args.enable_climo,
                enable_memory_logit=args.enable_memory_logit,
                enable_climate_logit=args.enable_climate_logit,
                enable_vegenergy_logit=args.enable_vegenergy_logit,
                climo_alpha=args.climo_alpha,
                climo_beta=args.climo_beta,
                logit_C=args.logit_C,
                logit_solver=args.logit_solver,
                logit_max_iter=args.logit_max_iter,
                logit_class_weight=logit_class_weight,
            )
            fold_fit_manifest.append({
                "protocol": str(protocol),
                "fold_id": int(fold_id),
                "fit_semantics": "p2prime_two_stage",
                "n_rows_train_fit_inner": int(fold_meta["n_rows_train_fit_inner"]),
                "n_rows_train_fit_outer": int(fold_meta["n_rows_train_fit_outer"]),
            })
        else:
            pred_fold, fold_meta = apply_single_stage_baselines(
                dff=dff,
                fs=fs,
                biome_col=biome_col,
                enable_climo=args.enable_climo,
                enable_memory_logit=args.enable_memory_logit,
                enable_climate_logit=args.enable_climate_logit,
                enable_vegenergy_logit=args.enable_vegenergy_logit,
                climo_alpha=args.climo_alpha,
                climo_beta=args.climo_beta,
                logit_C=args.logit_C,
                logit_solver=args.logit_solver,
                logit_max_iter=args.logit_max_iter,
                logit_class_weight=logit_class_weight,
            )
            fold_fit_manifest.append({
                "protocol": str(protocol),
                "fold_id": int(fold_id),
                "fit_semantics": "single",
                "n_rows_train_fit": int(fold_meta["n_rows_train_fit"]),
            })

        pred_fold["run_id"] = run_id
        out_parts.append(pred_fold)

    out = pd.concat(out_parts, axis=0, ignore_index=True)

    keep_cols = ["cell_id", "date", "y", "protocol", "fold_id", "split_role", "fit_stage", "run_id"]
    if biome_col in out.columns:
        keep_cols.append(biome_col)

    prob_cols = [c for c in out.columns if c.startswith("p_")]
    out = out[keep_cols + prob_cols].copy()

    # final safety checks
    check_pred_rows_match_split(out=out, split_df=split)
    check_prob_columns(out=out, prob_cols=prob_cols)

    # write preds
    ensure_dir(args.out_pred_parquet)
    out.to_parquet(args.out_pred_parquet, index=False)

    # QC
    rows = []
    for keys, g in out.groupby(["protocol", "fold_id", "split_role"], dropna=False):
        prot, fold, role = keys
        y = g["y"].to_numpy(dtype=int)
        rec = {
            "protocol": prot,
            "fold_id": int(fold),
            "split_role": role,
            "n": int(len(g)),
            "pos": int((y == 1).sum()),
            "y_rate": float(np.mean(y)) if len(y) else float("nan"),
        }
        for pc in prob_cols:
            yp = g[pc].to_numpy(dtype=float)
            rec[f"{pc}_auprc"] = auprc(y, yp)
            rec[f"{pc}_brier"] = brier(y, yp)
        rows.append(rec)

    qc = pd.DataFrame(rows).sort_values(["protocol", "fold_id", "split_role"]).reset_index(drop=True)
    ensure_dir(args.out_qc_csv)
    qc.to_csv(args.out_qc_csv, index=False)

    # manifest checks summary
    pred_check = {
        "coverage_matches_split_per_protocol_fold": True,
        "prob_cols_no_nan": True,
        "prob_cols_in_unit_interval": True,
    }

    manifest = {
        "run_id": run_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "featureset_in": args.featureset_in,
        "featureset_hash": file_hash(args.featureset_in),
        "split_in": args.split_in,
        "split_hash": file_hash(args.split_in),
        "out_pred_parquet": args.out_pred_parquet,
        "out_pred_hash": file_hash(args.out_pred_parquet),
        "out_qc_csv": args.out_qc_csv,

        "fit_granularity": "protocol x fold",
        "y_source": "split file only",
        "biome_col_used": biome_col,

        "enabled": {
            "climo": bool(args.enable_climo),
            "memory_logit": bool(args.enable_memory_logit),
            "climate_logit": bool(args.enable_climate_logit),
            "vegenergy_logit": bool(args.enable_vegenergy_logit),
        },

        "climo": {
            "alpha": float(args.climo_alpha),
            "beta": float(args.climo_beta),
            "hierarchical_fallback": ["BIOME x month", "BIOME", "global"],
        },

        "logit": {
            "C": float(args.logit_C),
            "solver": str(args.logit_solver),
            "max_iter": int(args.logit_max_iter),
            "class_weight": ("none" if logit_class_weight is None else logit_class_weight),
        },

        "veg_feature_policy": {
            "veg_drop_missing": True,
            "veg_drop_cov": bool(args.veg_drop_cov),
        },

        "feature_sets": asdict(fs),
        "prob_cols": prob_cols,
        "fold_fit_manifest": fold_fit_manifest,
        "prediction_checks": pred_check,
    }

    ensure_dir(args.out_manifest_json)
    with open(args.out_manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("✅ Step8_1 baselines done (final rigorous version).")
    print(f"- run_id: {run_id}")
    print(f"- preds: {args.out_pred_parquet}")
    print(f"- qc: {args.out_qc_csv}")
    print(f"- manifest: {args.out_manifest_json}")
    print(f"- prob_cols: {prob_cols}")
    print(f"- fit_granularity: protocol x fold")
    print(f"- y_source: split file only")
    print(f"- P2prime_semantics: inner_train->inner_val ; (inner_train+inner_val)->val/test")


if __name__ == "__main__":
    main()