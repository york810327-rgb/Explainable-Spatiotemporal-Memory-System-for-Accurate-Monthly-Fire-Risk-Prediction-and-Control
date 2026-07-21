#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6 Feature Engineering (Memory / Rolling / Fire-history) — F1 core gain

Compatible with Step5 v1.0.6:
- Supports CV-only spatial block columns: grid10km_id / spatial_block_id / tile_id
- These CV-only cols are NEVER used for any feature derivation and NEVER coerced to numeric.
- They are preserved in output as-is (pass-through).

Key fix (critical):
- fire_count_12m rolling is computed via explicit groupby(cell_id) rolling
  to eliminate any risk of cross-cell rolling contamination.

Design decisions remain:
- leakage-safe within-cell only (lags/rolling/fire history)
- NO anomaly/climatology here (Step8 fold-wise only)

Outputs:
- Default: NO IN-PLACE OVERWRITE (hard fail if output exists)
- Optionally allow overwrite with --overwrite (and optional backup)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


# -------------------------
# CV-only / meta cols (Step5 v1.0.6)
# -------------------------
CV_ONLY_BLOCK_COLS = {"grid10km_id", "spatial_block_id", "tile_id"}


# -------------------------
# Utilities
# -------------------------

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def file_hash(path: str, algo: str = "sha256", chunk_size: int = 2**20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"[{name}] missing required columns: {miss}")

def _safe_numeric(s: pd.Series) -> pd.Series:
    # IMPORTANT: caller must avoid passing CV-only cols here.
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def coerce_date(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "date" not in df.columns:
        raise KeyError(f"[{name}] missing column: date")
    out = df.copy()
    if not np.issubdtype(out["date"].dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        bad = out.loc[out["date"].isna()].head(10).to_dict(orient="records")
        raise RuntimeError(f"[{name}] date coercion produced NaT. examples={bad}")
    if (out["date"].dt.day != 1).any():
        bad = out.loc[out["date"].dt.day != 1, ["cell_id", "date"]].head(10).to_dict(orient="records")
        raise RuntimeError(f"[{name}] date must be month-start (day==1). examples={bad}")
    if "year" not in out.columns:
        out["year"] = out["date"].dt.year.astype(int)
    if "month" not in out.columns:
        out["month"] = out["date"].dt.month.astype(int)
    return out

def atomic_write_parquet(df: pd.DataFrame, out_path: str, *, compression: str = "snappy") -> None:
    """Write parquet atomically: write to temp file then os.replace."""
    out_path = str(out_path)
    out_dir = os.path.dirname(out_path)
    ensure_dir(out_dir)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_step6_", suffix=".parquet", dir=out_dir if out_dir else None)
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow", compression=compression)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def assert_panel_integrity(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Hard integrity: no duplicated (cell_id,date); date strictly increasing within each cell_id."""
    require_cols(df, ["cell_id", "date"], f"{name}.integrity")

    x = df.sort_values(["cell_id", "date"]).reset_index(drop=True)

    dup_n = int(x.duplicated(subset=["cell_id", "date"]).sum())
    if dup_n > 0:
        ex = x.loc[x.duplicated(subset=["cell_id", "date"]), ["cell_id", "date"]].head(10).to_dict(orient="records")
        raise RuntimeError(f"[{name}] duplicated (cell_id,date): n={dup_n}. examples={ex}")

    diffs = x.groupby("cell_id", sort=False)["date"].diff()
    bad_n = int((diffs.notna() & (diffs <= pd.Timedelta(0))).sum())
    if bad_n > 0:
        bad = x.loc[diffs.notna() & (diffs <= pd.Timedelta(0)), ["cell_id", "date"]].head(20).to_dict(orient="records")
        raise RuntimeError(f"[{name}] date not strictly increasing within cell_id. n={bad_n}. examples={bad}")

    return x

def assert_y_binary(df: pd.DataFrame, name: str, y_col: str = "y") -> None:
    require_cols(df, [y_col], f"{name}.ycheck")
    y = _safe_numeric(df[y_col])
    if y.isna().any():
        ex = df.loc[y.isna(), ["cell_id", "date", y_col]].head(10).to_dict(orient="records") if "cell_id" in df.columns and "date" in df.columns else []
        raise RuntimeError(f"[{name}] y has NaN. examples={ex}")
    vals = set(pd.unique(y))
    if not vals.issubset({0, 1}):
        raise RuntimeError(f"[{name}] y must be binary {{0,1}}. values={sorted(list(vals))[:20]}")

def assert_no_suspicious_future_cols(df: pd.DataFrame, name: str) -> None:
    """
    Very light name-based guard. Not a substitute for Step5.5 audit,
    but helps catch accidental wrong inputs.
    """
    bad = []
    pat = re.compile(r"(?:\blead\b|\bfuture\b|\bt\+|\blead\d+|\blag-|\bnext\b)", re.IGNORECASE)
    for c in df.columns:
        if pat.search(str(c)):
            bad.append(c)
    if bad:
        raise RuntimeError(f"[{name}] suspicious column names detected (possible future/lead leakage): {bad[:30]}")

def enforce_roll_specs_lag1(roll_specs: List[Tuple[str, str]], panel_name: str) -> None:
    bad = [c for c, _ in roll_specs if not str(c).endswith("_lag1")]
    if bad:
        raise RuntimeError(f"[{panel_name}] rolling specs must be *_lag1 for leakage safety. bad={bad}")

def maybe_overwrite_path(path: str, overwrite: bool, backup: bool, what: str) -> None:
    if not path:
        return
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"[NO OVERWRITE] {what} already exists: {path}")
        if backup:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            bak = f"{path}.bak.{ts}"
            ensure_dir(os.path.dirname(bak))
            shutil.move(path, bak)

def month_diff(a: pd.Series) -> pd.Series:
    """Compute month difference between consecutive dates (assumes datetime64)."""
    y = a.dt.year.astype(int)
    m = a.dt.month.astype(int)
    ym = y * 12 + m
    return ym.diff()

def check_monthly_continuity(
    df_sorted: pd.DataFrame,
    name: str,
    mode: str = "off",  # off|warn|fail
    max_examples: int = 10,
) -> Tuple[int, str]:
    """Check per cell whether month increments are exactly 1."""
    if mode == "off":
        return 0, ""

    dif = df_sorted.groupby("cell_id", sort=False)["date"].apply(month_diff).reset_index(level=0, drop=True)
    gap_mask = dif.notna() & (dif != 1)

    gap_rows_n = int(gap_mask.sum())
    gap_examples = ""
    if gap_rows_n > 0:
        ex = df_sorted.loc[gap_mask, ["cell_id", "date"]].head(max_examples).to_dict(orient="records")
        gap_examples = "|".join([str(x) for x in ex])

        msg = f"[{name}] monthly continuity violated: gap_rows_n={gap_rows_n}. examples={ex[:5]}"
        if mode == "fail":
            raise RuntimeError(msg)
        else:
            print("⚠️  " + msg)

    return gap_rows_n, gap_examples


# -------------------------
# Step6.0 CV-only guard
# -------------------------

def assert_cv_only_cols_passthrough(df: pd.DataFrame, name: str) -> None:
    """Ensure CV-only cols exist but never get touched; just sanity check dtype/NA."""
    hits = [c for c in CV_ONLY_BLOCK_COLS if c in df.columns]
    if not hits:
        return
    # Nothing hard-failing; just ensure they are not all-null if present (warn)
    for c in hits:
        na_rate = float(df[c].isna().mean())
        if na_rate > 0.5:
            print(f"⚠️  [{name}] CV-only col '{c}' has high NaN rate: {na_rate:.3f} (check Step5 v1.0.6 merge).")


# -------------------------
# Step6.1 Fire-history memory features
# -------------------------

def _groupby_rolling_sum(series: pd.Series, group: pd.Series, window: int) -> pd.Series:
    """Explicit leakage-safe rolling sum within groups."""
    r = (
        series.groupby(group, sort=False)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return r

def add_fire_history_features(
    df_sorted: pd.DataFrame,
    y_col: str = "y",
    cap_months: int = 60,
    add_has_fire_history: bool = True,
    add_ever_burned_before: bool = True,
    verify_firecount_rolling: bool = True,
    verify_k_cells: int = 50,
    verify_seed: int = 0,
) -> pd.DataFrame:
    """Per cell_id memory features: y_lag1/y_lag12/fire_count_12m/months_since_last_fire/fire_tslf."""
    require_cols(df_sorted, ["cell_id", "date", y_col], "firehist")

    out = df_sorted.copy()

    out[y_col] = _safe_numeric(out[y_col]).fillna(0).astype(np.int8)

    g = out.groupby("cell_id", sort=False)
    out["y_lag1"] = g[y_col].shift(1).fillna(0).astype(np.int8)
    out["y_lag12"] = g[y_col].shift(12).fillna(0).astype(np.int8)

    out["fire_count_12m"] = (
        _groupby_rolling_sum(out["y_lag1"].astype(np.int16), out["cell_id"], window=12)
        .fillna(0)
        .astype(np.int8)
    )

    # months_since_last_fire based on y_lag1 (past-only)
    ms = np.zeros(len(out), dtype=np.int32)
    for _, idx in g.indices.items():
        ii = np.asarray(idx, dtype=np.int64)
        ys = out.loc[ii, "y_lag1"].to_numpy(dtype=np.int32, copy=False)

        last_fire_pos = -1
        for k in range(len(ii)):
            if ys[k] == 1:
                last_fire_pos = k
                ms[ii[k]] = 0
            else:
                if last_fire_pos < 0:
                    ms[ii[k]] = cap_months
                else:
                    ms[ii[k]] = min(cap_months, k - last_fire_pos)

    out["months_since_last_fire"] = ms.astype(np.int16)
    out["fire_tslf"] = out["months_since_last_fire"].astype(np.int16)

    if add_has_fire_history:
        out["has_fire_history"] = g[y_col].transform("max").astype(np.int8)

    if add_ever_burned_before:
        s = g[y_col].shift(1).fillna(0)
        out["ever_burned_before"] = s.groupby(out["cell_id"], sort=False).cumsum().gt(0).astype(np.int8)

    # HARD ASSERTS
    for c in ["y_lag1", "y_lag12"]:
        vals = set(pd.unique(out[c]))
        if not vals.issubset({0, 1}):
            raise RuntimeError(f"[firehist] {c} not binary. values={sorted(list(vals))[:20]}")

    fc = pd.to_numeric(out["fire_count_12m"], errors="coerce")
    fc_min = int(fc.min())
    fc_max = int(fc.max())
    if not (0 <= fc_min and fc_max <= 12):
        raise RuntimeError(f"[firehist] fire_count_12m out of range: min={fc_min}, max={fc_max} (expected [0,12])")

    ms_s = pd.to_numeric(out["months_since_last_fire"], errors="coerce")
    ms_min = int(ms_s.min())
    ms_max = int(ms_s.max())
    if not (0 <= ms_min and ms_max <= cap_months):
        raise RuntimeError(
            f"[firehist] months_since_last_fire out of range: min={ms_min}, max={ms_max} (expected [0,{cap_months}])"
        )

    # Extra verification for rolling
    if verify_firecount_rolling and len(out) > 0:
        rng = np.random.default_rng(verify_seed)
        cells = out["cell_id"].unique()
        k = min(int(verify_k_cells), len(cells))
        if k > 0:
            pick = rng.choice(cells, size=k, replace=False)
            sub = out[out["cell_id"].isin(pick)].sort_values(["cell_id", "date"]).copy()
            exp = (
                sub["y_lag1"].astype(np.int16)
                .groupby(sub["cell_id"], sort=False)
                .rolling(window=12, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
                .astype(np.int16)
            )
            got = sub["fire_count_12m"].astype(np.int16)
            if not np.all(exp.values == got.values):
                m = (exp.values != got.values)
                ex = sub.loc[m, ["cell_id", "date", "y_lag1", "fire_count_12m"]].head(10).to_dict(orient="records")
                raise RuntimeError(f"[firehist] fire_count_12m verification failed (groupby rolling mismatch). examples={ex}")

    return out


# -------------------------
# Step6.2 Rolling accumulation (within-cell, safe)
# -------------------------

def add_roll_features(
    df_sorted: pd.DataFrame,
    roll_window: int,
    roll_specs: List[Tuple[str, str]],
    *,
    enforce_lag1: bool = False,
    panel_name: str = "panel",
) -> pd.DataFrame:
    """
    roll_specs: list of (col_name, agg) where agg in {"sum","mean","max","min"}.
    Produces: f"{col}_roll{roll_window}m_{agg}"
    """
    require_cols(df_sorted, ["cell_id", "date"], "rolling")
    if enforce_lag1:
        enforce_roll_specs_lag1(roll_specs, panel_name)

    out = df_sorted.copy()

    for col, agg in roll_specs:
        if col not in out.columns:
            continue

        # guard: never operate on CV-only cols
        if col in CV_ONLY_BLOCK_COLS:
            raise RuntimeError(f"[{panel_name}] roll spec includes CV-only col '{col}' (forbidden).")

        out[col] = _safe_numeric(out[col])

        g = out.groupby("cell_id", sort=False)[col]
        if agg == "sum":
            r = g.rolling(roll_window, min_periods=1).sum()
        elif agg == "mean":
            r = g.rolling(roll_window, min_periods=1).mean()
        elif agg == "max":
            r = g.rolling(roll_window, min_periods=1).max()
        elif agg == "min":
            r = g.rolling(roll_window, min_periods=1).min()
        else:
            raise ValueError(f"Unknown agg={agg}")

        out[f"{col}_roll{roll_window}m_{agg}"] = _safe_numeric(r.reset_index(level=0, drop=True))

    return out


# -------------------------
# Alignment sanity checks
# -------------------------

def spotcheck_y_lag1_consistency(df_sorted: pd.DataFrame, k: int, seed: int = 0, name: str = "panel") -> None:
    """Spot-check that y_lag1(t) == y(t-1) within sampled cells."""
    if k <= 0:
        return
    require_cols(df_sorted, ["cell_id", "date", "y", "y_lag1"], f"{name}.spotcheck")

    rng = np.random.default_rng(seed)
    cells = df_sorted["cell_id"].unique()
    if len(cells) == 0:
        return
    k_cells = min(k, len(cells))
    pick = rng.choice(cells, size=k_cells, replace=False)

    bad_total = 0
    examples: List[Dict[str, Any]] = []

    for cid in pick:
        sub = df_sorted[df_sorted["cell_id"] == cid].sort_values("date")
        y_prev = sub["y"].shift(1).fillna(0).astype(int)
        ok = (sub["y_lag1"].astype(int).values == y_prev.values)
        if not ok.all():
            bad_idx = np.where(~ok)[0][:5]
            bad_total += int((~ok).sum())
            for bi in bad_idx:
                examples.append({
                    "cell_id": cid,
                    "date": str(sub.iloc[bi]["date"].date()),
                    "y": int(sub.iloc[bi]["y"]),
                    "y_lag1": int(sub.iloc[bi]["y_lag1"]),
                    "expected_y_lag1": int(y_prev.iloc[bi]),
                })

    if bad_total > 0:
        raise RuntimeError(f"[{name}] spotcheck failed: y_lag1 != y(t-1). bad_n={bad_total}. examples={examples[:10]}")

def spotcheck_lead1_vs_nowcast_alignment(
    lead1_sorted: pd.DataFrame,
    now_sorted: pd.DataFrame,
    k: int,
    seed: int = 0,
    shift_mode: str = "auto",   # off|auto|plus1|same|minus1
    min_match: float = 0.98,
) -> Dict[str, object]:
    """Spot-check lead1.y aligns to nowcast.y under a month shift."""
    if k <= 0 or shift_mode == "off":
        return {"alignment_mode": "off", "checked_cells": 0}

    require_cols(lead1_sorted, ["cell_id", "date", "y"], "lead1.align")
    require_cols(now_sorted, ["cell_id", "date", "y"], "nowcast.align")

    rng = np.random.default_rng(seed)
    cells = np.intersect1d(lead1_sorted["cell_id"].unique(), now_sorted["cell_id"].unique())
    if len(cells) == 0:
        return {"alignment_mode": shift_mode, "checked_cells": 0}

    k_cells = min(k, len(cells))
    pick = rng.choice(cells, size=k_cells, replace=False)

    now_key = pd.MultiIndex.from_frame(now_sorted[["cell_id", "date"]])
    now_y = pd.Series(now_sorted["y"].astype(int).values, index=now_key)

    def _check_for_shift(month_shift: int) -> Tuple[float, int, List[dict]]:
        total = 0
        bad = 0
        examples: List[dict] = []

        for cid in pick:
            subL = lead1_sorted[lead1_sorted["cell_id"] == cid].sort_values("date")

            if month_shift == 0:
                d_ref = subL["date"]
            elif month_shift > 0:
                d_ref = subL["date"] + pd.offsets.MonthBegin(month_shift)
            else:
                d_ref = subL["date"] - pd.offsets.MonthBegin(abs(month_shift))

            d_ref = d_ref.dt.to_period("M").dt.to_timestamp()
            keys = pd.MultiIndex.from_arrays([np.full(len(subL), cid), d_ref])

            exp = now_y.reindex(keys)
            mask = exp.notna()
            if not mask.any():
                continue

            got = subL.loc[mask.values, "y"].astype(int).values
            want = exp.loc[mask].astype(int).values

            total += len(got)
            bad_mask = (got != want)
            bad += int(bad_mask.sum())

            if bad_mask.any() and len(examples) < 10:
                sub_chk = subL.loc[mask.values].reset_index(drop=True)
                want_chk = exp.loc[mask].reset_index(drop=True).astype(int)
                idxs = np.where(bad_mask)[0][: (10 - len(examples))]
                for bi in idxs:
                    examples.append({
                        "cell_id": cid,
                        "lead1_date": str(sub_chk.iloc[bi]["date"].date()),
                        "lead1_y": int(sub_chk.iloc[bi]["y"]),
                        "nowcast_date_used": str(d_ref.loc[subL.index[mask.values]].reset_index(drop=True).iloc[bi].date()),
                        "nowcast_y": int(want_chk.iloc[bi]),
                        "month_shift": month_shift,
                    })

        match = float("nan") if total == 0 else 1.0 - (bad / total)
        return match, total, examples

    candidates = {"plus1": +1, "same": 0, "minus1": -1}
    results: Dict[str, Dict[str, Any]] = {}
    for m, s in candidates.items():
        match, total, ex = _check_for_shift(s)
        results[m] = {"match_rate": match, "n_compared": total, "examples": ex}

    if shift_mode == "auto":
        best = max(results.items(), key=lambda kv: (-1 if np.isnan(kv[1]["match_rate"]) else kv[1]["match_rate"]))
        chosen_mode = best[0]
    else:
        chosen_mode = shift_mode

    chosen = results[chosen_mode]
    summary = {
        "alignment_mode": chosen_mode,
        "checked_cells": int(k_cells),
        "plus1_match_rate": results["plus1"]["match_rate"],
        "same_match_rate": results["same"]["match_rate"],
        "minus1_match_rate": results["minus1"]["match_rate"],
        "chosen_match_rate": chosen["match_rate"],
        "chosen_n_compared": chosen["n_compared"],
    }

    if shift_mode in ["plus1", "same", "minus1"]:
        mr = chosen["match_rate"]
        if (not np.isnan(mr)) and (mr < min_match):
            raise RuntimeError(
                f"[alignment] lead1_label_shift={shift_mode} failed: match_rate={mr:.4f} < {min_match}. "
                f"examples={chosen['examples']}"
            )

    if shift_mode == "auto":
        print(
            f"ℹ️  [alignment-auto] plus1={summary['plus1_match_rate']}, same={summary['same_match_rate']}, "
            f"minus1={summary['minus1_match_rate']} -> chosen={chosen_mode} ({summary['chosen_match_rate']})"
        )

    return summary


# -------------------------
# QC helpers
# -------------------------

def qc_bad_mask_firehist(df: pd.DataFrame, cap: int) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    if "fire_count_12m" in df.columns:
        fc = pd.to_numeric(df["fire_count_12m"], errors="coerce")
        mask = mask | fc.isna() | (fc < 0) | (fc > 12)
    if "months_since_last_fire" in df.columns:
        ms = pd.to_numeric(df["months_since_last_fire"], errors="coerce")
        mask = mask | ms.isna() | (ms < 0) | (ms > cap)
    if "y_lag1" in df.columns:
        yl1 = pd.to_numeric(df["y_lag1"], errors="coerce")
        mask = mask | yl1.isna() | (~yl1.isin([0, 1]))
    if "y_lag12" in df.columns:
        yl12 = pd.to_numeric(df["y_lag12"], errors="coerce")
        mask = mask | yl12.isna() | (~yl12.isin([0, 1]))

    return mask

def missing_rate_mean(df: pd.DataFrame, cols: List[str]) -> float:
    cols2 = [c for c in cols if c in df.columns]
    if not cols2:
        return float("nan")
    return float(np.mean([df[c].isna().mean() for c in cols2]))

def qc_missing_rate_cols(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in cols:
        if c in df.columns:
            out[f"missrate_{c}"] = float(df[c].isna().mean())
    return out

def qc_panel_summary(
    df: pd.DataFrame,
    panel_name: str,
    derived_cols: List[str],
    base_cols: List[str],
    firehist_cap: int,
    gap_rows_n: int,
    gap_examples: str,
    lag_cols: List[str],
    rolling_cols: List[str],
    base_cov_cols_for_missing: Optional[List[str]] = None,
    alignment_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    y = _safe_numeric(df["y"]) if "y" in df.columns else pd.Series(dtype=float)

    present = [c for c in derived_cols if c in df.columns]
    missing_rates = [float(df[c].isna().mean()) for c in present] if present else []

    base_set = set(base_cols)
    new_cols = sorted([c for c in df.columns if c not in base_set])

    fire_count_min = int(pd.to_numeric(df["fire_count_12m"], errors="coerce").min()) if "fire_count_12m" in df.columns else -1
    fire_count_max = int(pd.to_numeric(df["fire_count_12m"], errors="coerce").max()) if "fire_count_12m" in df.columns else -1
    tslf_min = int(pd.to_numeric(df["months_since_last_fire"], errors="coerce").min()) if "months_since_last_fire" in df.columns else -1
    tslf_max = int(pd.to_numeric(df["months_since_last_fire"], errors="coerce").max()) if "months_since_last_fire" in df.columns else -1

    bad_mask = qc_bad_mask_firehist(df, cap=firehist_cap)
    bad_rows_n = int(bad_mask.sum())

    bad_examples = ""
    if bad_rows_n > 0:
        cols = [c for c in ["cell_id", "date", "y", "y_lag1", "y_lag12", "fire_count_12m", "months_since_last_fire"] if c in df.columns]
        bad_examples = "|".join([str(x) for x in df.loc[bad_mask, cols].head(5).to_dict(orient="records")])

    cv_cols_present = sorted([c for c in CV_ONLY_BLOCK_COLS if c in df.columns])

    row: Dict[str, object] = {
        "panel": panel_name,
        "n_rows": int(df.shape[0]),
        "n_cells": int(df["cell_id"].nunique()) if "cell_id" in df.columns else int(-1),
        "date_min": str(df["date"].min().date()) if "date" in df.columns else "",
        "date_max": str(df["date"].max().date()) if "date" in df.columns else "",
        "y_rate": float(y.mean()) if len(y) else float("nan"),

        "firehist_cap": int(firehist_cap),
        "fire_count_12m_min": fire_count_min,
        "fire_count_12m_max": fire_count_max,
        "months_since_last_fire_min": tslf_min,
        "months_since_last_fire_max": tslf_max,
        "bad_rows_n": bad_rows_n,
        "bad_rows_examples": bad_examples,

        "gap_rows_n": int(gap_rows_n),
        "gap_rows_examples": gap_examples,

        "lag_cols_missing_rate_mean": missing_rate_mean(df, lag_cols),
        "rolling_cols_missing_rate_mean": missing_rate_mean(df, rolling_cols),

        "derived_cols_present_n": int(len(present)),
        "derived_missing_rate_mean": float(np.mean(missing_rates)) if missing_rates else float("nan"),
        "new_cols_n": int(len(new_cols)),
        "new_cols": "|".join(new_cols[:200]),

        "cv_only_cols_present": "|".join(cv_cols_present),
    }

    if base_cov_cols_for_missing:
        row.update(qc_missing_rate_cols(df, base_cov_cols_for_missing))
        mr_any = missing_rate_mean(df, base_cov_cols_for_missing)
        row["missrate_basecov_mean"] = float(mr_any)
        row["any_missing_in_base_covariates"] = int(bool((not np.isnan(mr_any)) and mr_any > 0))

    if alignment_meta:
        row["lead1_label_shift"] = str(alignment_meta.get("alignment_mode", ""))
        row["align_plus1_match_rate"] = alignment_meta.get("plus1_match_rate", np.nan)
        row["align_same_match_rate"] = alignment_meta.get("same_match_rate", np.nan)
        row["align_minus1_match_rate"] = alignment_meta.get("minus1_match_rate", np.nan)
        row["align_chosen_match_rate"] = alignment_meta.get("chosen_match_rate", np.nan)
        row["align_chosen_n_compared"] = alignment_meta.get("chosen_n_compared", np.nan)
        row["align_checked_cells"] = alignment_meta.get("checked_cells", np.nan)

    return row


# -------------------------
# Config: minimal derived set (NO anomaly)
# -------------------------

LEAD1_ROLL_SPECS_3M = [
    ("P_sum_mon_lag1", "sum"),
    ("PET_sum_mon_lag1", "sum"),
    ("VPD_mean_mon_lag1", "mean"),
    ("NDVI_mean_mon_lag1", "mean"),
]

NOWCAST_ROLL_SPECS_3M = [
    ("P_sum_mon", "sum"),
    ("PET_sum_mon", "sum"),
    ("VPD_mean_mon", "mean"),
    ("NDVI_mean_mon", "mean"),
]


@dataclass
class RunOutputs:
    run_id: str
    lead1_in: str
    nowcast_in: str
    lead1_out: str
    nowcast_out: str
    report: str
    qc_out: str
    qc_badrows_out: str
    hash_algo: str
    in_hashes: Dict[str, str]
    out_hashes: Dict[str, str]
    lead1_label_shift: str
    alignment_summary: Optional[Dict[str, object]]
    lead1_date_min: str
    lead1_date_max: str
    nowcast_date_min: str
    nowcast_date_max: str
    cv_only_cols_present_lead1: str
    cv_only_cols_present_nowcast: str


def write_report_md(out: RunOutputs, cap_tslf: int) -> None:
    lines: List[str] = []
    lines.append("# Step6 Feature Engineering Report (F1)\n\n")
    lines.append(f"- run_id: `{out.run_id}`\n")
    lines.append(f"- utc_time: `{datetime.now(timezone.utc).isoformat()}`\n")
    lines.append(f"- hash_algo: `{out.hash_algo}`\n\n")

    lines.append("## Inputs\n")
    for k, v in out.in_hashes.items():
        lines.append(f"- {k}: `{v}`\n")

    lines.append("\n## Outputs\n")
    for k, v in out.out_hashes.items():
        lines.append(f"- {k}: `{v}`\n")

    lines.append("\n## Panel coverage\n")
    lines.append(f"- lead1 date range: `{out.lead1_date_min}` → `{out.lead1_date_max}`\n")
    if out.nowcast_in:
        lines.append(f"- nowcast date range: `{out.nowcast_date_min}` → `{out.nowcast_date_max}`\n")
        if out.lead1_date_min != out.nowcast_date_min:
            lines.append("\n### Note on left-censoring / start-month mismatch\n")
            lines.append(
                "- lead1 starts later than nowcast. This is expected when the Step5 construction drops the first month "
                "due to lagged predictors/label setup. Interpret early-month fire-history features with left-censoring in mind.\n"
            )

    lines.append("\n## CV-only columns (pass-through)\n")
    lines.append(f"- lead1 cv-only present: `{out.cv_only_cols_present_lead1}`\n")
    if out.nowcast_in:
        lines.append(f"- nowcast cv-only present: `{out.cv_only_cols_present_nowcast}`\n")
    lines.append("- CV-only cols are kept for Step7 P2 split only; NEVER used as predictors; Step6 does not touch them.\n")

    lines.append("\n## Target/predictor timing convention\n")
    lines.append(f"- `lead1_label_shift`: `{out.lead1_label_shift}`\n")
    lines.append("- This run is documented as: target `y(t)` with predictors restricted to information up to `t-1`.\n")

    lines.append("\n## Derived features (Step6, leakage-safe)\n")
    lines.append("### Fire-history memory (per cell)\n")
    lines.append("- `y_lag1`: y(t-1)\n")
    lines.append("- `y_lag12`: y(t-12)\n")
    lines.append("- `fire_count_12m`: sum(y(t-1)...y(t-12)) computed via explicit groupby(cell_id) rolling\n")
    lines.append(f"- `months_since_last_fire`: months since last y==1 up to t-1 (cap={cap_tslf})\n")
    lines.append("- `fire_tslf`: alias of months_since_last_fire\n")
    lines.append("- `has_fire_history` (optional): whether the cell ever burned in observed period\n")
    lines.append("- `ever_burned_before` (optional): whether the cell burned before time t (past-only)\n\n")

    lines.append("### Rolling (per cell)\n")
    lines.append("- 3-month rolling for a minimal set (P/PET/VPD/NDVI)\n")
    lines.append("- For lead1, rolling is applied to `*_lag1` columns (<=t-1 by construction)\n\n")

    lines.append("## Explicitly NOT included\n")
    lines.append("- NO anomaly/climatology here. Step8 fold-wise only.\n\n")

    lines.append("## Hard QC invariants\n")
    lines.append("- `fire_count_12m` in [0,12]\n")
    lines.append(f"- `months_since_last_fire` in [0,{cap_tslf}]\n")
    lines.append("- `y_lag1`, `y_lag12` in {0,1}\n")
    lines.append("- Any violation -> write `qc_step6_badrows.csv` and hard fail\n\n")

    lines.append("## Alignment check (optional)\n")
    lines.append("- Optional alignment sanity check: lead1.y aligns to nowcast.y under `--lead1_label_shift`.\n")
    if out.alignment_summary:
        s = out.alignment_summary
        lines.append(f"- checked_cells: `{s.get('checked_cells')}`\n")
        lines.append(f"- plus1_match_rate: `{s.get('plus1_match_rate')}`\n")
        lines.append(f"- same_match_rate: `{s.get('same_match_rate')}`\n")
        lines.append(f"- minus1_match_rate: `{s.get('minus1_match_rate')}`\n")
        lines.append(f"- chosen_mode: `{s.get('alignment_mode')}`\n")
        lines.append(f"- chosen_match_rate: `{s.get('chosen_match_rate')}`\n")
        lines.append(f"- chosen_n_compared: `{s.get('chosen_n_compared')}`\n")
    else:
        lines.append("- alignment check disabled or unavailable.\n")

    ensure_dir(os.path.dirname(out.report))
    with open(out.report, "w", encoding="utf-8") as f:
        f.write("".join(lines))


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead1_in", required=True)
    ap.add_argument("--nowcast_in", default="")

    ap.add_argument("--lead1_out", required=True)
    ap.add_argument("--nowcast_out", default="")  # required iff nowcast_in provided

    ap.add_argument("--report", required=True)
    ap.add_argument("--qc_out", required=True)
    ap.add_argument("--qc_badrows_out", default="")  # optional; default next to report

    ap.add_argument("--hash_algo", choices=["sha256", "md5"], default="sha256")
    ap.add_argument("--cap_tslf", type=int, default=60)

    ap.add_argument("--add_has_fire_history", action="store_true")
    ap.add_argument("--add_ever_burned_before", action="store_true")

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backup_on_overwrite", action="store_true")

    ap.add_argument("--check_alignment_k", type=int, default=0)
    ap.add_argument("--check_seed", type=int, default=0)
    ap.add_argument("--enforce_monthly_continuity", choices=["off", "warn", "fail"], default="off")
    ap.add_argument(
        "--lead1_label_shift",
        choices=["off", "auto", "plus1", "same", "minus1"],
        default="same",
    )
    ap.add_argument("--alignment_min_match", type=float, default=0.98)

    # extra verification knobs
    ap.add_argument("--verify_firecount_rolling", action="store_true")
    ap.add_argument("--verify_firecount_k_cells", type=int, default=50)
    ap.add_argument("--verify_firecount_seed", type=int, default=0)

    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not os.path.exists(args.lead1_in):
        raise FileNotFoundError(args.lead1_in)
    if args.nowcast_in and (not os.path.exists(args.nowcast_in)):
        raise FileNotFoundError(args.nowcast_in)
    if args.nowcast_in and not args.nowcast_out.strip():
        raise ValueError("When --nowcast_in is provided, you must also provide --nowcast_out.")

    lead1_out = args.lead1_out.strip()
    nowcast_out = args.nowcast_out.strip() if args.nowcast_in else ""

    maybe_overwrite_path(lead1_out, args.overwrite, args.backup_on_overwrite, "lead1_out")
    if nowcast_out:
        maybe_overwrite_path(nowcast_out, args.overwrite, args.backup_on_overwrite, "nowcast_out")
    maybe_overwrite_path(args.report, args.overwrite, args.backup_on_overwrite, "report")
    maybe_overwrite_path(args.qc_out, args.overwrite, args.backup_on_overwrite, "qc_out")
    if args.qc_badrows_out.strip():
        maybe_overwrite_path(args.qc_badrows_out.strip(), args.overwrite, args.backup_on_overwrite, "qc_badrows_out")

    in_hashes = {"lead1_in": file_hash(args.lead1_in, algo=args.hash_algo)}
    if args.nowcast_in:
        in_hashes["nowcast_in"] = file_hash(args.nowcast_in, algo=args.hash_algo)

    # ---- load lead1 ----
    lead1 = pd.read_parquet(args.lead1_in)
    lead1 = coerce_date(lead1, "lead1")
    require_cols(lead1, ["cell_id", "date", "y"], "lead1")
    assert_y_binary(lead1, "lead1", y_col="y")
    assert_no_suspicious_future_cols(lead1, "lead1")
    assert_cv_only_cols_passthrough(lead1, "lead1")
    lead1_sorted = assert_panel_integrity(lead1, "lead1")
    lead1_base_cols = list(lead1.columns)

    lead1_gap_n, lead1_gap_ex = check_monthly_continuity(
        lead1_sorted, "lead1", mode=args.enforce_monthly_continuity
    )

    lead1_f1 = add_fire_history_features(
        lead1_sorted,
        y_col="y",
        cap_months=args.cap_tslf,
        add_has_fire_history=args.add_has_fire_history,
        add_ever_burned_before=args.add_ever_burned_before,
        verify_firecount_rolling=bool(args.verify_firecount_rolling),
        verify_k_cells=int(args.verify_firecount_k_cells),
        verify_seed=int(args.verify_firecount_seed),
    )
    lead1_f1 = add_roll_features(
        lead1_f1, roll_window=3, roll_specs=LEAD1_ROLL_SPECS_3M, enforce_lag1=True, panel_name="lead1"
    )

    if args.check_alignment_k > 0:
        spotcheck_y_lag1_consistency(
            lead1_f1.sort_values(["cell_id", "date"]),
            args.check_alignment_k,
            seed=args.check_seed,
            name="lead1",
        )

    lead1_derived_cols = [
        "y_lag1", "y_lag12", "fire_count_12m", "months_since_last_fire", "fire_tslf",
        "P_sum_mon_lag1_roll3m_sum", "PET_sum_mon_lag1_roll3m_sum",
        "VPD_mean_mon_lag1_roll3m_mean", "NDVI_mean_mon_lag1_roll3m_mean",
    ]
    if args.add_has_fire_history:
        lead1_derived_cols.append("has_fire_history")
    if args.add_ever_burned_before:
        lead1_derived_cols.append("ever_burned_before")

    lead1_lag_cols = ["y_lag1", "y_lag12"]
    lead1_roll_cols = [
        "P_sum_mon_lag1_roll3m_sum",
        "PET_sum_mon_lag1_roll3m_sum",
        "VPD_mean_mon_lag1_roll3m_mean",
        "NDVI_mean_mon_lag1_roll3m_mean",
    ]
    lead1_basecov_cols = ["P_sum_mon_lag1", "PET_sum_mon_lag1", "VPD_mean_mon_lag1", "NDVI_mean_mon_lag1"]

    # ---- optional nowcast ----
    now_f1: Optional[pd.DataFrame] = None
    now_sorted: Optional[pd.DataFrame] = None
    now_base_cols: List[str] = []
    now_gap_n, now_gap_ex = 0, ""
    align_summary: Optional[Dict[str, object]] = None

    now_date_min = ""
    now_date_max = ""

    if args.nowcast_in:
        now = pd.read_parquet(args.nowcast_in)
        now = coerce_date(now, "nowcast")
        require_cols(now, ["cell_id", "date", "y"], "nowcast")
        assert_y_binary(now, "nowcast", y_col="y")
        assert_no_suspicious_future_cols(now, "nowcast")
        assert_cv_only_cols_passthrough(now, "nowcast")
        now_sorted = assert_panel_integrity(now, "nowcast")
        now_base_cols = list(now.columns)

        now_gap_n, now_gap_ex = check_monthly_continuity(
            now_sorted, "nowcast", mode=args.enforce_monthly_continuity
        )

        now_f1 = add_fire_history_features(
            now_sorted,
            y_col="y",
            cap_months=args.cap_tslf,
            add_has_fire_history=args.add_has_fire_history,
            add_ever_burned_before=args.add_ever_burned_before,
            verify_firecount_rolling=bool(args.verify_firecount_rolling),
            verify_k_cells=int(args.verify_firecount_k_cells),
            verify_seed=int(args.verify_firecount_seed),
        )
        now_f1 = add_roll_features(
            now_f1, roll_window=3, roll_specs=NOWCAST_ROLL_SPECS_3M, enforce_lag1=False, panel_name="nowcast"
        )

        if args.check_alignment_k > 0 and now_sorted is not None:
            align_summary = spotcheck_lead1_vs_nowcast_alignment(
                lead1_sorted=lead1_sorted,
                now_sorted=now_sorted,
                k=args.check_alignment_k,
                seed=args.check_seed,
                shift_mode=args.lead1_label_shift,
                min_match=args.alignment_min_match,
            )

        now_date_min = str(now_sorted["date"].min().date()) if now_sorted is not None else ""
        now_date_max = str(now_sorted["date"].max().date()) if now_sorted is not None else ""

    # ---- write outputs ----
    atomic_write_parquet(lead1_f1, lead1_out)
    if now_f1 is not None:
        atomic_write_parquet(now_f1, nowcast_out)

    out_hashes = {"lead1_out": file_hash(lead1_out, algo=args.hash_algo)}
    if now_f1 is not None:
        out_hashes["nowcast_out"] = file_hash(nowcast_out, algo=args.hash_algo)

    qc_badrows_out = args.qc_badrows_out.strip()
    if not qc_badrows_out:
        qc_badrows_out = os.path.join(os.path.dirname(args.report), "qc_step6_badrows.csv")

    lead1_date_min = str(lead1_sorted["date"].min().date())
    lead1_date_max = str(lead1_sorted["date"].max().date())

    cv_lead1 = "|".join(sorted([c for c in CV_ONLY_BLOCK_COLS if c in lead1.columns]))
    cv_now = "|".join(sorted([c for c in CV_ONLY_BLOCK_COLS if (args.nowcast_in and c in (now.columns if args.nowcast_in else []))]))

    ro = RunOutputs(
        run_id=run_id,
        lead1_in=args.lead1_in,
        nowcast_in=args.nowcast_in,
        lead1_out=lead1_out,
        nowcast_out=nowcast_out,
        report=args.report,
        qc_out=args.qc_out,
        qc_badrows_out=qc_badrows_out,
        hash_algo=args.hash_algo,
        in_hashes=in_hashes,
        out_hashes=out_hashes,
        lead1_label_shift=args.lead1_label_shift,
        alignment_summary=align_summary,
        lead1_date_min=lead1_date_min,
        lead1_date_max=lead1_date_max,
        nowcast_date_min=now_date_min,
        nowcast_date_max=now_date_max,
        cv_only_cols_present_lead1=cv_lead1,
        cv_only_cols_present_nowcast=cv_now,
    )
    write_report_md(ro, cap_tslf=args.cap_tslf)

    # ---- qc ----
    qc_rows = []
    qc_rows.append(qc_panel_summary(
        lead1_f1,
        "lead1_F1_panel",
        lead1_derived_cols,
        lead1_base_cols,
        firehist_cap=args.cap_tslf,
        gap_rows_n=lead1_gap_n,
        gap_examples=lead1_gap_ex,
        lag_cols=lead1_lag_cols,
        rolling_cols=lead1_roll_cols,
        base_cov_cols_for_missing=lead1_basecov_cols,
        alignment_meta=align_summary if align_summary else {"alignment_mode": args.lead1_label_shift},
    ))

    if now_f1 is not None:
        now_derived_cols = [
            "y_lag1", "y_lag12", "fire_count_12m", "months_since_last_fire", "fire_tslf",
            "P_sum_mon_roll3m_sum", "PET_sum_mon_roll3m_sum",
            "VPD_mean_mon_roll3m_mean", "NDVI_mean_mon_roll3m_mean",
        ]
        if args.add_has_fire_history:
            now_derived_cols.append("has_fire_history")
        if args.add_ever_burned_before:
            now_derived_cols.append("ever_burned_before")

        now_lag_cols = ["y_lag1", "y_lag12"]
        now_roll_cols = [
            "P_sum_mon_roll3m_sum",
            "PET_sum_mon_roll3m_sum",
            "VPD_mean_mon_roll3m_mean",
            "NDVI_mean_mon_roll3m_mean",
        ]
        now_basecov_cols = ["P_sum_mon", "PET_sum_mon", "VPD_mean_mon", "NDVI_mean_mon"]

        qc_rows.append(qc_panel_summary(
            now_f1,
            "nowcast_F1_panel",
            now_derived_cols,
            now_base_cols,
            firehist_cap=args.cap_tslf,
            gap_rows_n=now_gap_n,
            gap_examples=now_gap_ex,
            lag_cols=now_lag_cols,
            rolling_cols=now_roll_cols,
            base_cov_cols_for_missing=now_basecov_cols,
            alignment_meta=None,
        ))

    ensure_dir(os.path.dirname(args.qc_out))
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(args.qc_out, index=False)

    if "bad_rows_n" in qc_df.columns and (qc_df["bad_rows_n"] > 0).any():
        bad_panels = qc_df.loc[qc_df["bad_rows_n"] > 0, ["panel", "bad_rows_n"]].to_dict(orient="records")

        ensure_dir(os.path.dirname(qc_badrows_out))
        bad_rows_all = []

        def _collect_bad(df_panel: pd.DataFrame, panel_name: str) -> None:
            mask = qc_bad_mask_firehist(df_panel, cap=args.cap_tslf)
            if mask.any():
                cols = [c for c in ["cell_id", "date", "y", "y_lag1", "y_lag12", "fire_count_12m", "months_since_last_fire"] if c in df_panel.columns]
                for extra in ["has_fire_history", "ever_burned_before"]:
                    if extra in df_panel.columns:
                        cols.append(extra)
                # include CV-only cols if exist (help debugging joins)
                for c in sorted(CV_ONLY_BLOCK_COLS):
                    if c in df_panel.columns:
                        cols.append(c)
                tmp = df_panel.loc[mask, cols].copy()
                tmp.insert(0, "panel", panel_name)
                bad_rows_all.append(tmp.head(2000))

        _collect_bad(lead1_f1, "lead1_F1_panel")
        if now_f1 is not None:
            _collect_bad(now_f1, "nowcast_F1_panel")

        if bad_rows_all:
            pd.concat(bad_rows_all, axis=0).to_csv(qc_badrows_out, index=False)

        raise RuntimeError(f"[Step6 QC] fire-history invariants violated. panels={bad_panels}. badrows={qc_badrows_out}")

    print("✅ Step6 done (F1: memory + rolling + fire-history; NO anomaly).")
    print(f"- run_id: {run_id}")
    print(f"- lead1_out: {lead1_out} (hash={out_hashes['lead1_out']})")
    if now_f1 is not None:
        print(f"- nowcast_out: {nowcast_out} (hash={out_hashes['nowcast_out']})")
    print(f"- lead1_label_shift: {args.lead1_label_shift}")
    if align_summary:
        print(f"- alignment: {align_summary}")
    if args.verify_firecount_rolling:
        print(f"- fire_count_12m verification: ON (k_cells={args.verify_firecount_k_cells})")
    print(f"- cv_only_cols_present (lead1): {cv_lead1}")
    if args.nowcast_in:
        print(f"- cv_only_cols_present (nowcast): {cv_now}")
    print(f"- report: {args.report}")
    print(f"- qc: {args.qc_out}")
    print(f"- qc_badrows_out (if fail): {qc_badrows_out}")


if __name__ == "__main__":
    main()