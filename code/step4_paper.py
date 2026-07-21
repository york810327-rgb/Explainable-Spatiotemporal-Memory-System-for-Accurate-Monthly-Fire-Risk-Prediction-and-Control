#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step4 数据对象（Data object + Label体系） — Journal-grade, leakage-hard (revised)

关键修订（针对你列出的“最容易被误用/被审稿人抓住”的点）：
1) ✅ valid_weight 火信息混入风险：在 processed（nowcast/lead1）里直接 drop，并加入 forbidden 检查硬拦截
2) ✅ cell_id 生成可审计：panel_base（interim）保留 system:index 原始字段，并新增 cell_id_raw_dyn / cell_id_raw_sta
3) ✅ Lead-1 lag 列识别更稳：is_dynamic_or_quality_col 用 pattern/regex 加固（不再只 hardcode 单列）
4) ✅ processed forbidden 扩展：除了 burned_area_m2/burned_flag/burned_frac/ever_burned，还硬拦截：
   - valid_weight（以及 valid_weight*）
   - 所有 burned_* 前缀（避免未来残留 label proxy）
5) ✅ alignment QC：优先把 out_json 直接交给 qc_lead1_alignment（若函数支持），否则 fallback 自己写 json
   并把 checked_lag_cols_sha256 写入 qc_panel_base.csv（align_cols_sha256）

Inputs:
  --dynamic  /path/CNVPAS_fireDyn_v2_..._Y*.csv
  --static   /path/CNVPAS_static_v2_....csv
  [optional] --samples_csv /path/CN_VPAS_v3lite_samples_10km_2015_2024.csv

Outputs:
  data/interim/panel_base.parquet
  data/interim/qc_panel_base.csv
  data/interim/qc_attrition_cells.csv (optional)
  data/interim/label_aux_burned_area_m2.parquet (optional)
  data/processed/panel_nowcast.parquet
  data/processed/panel_lead1.parquet
  artifacts/data_dictionary.xlsx
  artifacts/qc_label_consistency.json
  artifacts/qc_lead1_alignment.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_hash(path: str, algo: str = "sha256", chunk_size: int = 2**20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_cell_id_from_system_index(s: str) -> str:
    """
    动态表 cell_id：从 system:index 提取
    - 优先取第一个 '_' 之后的部分（split('_',1)[1]）
    - 若无 '_'，则原样返回
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    if "_" in s:
        return s.split("_", 1)[1]
    return s


def to_month_start_date(year: pd.Series, month: pd.Series) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce").astype("Int64")
    m = pd.to_numeric(month, errors="coerce").astype("Int64")
    dt = pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": 1}), errors="coerce")
    return dt


def drop_reducer_duplicate_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除 *_mean / *_count（若 base 列存在），保留 tidy 套装：x, x_n, x_cov, x_missing
    """
    cols = df.columns.tolist()
    drop_cols = []

    for c in cols:
        if c.endswith("_mean") or c.endswith("_count"):
            base = re.sub(r"_(mean|count)$", "", c)
            if base in df.columns:
                drop_cols.append(c)

    for c in cols:
        if c.endswith("_mean") and c[:-5] in df.columns:
            drop_cols.append(c)
        if c.endswith("_count") and c[:-6] in df.columns:
            drop_cols.append(c)

    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def drop_reducer_intermediate_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除 reduceRegions / reducer 的典型中间输出列：
      - sum
      - pix_mean / pix
    注意：pix_count 必须保留
    """
    bad = [c for c in ["sum", "pix_mean", "pix"] if c in df.columns]
    if bad:
        df = df.drop(columns=bad, errors="ignore")
    return df


def add_month_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    """
    安全季节性编码（不会泄漏）
    """
    m = pd.to_numeric(df["month"], errors="coerce").astype(float)
    df["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12.0)
    return df


def call_qc_func_adaptive(func, **kwargs):
    """
    Call QC function with only kwargs it accepts.
    """
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return func(**filtered)


# -------------------------
# Leakage / Forbidden rules
# -------------------------

FORBIDDEN_IN_PROCESSED_EXACT = {
    # raw labels
    "burned_area_m2",
    "burned_flag",
    # full-period summaries / label proxy
    "burned_frac",
    "ever_burned",
    # 🔥 risk: contains burned_frac by construction in your Step1
    "valid_weight",
}

FORBIDDEN_IN_PROCESSED_PREFIX = (
    "label_aux_",
    # ✅ 统一拦截所有 burned_*，防止未来残留 label proxy（比如 burned_frac_mon 之类）
    "burned_",
    # ✅ 拦截 valid_weight* 的未来演化（valid_weight2 / valid_weight_v2 等）
    "valid_weight",
)

FORBIDDEN_IN_PROCESSED_META = {
    "system:index",
    ".geo",
    "__source_file__",
}


def assert_no_forbidden_processed(df: pd.DataFrame, where: str) -> None:
    bad: List[str] = []
    for c in df.columns:
        if c in FORBIDDEN_IN_PROCESSED_EXACT:
            bad.append(c)
        for p in FORBIDDEN_IN_PROCESSED_PREFIX:
            if c.startswith(p) and c not in {"burned_flag"}:
                # burned_flag 已在 exact 里；这里避免重复即可
                bad.append(c)
        if c in FORBIDDEN_IN_PROCESSED_META:
            bad.append(c)

    # 允许 month_sin/month_cos 等 derived
    bad = sorted(set(bad))
    if bad:
        raise RuntimeError(f"[{where}] Forbidden columns present in processed output: {bad}")


def find_forbidden_cols(df: pd.DataFrame) -> List[str]:
    bad: List[str] = []
    for c in df.columns:
        if c in FORBIDDEN_IN_PROCESSED_EXACT:
            bad.append(c)
        for p in FORBIDDEN_IN_PROCESSED_PREFIX:
            if c.startswith(p):
                bad.append(c)
        if c in FORBIDDEN_IN_PROCESSED_META:
            bad.append(c)
    return sorted(set(bad))


# -------------------------
# Lag column identification
# -------------------------

_STRONG_WS_PAT = re.compile(r"(strong_frac|WS_strong)", re.IGNORECASE)

def is_dynamic_or_quality_col(c: str) -> bool:
    """
    更稳健的动态/质量列识别（用于 Lead-1 lag）
      - 动态：含 "_mon"（VPD_mean_mon 等）
      - 质量：以 _cov/_missing/_n 结尾
      - pix_count 属于质量（应 lag1）
      - 强风分数类：匹配 strong_frac/WS_strong（避免未来 schema 演化漏 lag）
    """
    if c == "pix_count":
        return True
    if "_mon" in c:
        return True
    if c.endswith(("_cov", "_missing", "_n")):
        return True
    if _STRONG_WS_PAT.search(c) is not None:
        return True
    return False


def coerce_lag_int_columns(panel_lead1: pd.DataFrame) -> pd.DataFrame:
    """
    规范化 Lead1 里 shift 后可能变成 float 的计数/缺失列
    """
    df = panel_lead1
    int_like_cols = []
    for c in df.columns:
        if c.endswith("_missing_lag1") or c.endswith("_n_lag1") or c == "pix_count_lag1":
            int_like_cols.append(c)

    for c in int_like_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")

    return df


# -------------------------
# Config
# -------------------------

@dataclass
class BuildConfig:
    start_year: int = 2015
    end_year: int = 2024


# -------------------------
# Step4.1 Build panel_base (interim)
# -------------------------

def read_dynamic_files(dynamic_glob: str) -> Tuple[pd.DataFrame, List[str]]:
    files = sorted(glob.glob(dynamic_glob))
    if not files:
        raise FileNotFoundError(f"No dynamic files matched: {dynamic_glob}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        df["__source_file__"] = os.path.basename(fp)
        dfs.append(df)

    dyn = pd.concat(dfs, ignore_index=True)
    dyn.columns = [c.strip() for c in dyn.columns]
    return dyn, files


def build_panel_base(
    dynamic_glob: str,
    static_path: str,
    cfg: BuildConfig,
    pi_policy: str = "keep",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    dyn, dyn_files = read_dynamic_files(dynamic_glob)
    sta = pd.read_csv(static_path, low_memory=False)
    sta.columns = [c.strip() for c in sta.columns]

    if "system:index" not in dyn.columns or "system:index" not in sta.columns:
        raise KeyError("Both dynamic and static must contain 'system:index'.")

    # --- audit raw ids (interim only) ---
# --- audit raw ids (interim only) + stable join key ---
    dyn = dyn.assign(
        cell_id_raw_dyn=dyn["system:index"].astype(str),
        cell_id=dyn["system:index"].map(parse_cell_id_from_system_index),
    )

    sta = sta.assign(
        cell_id_raw_sta=sta["system:index"].astype(str),
        cell_id=sta["system:index"].astype(str),
    )

    # （可选再加一刀：彻底去碎片）
    dyn = dyn.copy()
    sta = sta.copy()

    # --- date ---
    if "year" not in dyn.columns or "month" not in dyn.columns:
        raise KeyError("Dynamic must contain 'year' and 'month'.")
    dyn["date"] = to_month_start_date(dyn["year"], dyn["month"])
    bad = int(dyn["date"].isna().sum())
    if bad > 0:
        raise ValueError(f"Dynamic has {bad} rows with invalid (year,month)->date parsing.")

    # --- enforce month dtype early ---
    dyn["month"] = pd.to_numeric(dyn["month"], errors="coerce").astype("Int64")
    if dyn["month"].isna().any():
        raise ValueError(f"Dynamic has NaN month after coercion: {int(dyn['month'].isna().sum())} rows")
    dyn["month"] = dyn["month"].astype(np.int8)

    # --- required fields ---
    needed_dyn = {"cell_id", "date", "year", "month", "burned_flag", "burned_area_m2"}
    missing_needed = sorted(list(needed_dyn - set(dyn.columns)))
    if missing_needed:
        raise KeyError(f"Dynamic missing required columns: {missing_needed}")

    # --- drop duplicates & intermediates ---
    dyn = drop_reducer_duplicate_cols(dyn)
    dyn = drop_reducer_intermediate_cols(dyn)

    # --- static de-dup for join ---
    overlap = sorted(set(dyn.columns) & set(sta.columns))
    overlap_remove_from_static = [c for c in overlap if c not in {"cell_id"}]
    sta2 = sta.drop(columns=overlap_remove_from_static, errors="ignore")

    # --- LEFT JOIN dynamic as master ---
    panel_base = dyn.merge(sta2, on="cell_id", how="left", validate="many_to_one")

    # --- sort ---
    panel_base = panel_base.sort_values(["cell_id", "date"]).reset_index(drop=True)

    # --- QC: expected months ---
    expected_months = pd.date_range(
        start=f"{cfg.start_year}-01-01",
        end=f"{cfg.end_year}-12-01",
        freq="MS",
    )
    exp_n_months = len(expected_months)

    # --- QC summary ---
    n_cells_dynamic = int(panel_base["cell_id"].nunique())
    n_cells_static = int(sta["cell_id"].nunique())
    STATIC_AUDIT_EXCLUDE = {
        "cell_id",
        "cell_id_raw_sta",
        "system:index",
    }

    static_feature_cols = [c for c in sta2.columns if c not in STATIC_AUDIT_EXCLUDE]
    n_cells_joined = int(
        panel_base.loc[
            panel_base[static_feature_cols].notna().any(axis=1),
            "cell_id",
        ].nunique()
    )

    months_per_cell = panel_base.groupby("cell_id")["date"].nunique()
    missing_months_count = exp_n_months - months_per_cell
    missing_months_count_total = int((missing_months_count.clip(lower=0)).sum())
    burn_rate_overall = float(pd.to_numeric(panel_base["burned_flag"], errors="coerce").mean())

    # --- pi audit ---
    pi_over1_count = np.nan
    pi_over1_rate = np.nan
    pi_nonpos_count = np.nan
    pi_nonpos_rate = np.nan

    if "pi" in panel_base.columns:
        pi = pd.to_numeric(panel_base["pi"], errors="coerce")
        pi_over1_count = int((pi > 1).sum())
        pi_over1_rate = float((pi > 1).mean())
        pi_nonpos_count = int((pi <= 0).sum())
        pi_nonpos_rate = float((pi <= 0).mean())

        if pi_policy == "clip":
            panel_base["pi_clip"] = pi.clip(lower=0, upper=1)
            if "d_weight" in panel_base.columns:
                panel_base["d_weight_clip"] = np.where(panel_base["pi_clip"] > 0, 1.0 / panel_base["pi_clip"], np.nan)

    # --- missing static rates ---
    missing_rates: Dict[str, float] = {}
    for c in static_feature_cols:
        missing_rates[f"missing_static_rate__{c}"] = float(panel_base[c].isna().mean())

    qc_row: Dict[str, Any] = {
        "n_cells_dynamic": n_cells_dynamic,
        "n_cells_static": n_cells_static,
        "n_cells_joined": n_cells_joined,
        "months_per_cell_min": int(months_per_cell.min()),
        "months_per_cell_median": float(months_per_cell.median()),
        "months_per_cell_max": int(months_per_cell.max()),
        "missing_months_count_total": missing_months_count_total,
        "missing_months_count_median": float((missing_months_count.clip(lower=0)).median()),
        "missing_months_count_max": int((missing_months_count.clip(lower=0)).max()),
        "burn_rate_overall": burn_rate_overall,
        "dynamic_files_n": len(dyn_files),
        "dynamic_files_example": os.path.basename(dyn_files[0]),
        "static_file": os.path.basename(static_path),
        "expected_months_per_cell": exp_n_months,
        "pi_over1_count": pi_over1_count,
        "pi_over1_rate": pi_over1_rate,
        "pi_nonpos_count": pi_nonpos_count,
        "pi_nonpos_rate": pi_nonpos_rate,
        "pi_policy": pi_policy,
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "static_hash": "",           # filled in main
        "dynamic_hash_example": "",  # filled in main
        **missing_rates,
    }
    qc = pd.DataFrame([qc_row])

    return panel_base, qc


# -------------------------
# Step4.2 Nowcast & Lead-1 (processed panels)
# -------------------------

def _processed_drop_cols() -> List[str]:
    """
    所有 processed 必须 drop 的列（统一维护）
    """
    drop_cols = [
        # labels / label proxies
        "burned_area_m2",
        "burned_flag",
        "burned_frac",
        "ever_burned",
        "valid_weight",     # ✅ 핵심：直接在 processed 切断火信息混入风险

        # meta / geometry / source
        "system:index",
        ".geo",
        "__source_file__",

        # coords (你现在作为 static covariates 不用；且很多期刊不喜欢经纬度当 proxy)
        "lon",
        "lat",

        # occasional artifacts
        "mode",

        # audit raw ids (interim only)
        "cell_id_raw_dyn",
        "cell_id_raw_sta",
    ]
    return drop_cols


def derive_nowcast(panel_base: pd.DataFrame) -> pd.DataFrame:
    """
    Nowcast-full:
      y(t) + X(t) (dynamic same-month + quality + static + weights/stratifiers)
    """
    df = panel_base.copy()

    # label
    df = df.rename(columns={"burned_flag": "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if df["y"].isna().any():
        raise ValueError(f"Nowcast: y has NaN after coercion: {int(df['y'].isna().sum())} rows")
    u = set(df["y"].unique().tolist())
    if not u.issubset({0, 1}):
        raise ValueError(f"Nowcast: y has non-binary values: {sorted(u)}")
    df["y"] = df["y"].astype(np.int8)

    # month int
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    if df["month"].isna().any():
        raise ValueError(f"Nowcast: month has NaN after coercion: {int(df['month'].isna().sum())} rows")
    df["month"] = df["month"].astype(np.int8)

    # drop forbidden/audit/geometry/source fields in processed
    drop_cols = _processed_drop_cols()
    drop_cols += [c for c in df.columns if c.startswith("label_aux_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # add seasonality
    df = add_month_sin_cos(df)

    # required
    for k in ["cell_id", "date", "year", "month", "y"]:
        if k not in df.columns:
            raise KeyError(f"Nowcast missing required field: {k}")

    assert_no_forbidden_processed(df, where="nowcast")
    return df


def derive_lead1(panel_base: pd.DataFrame, static_cols_all: Set[str]) -> pd.DataFrame:
    """
    Lead-1-full:
      y(t) kept at time t
      X = dynamic/quality at t-1 -> *_lag1
      static/weights/stratifiers stay at t (not lagged)
    """
    df = panel_base.copy().sort_values(["cell_id", "date"]).reset_index(drop=True)

    # label
    df = df.rename(columns={"burned_flag": "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if df["y"].isna().any():
        raise ValueError(f"Lead1: y has NaN after coercion: {int(df['y'].isna().sum())} rows")
    u = set(df["y"].unique().tolist())
    if not u.issubset({0, 1}):
        raise ValueError(f"Lead1: y has non-binary values: {sorted(u)}")
    df["y"] = df["y"].astype(np.int8)

    # month int
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    if df["month"].isna().any():
        raise ValueError(f"Lead1: month has NaN after coercion: {int(df['month'].isna().sum())} rows")
    df["month"] = df["month"].astype(np.int8)

    # drop forbidden/audit/geometry/source early
    drop_early = _processed_drop_cols()
    drop_early += [c for c in df.columns if c.startswith("label_aux_")]
    df = df.drop(columns=drop_early, errors="ignore")

    # add seasonality
    df = add_month_sin_cos(df)

    # identify columns
    id_cols = {"cell_id", "date", "year", "month"}
    label_cols = {"y"}
    season_cols = {"month_sin", "month_cos"}

    weight_like = {"pi", "d_weight", "pi_clip", "d_weight_clip", "weight"}
    strat_like = {"BIOME", "HUMAN_bin", "STRATUM", "VPD_bin", "VPD_mean_bin", "Nh", "nh", "valid_core"}

    # ✅ 注意：valid_weight 已被 drop，不再列入 static_like
    static_like = set(static_cols_all) | strat_like | weight_like

    candidate = [c for c in df.columns if c not in (id_cols | label_cols | season_cols)]
    lag_cols = [c for c in candidate if (c not in static_like) and is_dynamic_or_quality_col(c)]
    if not lag_cols:
        raise RuntimeError("Lead1: No lag columns detected. Check naming rules or input schema.")

    lagged = (
        df.groupby("cell_id", sort=False)[lag_cols]
          .shift(1)
          .add_suffix("_lag1")
    )

    out = pd.concat([df.drop(columns=lag_cols, errors="ignore"), lagged], axis=1)

    # drop first row of each cell
    out["_row_in_cell"] = out.groupby("cell_id", sort=False).cumcount()
    out = out.loc[out["_row_in_cell"] > 0].drop(columns=["_row_in_cell"]).reset_index(drop=True)

    # sanity: months per cell (2015-2024 => 120 months, lead1 => 119)
    exp = 119
    cnt = out.groupby("cell_id")["date"].nunique()
    if not (cnt == exp).all():
        bad_cells = cnt[cnt != exp].sort_values()
        raise ValueError(
            f"Lead1 months per cell not equal {exp}. "
            f"bad_cells_n={len(bad_cells)} examples={bad_cells.head(10).to_dict()}"
        )

    for k in ["cell_id", "date", "year", "month", "y", "month_sin", "month_cos"]:
        if k not in out.columns:
            raise KeyError(f"Lead1 missing required field: {k}")

    assert_no_forbidden_processed(out, where="lead1")
    return out


# -------------------------
# Step4.3 Data dictionary
# -------------------------

def guess_column_type(col: str, static_cols: set, dynamic_cols: set) -> str:
    if col in {"cell_id", "date", "year", "month"}:
        return "key"
    if col in {"burned_flag", "burned_area_m2", "y"}:
        return "label"
    if col in {"pi", "d_weight", "pi_clip", "d_weight_clip", "weight"}:
        return "weight"
    if col.startswith("valid_weight"):
        return "forbidden"
    if col.endswith("_cov") or col.endswith("_missing") or col.endswith("_n") or col == "pix_count":
        return "quality"
    if col in {"month_sin", "month_cos"}:
        return "derived"
    if col in static_cols:
        return "static"
    if col in dynamic_cols or col.endswith("_lag1"):
        return "dynamic"
    return "other"


def build_data_dictionary(panel_base: pd.DataFrame, static_cols: set, dynamic_cols: set) -> pd.DataFrame:
    source_map = {
        "VPD": "ERA5-Land DAILY_AGGR",
        "RH": "ERA5-Land DAILY_AGGR",
        "Tmean": "ERA5-Land DAILY_AGGR",
        "Tmax": "ERA5-Land DAILY_AGGR",
        "WS": "ERA5-Land DAILY_AGGR",
        "WD_": "ERA5-Land DAILY_AGGR",
        "SM1": "ERA5-Land DAILY_AGGR",
        "SM2": "ERA5-Land DAILY_AGGR",
        "TP_sum_mon": "ERA5-Land DAILY_AGGR",
        "PET_sum_mon": "ERA5-Land DAILY_AGGR",
        "P_sum_mon": "IMERG MONTHLY V07",
        "NDVI": "MODIS MOD13A2",
        "EVI": "MODIS MOD13A2",
        "LST_": "MODIS MOD11A2",
        "POP_mean": "GHSL POP (P2023A)",
        "NTL_mean": "VIIRS DNB ANNUAL",
        "BUILT_mean": "GHSL BUILT (P2023A)",
        "dist_water_m": "JRC GSW 1.4 (distance transform)",
        "dist_built_m": "GHSL BUILT (distance transform)",
        "elev_mean": "SRTM",
        "slope_mean": "SRTM-derived",
        "aspect_mean": "SRTM-derived",
        "treecover_2015": "MODIS MOD44B",
        "LC_mode": "MODIS MCD12Q1",
        "frac_crop": "MODIS MCD12Q1 (derived)",
        "frac_grass": "MODIS MCD12Q1 (derived)",
        "frac_forest": "MODIS MCD12Q1 (derived)",
        "burned_area_m2": "MODIS MCD64A1",
        "burned_flag": "MODIS MCD64A1 (derived)",
        "burned_frac": "MODIS MCD64A1 (derived; full-period summary)",
        "pi": "VPAS sampling design (derived)",
        "d_weight": "VPAS sampling design (derived)",
        "pi_clip": "VPAS sampling design (derived; clipped for audit)",
        "d_weight_clip": "VPAS sampling design (derived; clipped for audit)",
        "valid_core": "MCD64A1 QA + landmask (derived)",
        "valid_weight": "⚠️ DO NOT USE (contains burned_frac); dropped in processed",
        "pix_count": "reduceRegions pixel count (quality)",
        "month_sin": "calendar derived",
        "month_cos": "calendar derived",
    }

    unit_map = {
        "burned_area_m2": "m^2",
        "P_sum_mon": "mm/month",
        "TP_sum_mon": "mm/month",
        "PET_sum_mon": "mm/month",
        "VPD_mean_mon": "kPa",
        "VPD_max_mon": "kPa",
        "Tmean_mon": "°C",
        "Tmax_mon": "°C",
        "LST_day_mon": "°C",
        "LST_night_mon": "°C",
        "RH_mean_mon": "%",
        "RH_min_mon": "%",
        "WS_mean_mon": "m/s",
        "WS_max_mon": "m/s",
        "WD_u_mon": "unitless",
        "WD_v_mon": "unitless",
        "WD_R_mon": "0-1",
        "SM1_mean_mon": "m^3/m^3",
        "SM2_mean_mon": "m^3/m^3",
        "NDVI_mean_mon": "unitless",
        "EVI_mean_mon": "unitless",
        "POP_mean": "persons",
        "NTL_mean": "nW/cm^2/sr (VIIRS avg_rad)",
        "BUILT_mean": "built_surface (GHSL)",
        "dist_water_m": "m",
        "dist_built_m": "m",
        "elev_mean": "m",
        "slope_mean": "degrees",
        "aspect_mean": "degrees",
        "treecover_2015": "%",
        "frac_crop": "fraction",
        "frac_grass": "fraction",
        "frac_forest": "fraction",
        "pix_count": "count",
        "month_sin": "unitless",
        "month_cos": "unitless",
    }

    def infer_source(col: str) -> str:
        for k, v in source_map.items():
            if col == k or col.startswith(k):
                return v
        if col.endswith("_lag1"):
            return infer_source(col[:-5])
        for suf in ("_cov", "_missing", "_n"):
            if col.endswith(suf):
                return infer_source(col[: -len(suf)])
        return ""

    def infer_unit(col: str) -> str:
        if col in unit_map:
            return unit_map[col]
        if col.endswith("_lag1"):
            return infer_unit(col[:-5])
        for suf in ("_cov", "_missing", "_n"):
            if col.endswith(suf):
                return "fraction" if suf == "_cov" else ("0/1" if suf == "_missing" else "count")
        return ""

    def infer_availability(col: str) -> str:
        if col in {"burned_flag", "burned_area_m2"}:
            return "forbidden"
        if col.startswith("burned_") or col in {"ever_burned"}:
            return "forbidden(processed)"
        if col.startswith("valid_weight"):
            return "forbidden(processed)"
        if col.endswith("_lag1"):
            return "lead1_ok"
        if col.endswith(("_cov", "_missing", "_n")) or ("_mon" in col) or is_dynamic_or_quality_col(col):
            return "nowcast_ok"
        if col in static_cols or col in {"pi", "d_weight", "pi_clip", "d_weight_clip", "weight"}:
            return "nowcast_ok & lead1_ok"
        if col in {"month_sin", "month_cos"}:
            return "nowcast_ok & lead1_ok"
        return ""

    def infer_leakage_risk(col: str) -> str:
        if col.startswith("burned_") or col in {"ever_burned"}:
            return "high"
        if col.startswith("valid_weight"):
            return "high(label_proxy)"
        return "none"

    rows = []
    for col in panel_base.columns:
        ctype = guess_column_type(col, static_cols, dynamic_cols)
        rows.append({
            "name": col,
            "type": ctype,
            "definition": "",
            "unit": infer_unit(col),
            "source_product": infer_source(col),
            "temporal_availability": infer_availability(col),
            "leakage_risk": infer_leakage_risk(col),
            "missingness_notes": "",
            "used_in_feature_set": "",
        })
    return pd.DataFrame(rows)


# -------------------------
# Attrition audit (optional)
# -------------------------

def maybe_write_attrition_audit(panel_base: pd.DataFrame, samples_csv: Optional[str], out_path: str) -> Optional[pd.DataFrame]:
    if not samples_csv:
        return None
    if not os.path.exists(samples_csv):
        raise FileNotFoundError(f"--samples_csv not found: {samples_csv}")

    s = pd.read_csv(samples_csv, low_memory=False)
    if "system:index" not in s.columns:
        raise KeyError("samples_csv must contain 'system:index' column from VPAS export.")

    s = s.copy()
    s["cell_id"] = s["system:index"].astype(str)

    kept = set(panel_base["cell_id"].astype(str).unique().tolist())
    allc = s["cell_id"].astype(str)

    miss_mask = ~allc.isin(kept)
    miss = s.loc[miss_mask].copy()
    miss["reason"] = "missing_in_dynamic_export"

    keep_cols = [c for c in [
        "cell_id", "BIOME", "HUMAN_bin", "STRATUM", "VPD_bin",
        "valid_core", "valid_weight", "burned_frac", "pi", "d_weight"
    ] if c in miss.columns]
    keep_cols = keep_cols + [c for c in ["reason"] if c in miss.columns]

    miss_out = miss[keep_cols].reset_index(drop=True)
    miss_out.to_csv(out_path, index=False)
    return miss_out


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", required=True, help="Dynamic CSV glob, e.g., /path/..._Y*.csv")
    parser.add_argument("--static", required=True, help="Static CSV path")
    parser.add_argument("--out_root", default=".", help="Project root for output folders (default: current dir)")
    parser.add_argument("--start_year", type=int, default=2015)
    parser.add_argument("--end_year", type=int, default=2024)

    parser.add_argument("--pi_policy", choices=["keep", "clip"], default="keep",
                        help="keep: only audit pi; clip: create pi_clip & d_weight_clip (keep original pi/d_weight)")
    parser.add_argument("--hash_algo", choices=["sha256", "md5"], default="sha256",
                        help="Hash algorithm for reproducibility fields in QC")

    parser.add_argument("--export_label_aux", action="store_true",
                        help="Export label-only aux table (burned_area_m2) to data/interim, not mixed into processed panels.")
    parser.add_argument("--samples_csv", default=None,
                        help="Optional: original VPAS samples CSV (3000 cells) for attrition audit (qc_attrition_cells.csv).")
    parser.add_argument("--expected_n_cells", type=int, default=None,
                        help="Optional: record expected cell count (e.g., 3000) in QC summary.")

    # alignment qc knobs
    parser.add_argument("--align_sample_cells", type=int, default=200,
                        help="Lead1 alignment spot-check: number of cells to sample (0 => attempt full).")
    parser.add_argument("--align_sample_months", type=int, default=24,
                        help="Lead1 alignment spot-check: number of months per sampled cell.")
    parser.add_argument("--align_tol", type=float, default=0.0,
                        help="Numeric tolerance for alignment mismatches (default 0.0).")

    args = parser.parse_args()
    cfg = BuildConfig(start_year=args.start_year, end_year=args.end_year)

    out_interim = os.path.join(args.out_root, "data", "interim")
    out_processed = os.path.join(args.out_root, "data", "processed")
    out_artifacts = os.path.join(args.out_root, "artifacts")
    ensure_dir(out_interim)
    ensure_dir(out_processed)
    ensure_dir(out_artifacts)

    # Read static once for column inference & hash
    sta0 = pd.read_csv(args.static, low_memory=False)
    sta0.columns = [c.strip() for c in sta0.columns]
    static_cols_all = set(sta0.columns) - {"system:index"}

    static_hash = file_hash(args.static, algo=args.hash_algo)

    # Step4.1
    panel_base, qc = build_panel_base(
        args.dynamic,
        args.static,
        cfg,
        pi_policy=args.pi_policy,
    )

    dyn_files = sorted(glob.glob(args.dynamic))
    dyn_hash_example = file_hash(dyn_files[0], algo=args.hash_algo) if dyn_files else ""
    qc.loc[0, "static_hash"] = static_hash
    qc.loc[0, "dynamic_hash_example"] = dyn_hash_example

    # record expected_n_cells if provided
    if args.expected_n_cells is not None:
        qc.loc[0, "expected_n_cells"] = int(args.expected_n_cells)
        qc.loc[0, "attrition_n_cells"] = int(args.expected_n_cells) - int(qc.loc[0, "n_cells_dynamic"])
        qc.loc[0, "attrition_rate"] = float(qc.loc[0, "attrition_n_cells"] / max(int(args.expected_n_cells), 1))

    # write panel_base (interim)
    panel_base_path = os.path.join(out_interim, "panel_base.parquet")
    panel_base.to_parquet(panel_base_path, index=False)

    qc_path = os.path.join(out_interim, "qc_panel_base.csv")

    # Optional: attrition audit
    attr_path = None
    if args.samples_csv:
        attr_path = os.path.join(out_interim, "qc_attrition_cells.csv")
        maybe_write_attrition_audit(panel_base, args.samples_csv, attr_path)

    # Optional: export label-only aux table (for T2)
    aux_path = None
    if args.export_label_aux:
        aux = panel_base[["cell_id", "date", "year", "month", "burned_area_m2"]].copy()
        aux_path = os.path.join(out_interim, "label_aux_burned_area_m2.parquet")
        aux.to_parquet(aux_path, index=False)

    # Step4.2 (processed panels)
    panel_now = derive_nowcast(panel_base)
    panel_lead1 = derive_lead1(panel_base, static_cols_all=static_cols_all)
    panel_lead1 = coerce_lag_int_columns(panel_lead1)

    # -------------------------
    # QC: journal-grade gates
    # -------------------------
    print("🔍 Running Step4 QC gates (label/index/alignment)...")

    from utils.qc_checks import qc_label_consistency, qc_panel_index, qc_lead1_alignment

    # 1) Label consistency (panel_base only)
    label_qc_json = os.path.join(out_artifacts, "qc_label_consistency.json")
    label_summary = call_qc_func_adaptive(
        qc_label_consistency,
        df=panel_base,
        area_col="burned_area_m2",
        y_col="burned_flag",
        out_json=label_qc_json,
        return_summary=True,
    )
    if isinstance(label_summary, dict):
        for k in ["n_checked", "n_mismatch", "mismatch_rate", "n_area_pos", "n_y_pos"]:
            if k in label_summary:
                qc.loc[0, f"label_{k}"] = label_summary[k]

    # 2) Index integrity on processed panels
    qc_panel_index(panel_now)
    qc_panel_index(panel_lead1)

    # 3) Lead1 alignment spot-check
    align_json = os.path.join(out_artifacts, "qc_lead1_alignment.json")

    # 优先把 out_json 交给 qc_lead1_alignment（若签名支持）
    align_summary = call_qc_func_adaptive(
        qc_lead1_alignment,
        lead1_df=panel_lead1,
        nowcast_df=panel_now,
        sample_cells=args.align_sample_cells,
        sample_months=args.align_sample_months,
        tol=args.align_tol,
        out_json=align_json,
        return_summary=True,
    )

    # 若函数没返回（或旧版本不支持 out_json），fallback：最保守 positional + 手写 json
    if align_summary is None:
        try:
            align_summary = qc_lead1_alignment(panel_lead1, panel_now)
        except Exception:
            align_summary = None

    if align_summary is None:
        align_payload = {
            "note": "qc_lead1_alignment did not return a summary; executed without captured stats.",
            "align_sample_cells": int(args.align_sample_cells),
            "align_sample_months": int(args.align_sample_months),
            "align_tol": float(args.align_tol),
        }
        with open(align_json, "w", encoding="utf-8") as f:
            json.dump(align_payload, f, ensure_ascii=False, indent=2)
    else:
        # qc csv 写入关键字段 + SHA
        if isinstance(align_summary, dict):
            for k in ["n_pairs_checked", "n_mismatch_cells", "mismatch_rate_overall", "n_cols_checked", "n_mismatch"]:
                if k in align_summary:
                    qc.loc[0, f"align_{k}"] = align_summary[k]
            if "checked_lag_cols_sha256" in align_summary:
                qc.loc[0, "align_cols_sha256"] = align_summary["checked_lag_cols_sha256"]

    print("✅ QC gates passed.")

    # -------------------------
    # processed leakage audit (hard fail)
    # -------------------------
    now_forbidden = find_forbidden_cols(panel_now)
    lead1_forbidden = find_forbidden_cols(panel_lead1)

    qc.loc[0, "n_cols_panel_base"] = int(panel_base.shape[1])
    qc.loc[0, "n_cols_nowcast"] = int(panel_now.shape[1])
    qc.loc[0, "n_cols_lead1"] = int(panel_lead1.shape[1])
    qc.loc[0, "processed_nowcast_forbidden_cols"] = "|".join(now_forbidden) if now_forbidden else ""
    qc.loc[0, "processed_lead1_forbidden_cols"] = "|".join(lead1_forbidden) if lead1_forbidden else ""

    if now_forbidden:
        raise RuntimeError(f"Forbidden columns leaked into processed nowcast: {now_forbidden}")
    if lead1_forbidden:
        raise RuntimeError(f"Forbidden columns leaked into processed lead1: {lead1_forbidden}")

    # write processed parquet
    panel_now_path = os.path.join(out_processed, "panel_nowcast.parquet")
    panel_lead1_path = os.path.join(out_processed, "panel_lead1.parquet")
    panel_now.to_parquet(panel_now_path, index=False)
    panel_lead1.to_parquet(panel_lead1_path, index=False)

    # ensure numeric QC cols
    for c in ["n_cols_panel_base", "n_cols_nowcast", "n_cols_lead1"]:
        if c in qc.columns:
            qc[c] = pd.to_numeric(qc[c], errors="coerce").astype("Int64")

    qc.to_csv(qc_path, index=False)

    # Step4.3 data dictionary (based on panel_base schema)
    static_cols_for_dict = set(sta0.columns) - {"system:index"}
    dyn_like = set([c for c in panel_base.columns if is_dynamic_or_quality_col(c)]) | {"pix_count"}
    dd = build_data_dictionary(panel_base, static_cols=static_cols_for_dict, dynamic_cols=dyn_like)

    dd_path = os.path.join(out_artifacts, "data_dictionary.xlsx")
    with pd.ExcelWriter(dd_path, engine="openpyxl") as writer:
        dd.to_excel(writer, sheet_name="data_dictionary", index=False)
        qc.to_excel(writer, sheet_name="qc_summary", index=False)
        pd.DataFrame({"col": panel_now.columns}).to_excel(writer, sheet_name="schema_nowcast", index=False)
        pd.DataFrame({"col": panel_lead1.columns}).to_excel(writer, sheet_name="schema_lead1", index=False)

    print("✅ Step4 done (journal-grade, revised).")
    print(f"- panel_base:   {panel_base_path}  (rows={len(panel_base):,}, cols={panel_base.shape[1]})")
    print(f"- qc report:    {qc_path}")
    if attr_path:
        print(f"- attrition:    {attr_path}  (audit only)")
    if aux_path:
        print(f"- label_aux:    {aux_path}  (label-only, for T2; NOT in processed panels)")
    print(f"- nowcast:      {panel_now_path}   (rows={len(panel_now):,}, cols={panel_now.shape[1]})")
    print(f"- lead1:        {panel_lead1_path} (rows={len(panel_lead1):,}, cols={panel_lead1.shape[1]})")
    print(f"- dictionary:   {dd_path}")
    print(f"- qc_label_json:{label_qc_json}")
    print(f"- qc_align_json:{align_json}")
    print(f"- pi_policy:    {args.pi_policy}")
    print(f"- hash_algo:    {args.hash_algo}")


if __name__ == "__main__":
    main()