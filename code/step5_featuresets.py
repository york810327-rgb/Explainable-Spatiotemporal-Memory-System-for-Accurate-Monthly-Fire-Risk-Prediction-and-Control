#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step5 Feature Admission (F0/F1/F2) + QC + Manifest + Optional metadata_cells
v1.0.6 — adds stable spatial block id (grid10km_id) derived from panel_base lon/lat (or existing id cols)

What this adds
- A stable CV-only spatial block id column (default: grid10km_id) is merged into lead1/nowcast panels
  using cell_id, derived from panel_base lon/lat (EPSG:3857 meters) or taken from an existing id column
  (tile_id / spatial_block_id / grid10km_id) if present.
- The spatial block id is kept in the exported featureset parquet datasets (F0/F1/F2) BUT NEVER in predictors.

Why this is safe
- We never merge lon/lat into the modeling panel (lon/lat remain forbidden in Step5 input schema).
- grid10km_id is only a CV metadata column for Step7 P2, not used as a predictor.

Why 10km
- Your sampling/grid unit is 10km (VPAS v3-lite 10km grid). Using 10km as the spatial block size:
  1) matches the fundamental spatial support of cell_id,
  2) avoids mixing multiple cells into the same block too aggressively (like 50–100km),
  3) avoids making blocks too tiny/fragmented (like 1–2km), which can collapse to near cell-level groups and
     reduce the value of block-based leakage audits.
- If you want coarser spatial leakage audits, set --grid_km 25 or 50 (recommended sensitivity analysis).

Outputs
- Same as v1.0.5, plus the spatial_block handling info in manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Utilities
# =========================

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

def _dedup_keep_order(df: pd.DataFrame, xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x in df.columns and x not in seen:
            out.append(x)
            seen.add(x)
    return out

def materialize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.loc[:, cols].copy()

def safe_quantile(x: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return float("nan")
    return float(x.quantile(q))

def numeric_summary(s: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(s, errors="coerce")
    n = int(len(x))
    nan_n = int(x.isna().sum())
    arr = x.to_numpy(dtype="float64", copy=False)
    inf_n = int(np.isinf(arr).sum()) if n > 0 else 0

    xf = x.replace([np.inf, -np.inf], np.nan).dropna()
    neg_rate = float((xf < 0).mean()) if len(xf) > 0 else float("nan")

    return {
        "min": float(xf.min()) if len(xf) else float("nan"),
        "p01": safe_quantile(x, 0.01),
        "median": float(xf.median()) if len(xf) else float("nan"),
        "p99": safe_quantile(x, 0.99),
        "max": float(xf.max()) if len(xf) else float("nan"),
        "nan_rate": (nan_n / n) if n else float("nan"),
        "inf_rate": (inf_n / n) if n else float("nan"),
        "neg_rate": neg_rate,
        "n": float(n),
    }

def n_unique_nonnull(s: pd.Series) -> int:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return int(x.nunique()) if len(x) else 0

def coerce_date_column(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if "date" not in df.columns:
        raise KeyError(f"[{scenario}] Missing required column: date")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(10).to_dict(orient="records")
        raise RuntimeError(f"[{scenario}] date coercion produced NaT. Examples: {bad}")
    return df


# =========================
# Leakage / admission policy
# =========================

FORBIDDEN_EXACT = {
    "burned_area_m2", "burned_flag",
    "burned_frac", "ever_burned",
    "system:index", ".geo", "__source_file__",
    "lon", "lat",  # IMPORTANT: lon/lat forbidden in Step5 panel input; we will NOT merge them in.
}

FORBIDDEN_SUBSTR = [
    "BurnDate", "burndate", "MCD64A1",
    "future",
    r"lead\+",
    "lead_",
    r"t\+1",
    r"t\+2",
    "next_month", "forward",
    "_lead1", "lead1_", "target_", "label_",
]
FORBIDDEN_SUBSTR_RE = [re.compile(p) for p in FORBIDDEN_SUBSTR]

ALLOWED_FIREHIST_PATTERNS = [
    r"^months_since_last_fire$",
    r"^fire_count_(12m|24m|36m)$",
    r"^burned_area_(12m|24m|36m)$",
    r"^y_lag\d+$",
    r"^y_roll\d+m_(sum|mean|max|min)$",
    r"^fire_(tslf|tslb)$",
]

WEIGHT_COLS_ALL = [
    "pi", "d_weight", "pi_clip", "d_weight_clip", "weight",
    "valid_core_pass", "valid_core_raw",
]

QC_ONLY_QUALITY_COLS = {"pix_count", "pix_count_lag1"}

KEY_COLS = ["cell_id", "date", "year", "month"]
SEASON_COLS = ["month_sin", "month_cos"]
LABEL_COL = "y"

STRATA_COLS_CAND = ["BIOME", "HUMAN_bin", "STRATUM", "VPD_bin", "VPD_mean_bin", "Nh", "nh"]

# CV-only spatial block candidates (kept in dataset, never predictors)
CV_BLOCK_COLS_CAND = ["grid10km_id", "spatial_block_id", "tile_id"]

STATIC_COLS_CAND = [
    "BUILT_mean", "LC_mode", "NTL_mean", "POP_mean", "aspect_mean",
    "dist_built_m", "dist_water_m", "elev_mean",
    "frac_crop", "frac_forest", "frac_grass",
    "slope_mean", "treecover_2015",
]

EXPECTED_DYNAMIC_BASE = [
    "EVI_mean_mon", "LST_day_mon", "LST_night_mon", "NDVI_mean_mon",
    "PET_sum_mon", "P_sum_mon", "RH_mean_mon", "RH_min_mon",
    "SM1_mean_mon", "SM2_mean_mon", "TP_sum_mon",
    "Tmax_mon", "Tmean_mon", "VPD_max_mon", "VPD_mean_mon",
    "WD_R_mon", "WD_u_mon", "WD_v_mon",
    "WS_max_mon", "WS_mean_mon", "WS_strong_frac",
]

def matches_any_pattern(name: str) -> bool:
    return any(p.search(name) for p in FORBIDDEN_SUBSTR_RE)

def is_allowed_firehist(name: str) -> bool:
    return any(re.match(p, name) for p in ALLOWED_FIREHIST_PATTERNS)

def assert_no_forbidden_cols(cols: List[str], where: str) -> None:
    bad = []
    for c in cols:
        if c in FORBIDDEN_EXACT:
            bad.append(c)
            continue
        if "burned_area" in c and not is_allowed_firehist(c):
            bad.append(c)
            continue
        if matches_any_pattern(c):
            bad.append(c)
            continue
        if c.startswith("label_aux_"):
            bad.append(c)
            continue
    if bad:
        raise RuntimeError(f"[{where}] Forbidden/leakage columns present: {sorted(set(bad))}")

def enforce_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pi_clip" not in out.columns and "pi" in out.columns:
        pi = pd.to_numeric(out["pi"], errors="coerce")
        out["pi_clip"] = pi.clip(lower=0, upper=1)
    if "d_weight_clip" not in out.columns:
        if "pi_clip" in out.columns:
            pic = pd.to_numeric(out["pi_clip"], errors="coerce")
            out["d_weight_clip"] = np.where(pic > 0, 1.0 / pic, np.nan)
        elif "d_weight" in out.columns:
            out["d_weight_clip"] = pd.to_numeric(out["d_weight"], errors="coerce")
    return out

def drop_constant_columns(df: pd.DataFrame, cols: List[str]) -> Tuple[List[str], List[str]]:
    dropped, kept = [], []
    for c in cols:
        if c not in df.columns:
            continue
        nun = df[c].nunique(dropna=False)
        if nun <= 1:
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


# =========================
# Predictor hard exclusion (QC-only)
# =========================

QC_ONLY_PREDICTOR_RE = re.compile(r"^(pi(_clip)?|weight|d_weight.*|valid_.*|pix_count.*)$")

def assert_predictors_exclude_qc_only(predictor_cols: Dict[str, List[str]], where: str) -> None:
    bad_all: Dict[str, List[str]] = {}
    for fs, cols in predictor_cols.items():
        bad = sorted({c for c in cols if QC_ONLY_PREDICTOR_RE.match(c)})
        if bad:
            bad_all[fs] = bad
    if bad_all:
        raise RuntimeError(f"[{where}] QC-only columns found in predictors (hard forbidden): {bad_all}")


# =========================
# valid_core repair (QC-only)
# =========================

def _is_binary_or_constant(s: pd.Series) -> bool:
    x = pd.to_numeric(s, errors="coerce")
    uniq = set(x.dropna().unique().tolist())
    if len(uniq) <= 1:
        return True
    return uniq.issubset({0, 1})

def maybe_merge_valid_core_from_base(panel_base_path: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    - If df has valid_core and it's binary/constant -> rename to valid_core_pass.
    - Only merge base valid_core as valid_core_raw if it is continuous & non-constant (per cell_id).
    """
    out = df.copy()
    info: Dict[str, object] = {"valid_core_action": "none"}

    if "valid_core" in out.columns:
        if _is_binary_or_constant(out["valid_core"]):
            out = out.rename(columns={"valid_core": "valid_core_pass"})
            info["valid_core_action"] = "rename_valid_core_to_pass"
        else:
            out = out.rename(columns={"valid_core": "valid_core_inpanel_raw"})
            info["valid_core_action"] = "rename_valid_core_to_inpanel_raw"

    if panel_base_path and os.path.exists(panel_base_path):
        base = pd.read_parquet(panel_base_path)
        if "cell_id" in base.columns and "valid_core" in base.columns:
            base_vc = base[["cell_id", "valid_core"]].copy()
            base_vc = base_vc.groupby("cell_id", as_index=False).agg({"valid_core": "first"})
            vc = pd.to_numeric(base_vc["valid_core"], errors="coerce")
            if (not _is_binary_or_constant(vc)) and (vc.nunique(dropna=False) > 1):
                base_vc = base_vc.rename(columns={"valid_core": "valid_core_raw"})
                if "valid_core_raw" not in out.columns:
                    out = out.merge(base_vc, on="cell_id", how="left")
                info["valid_core_action"] = info["valid_core_action"] + "|merge_base_raw"
            else:
                info["valid_core_action"] = info["valid_core_action"] + "|skip_merge_base_raw_constant_or_binary"

    return out, info


# =========================
# Spatial block id (CV-only) from panel_base
# =========================

def lonlat_to_webmercator_m(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lon/lat (EPSG:4326) to Web Mercator meters (EPSG:3857).
    Stable, dependency-free.

    Notes:
    - WebMercator distorts distances with latitude, but is perfectly fine for creating *stable* grid blocks.
    - We clip latitude to the valid range for Mercator.
    """
    R = 6378137.0
    lon = lon.astype("float64")
    lat = lat.astype("float64")
    lat = np.clip(lat, -85.05112878, 85.05112878)
    x = R * np.deg2rad(lon)
    y = R * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))
    return x, y

def make_grid_id_3857(lon: pd.Series, lat: pd.Series, grid_km: float, prefix: str = "g") -> pd.Series:
    """
    Create a stable grid id using EPSG:3857 meters.
    grid_km = 10 => 10,000m grid.

    Output format: f"{prefix}{grid_km}km_x{ix}_y{iy}"
    """
    if grid_km <= 0:
        raise ValueError("grid_km must be > 0")

    lo = pd.to_numeric(lon, errors="coerce")
    la = pd.to_numeric(lat, errors="coerce")
    if lo.isna().any() or la.isna().any():
        raise RuntimeError("lon/lat contain NaN; cannot build grid id.")

    x, y = lonlat_to_webmercator_m(lo.to_numpy(), la.to_numpy())
    gs = float(grid_km) * 1000.0
    ix = np.floor(x / gs).astype(np.int64)
    iy = np.floor(y / gs).astype(np.int64)

    return (prefix + str(grid_km).replace(".", "p") + "km_x" + pd.Series(ix).astype(str) + "_y" + pd.Series(iy).astype(str))

def maybe_merge_spatial_block_from_base(
    panel_base_path: str,
    df: pd.DataFrame,
    *,
    out_col: str = "grid10km_id",
    policy: str = "auto",  # off|auto|require
    prefer_cols: Optional[List[str]] = None,
    grid_km: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Merge a stable CV-only spatial block id by cell_id.

    Priority:
    1) If df already has out_col -> keep it
    2) If panel_base has one of prefer_cols -> use it
    3) Else if panel_base has lon/lat -> derive grid id (EPSG:3857) at grid_km
    4) Else fallback: use STRATUM as a last-resort proxy (only if exists), but warn via info

    Safety:
    - Never merges lon/lat into df.
    - out_col is CV-only (kept in dataset, not in predictors).
    """
    info: Dict[str, object] = {
        "spatial_block_action": "none",
        "spatial_block_policy": policy,
        "spatial_block_out_col": out_col,
        "spatial_block_source": "",
        "grid_km": grid_km,
        "nan_rate": None,
    }

    if policy not in {"off", "auto", "require"}:
        raise ValueError(f"Invalid spatial_block policy: {policy}")

    if policy == "off":
        info["spatial_block_action"] = "off"
        return df, info

    out = df.copy()
    if out_col in out.columns:
        info["spatial_block_action"] = "already_present"
        info["nan_rate"] = float(pd.to_numeric(out[out_col], errors="ignore").isna().mean()) if len(out) else 0.0
        return out, info

    if not panel_base_path or (not os.path.exists(panel_base_path)):
        info["spatial_block_action"] = "panel_base_missing"
        if policy == "require":
            raise RuntimeError("[spatial_block] require but panel_base missing.")
        return out, info

    base = pd.read_parquet(panel_base_path)
    if "cell_id" not in base.columns:
        info["spatial_block_action"] = "panel_base_no_cell_id"
        if policy == "require":
            raise RuntimeError("[spatial_block] require but panel_base has no cell_id.")
        return out, info

    prefer_cols = prefer_cols or ["grid10km_id", "spatial_block_id", "tile_id"]
    pick = next((c for c in prefer_cols if c in base.columns), "")

    if pick:
        b = base[["cell_id", pick]].copy()
        b = b.groupby("cell_id", as_index=False).agg({pick: "first"})
        b = b.rename(columns={pick: out_col})
        out = out.merge(b, on="cell_id", how="left")
        info["spatial_block_action"] = "merged_existing_id"
        info["spatial_block_source"] = pick
        info["nan_rate"] = float(out[out_col].isna().mean())
        if policy == "require" and (out[out_col].isna().any()):
            raise RuntimeError(f"[spatial_block] require but merged '{pick}' produced NaN in {out_col}.")
        return out, info

    # derive from lon/lat
    if ("lon" in base.columns) and ("lat" in base.columns):
        b = base[["cell_id", "lon", "lat"]].copy()
        b = b.groupby("cell_id", as_index=False).agg({"lon": "first", "lat": "first"})
        b[out_col] = make_grid_id_3857(b["lon"], b["lat"], grid_km=grid_km, prefix="g")
        b = b[["cell_id", out_col]]
        out = out.merge(b, on="cell_id", how="left")
        info["spatial_block_action"] = "derived_from_lonlat_3857"
        info["spatial_block_source"] = f"lonlat->3857 grid {grid_km}km"
        info["nan_rate"] = float(out[out_col].isna().mean())
        if policy == "require" and (out[out_col].isna().any()):
            raise RuntimeError(f"[spatial_block] require but derived {out_col} has NaN.")
        return out, info

    # fallback: STRATUM proxy
    if "STRATUM" in base.columns:
        b = base[["cell_id", "STRATUM"]].copy()
        b = b.groupby("cell_id", as_index=False).agg({"STRATUM": "first"}).rename(columns={"STRATUM": out_col})
        out = out.merge(b, on="cell_id", how="left")
        info["spatial_block_action"] = "fallback_stratum_proxy"
        info["spatial_block_source"] = "STRATUM"
        info["nan_rate"] = float(out[out_col].isna().mean())
        if policy == "require" and (out[out_col].isna().any()):
            raise RuntimeError(f"[spatial_block] require but STRATUM proxy produced NaN in {out_col}.")
        return out, info

    info["spatial_block_action"] = "no_source_available"
    if policy == "require":
        raise RuntimeError("[spatial_block] require but no usable source (id cols / lonlat / STRATUM) found in panel_base.")
    return out, info


# =========================
# Column classifiers
# =========================

def is_quality_col(c: str) -> bool:
    return (
        c.endswith(("_cov", "_missing", "_n"))
        or c.endswith(("_cov_lag1", "_missing_lag1", "_n_lag1"))
        or c in {"pix_count", "pix_count_lag1"}
    )

def is_quality_n_col(c: str) -> bool:
    return c.endswith(("_n", "_n_lag1"))

def is_dynamic_nowcast_value(c: str) -> bool:
    if c == "WS_strong_frac":
        return True
    if "_mon" in c and not c.endswith(("_cov", "_missing", "_n")):
        return True
    return False

def is_dynamic_lead1_value(c: str) -> bool:
    if not c.endswith("_lag1"):
        return False
    if c.endswith(("_cov_lag1", "_missing_lag1", "_n_lag1")):
        return False
    if c == "pix_count_lag1":
        return False
    return True


# =========================
# Expected core columns + *_n policy
# =========================

def expected_lead1_core_dyn() -> List[str]:
    return [c + "_lag1" for c in EXPECTED_DYNAMIC_BASE]

def expected_lead1_core_quality_no_n() -> List[str]:
    q: List[str] = []
    for base in EXPECTED_DYNAMIC_BASE:
        q += [f"{base}_cov_lag1", f"{base}_missing_lag1"]
    q += ["pix_count_lag1"]
    return q

def expected_lead1_core_quality_n_only() -> List[str]:
    return [f"{base}_n_lag1" for base in EXPECTED_DYNAMIC_BASE]

def expected_nowcast_core_dyn() -> List[str]:
    return list(EXPECTED_DYNAMIC_BASE)

def expected_nowcast_core_quality_no_n() -> List[str]:
    q: List[str] = []
    for base in EXPECTED_DYNAMIC_BASE:
        q += [f"{base}_cov", f"{base}_missing"]
    q += ["pix_count"]
    return q

def expected_nowcast_core_quality_n_only() -> List[str]:
    return [f"{base}_n" for base in EXPECTED_DYNAMIC_BASE]


# =========================
# Step6 feature hooks (optional)
# =========================

MEMORY_ALLOWED_PREFIX = ("mem_", "anom_", "acc_", "roll_", "clim_", "drought_", "firehist_")
MEMORY_ALLOWED_SUFFIX_RE = re.compile(
    r".*(_anom|_z|_pct|_p\d{2}|_roll\d+m_(sum|mean|max|min)|_(sum|mean|max|min)\d+m|_eventfreq_\d+m)$"
)

EXT_ALLOWED_PREFIX = ("x2_", "x3_", "inter_", "interaction_", "ext_", "longwin_")
EXT_ALLOWED_SUFFIX_RE = re.compile(r".*(_lag(2|3|6|12)|_roll(18|24|36)m_.*|_(p95|p99|iqr)|_nonlinear)$")


# =========================
# Feature sets
# =========================

@dataclass
class FeatureSets:
    F0: List[str]
    F1: List[str]
    F2: List[str]

@dataclass
class AdmissionDiagnostics:
    scenario: str
    admission_version: str

    f0_missing_core_dyn: List[str]
    f0_missing_core_quality: List[str]
    f2_missing_quality_n: List[str]

    f1_added_raw: List[str]
    f2_added_raw: List[str]

    f1_added_kept: List[str]
    f2_added_kept: List[str]

    dropped_constant_cols_by_set: Dict[str, List[str]]
    dropped_constant_cols: List[str]

    weight_cols_present: List[str]
    predictor_cols: Dict[str, List[str]]

    notes: Dict[str, object]


def list_static_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in STATIC_COLS_CAND if c in df.columns]

def list_strata_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in STRATA_COLS_CAND if c in df.columns]

def list_cv_block_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in CV_BLOCK_COLS_CAND if c in df.columns]

def list_weight_cols_present(df: pd.DataFrame) -> List[str]:
    return [c for c in WEIGHT_COLS_ALL if c in df.columns]

def infer_step6_memory_cols(df: pd.DataFrame, scenario: str) -> List[str]:
    cols: List[str] = []
    core_dyn = expected_lead1_core_dyn() if scenario == "lead1" else expected_nowcast_core_dyn()
    core_q0 = expected_lead1_core_quality_no_n() if scenario == "lead1" else expected_nowcast_core_quality_no_n()
    core_qn = expected_lead1_core_quality_n_only() if scenario == "lead1" else expected_nowcast_core_quality_n_only()
    core_all = set(core_dyn + core_q0 + core_qn)

    for c in df.columns:
        if c in KEY_COLS or c in SEASON_COLS or c == LABEL_COL:
            continue
        if c in STRATA_COLS_CAND or c in STATIC_COLS_CAND or c in WEIGHT_COLS_ALL:
            continue
        if c in CV_BLOCK_COLS_CAND:
            continue
        if c in core_all:
            continue
        if c in QC_ONLY_QUALITY_COLS:
            continue

        if is_allowed_firehist(c):
            cols.append(c)
            continue
        if c.startswith(MEMORY_ALLOWED_PREFIX) or MEMORY_ALLOWED_SUFFIX_RE.match(c):
            cols.append(c)

    return sorted(set(cols))

def infer_extended_cols(df: pd.DataFrame, already_in_f1: List[str]) -> List[str]:
    cols: List[str] = []
    s1 = set(already_in_f1)
    for c in df.columns:
        if c in s1:
            continue
        if c in KEY_COLS or c in SEASON_COLS or c == LABEL_COL:
            continue
        if c in STRATA_COLS_CAND or c in STATIC_COLS_CAND or c in WEIGHT_COLS_ALL:
            continue
        if c in CV_BLOCK_COLS_CAND:
            continue
        if c in QC_ONLY_QUALITY_COLS:
            continue
        if c.startswith(EXT_ALLOWED_PREFIX) or EXT_ALLOWED_SUFFIX_RE.match(c):
            cols.append(c)
    return sorted(set(cols))

def assert_lead1_no_nonlag_dyn_quality(df: pd.DataFrame) -> None:
    nonlag_dyn = [
        c for c in df.columns
        if (not c.endswith("_lag1")) and is_dynamic_nowcast_value(c) and (not is_quality_col(c))
    ]
    if nonlag_dyn:
        raise RuntimeError(
            "[lead1] Non-lag dynamic columns found (should not exist in lead1 panel). "
            f"examples={sorted(set(nonlag_dyn))[:30]}"
        )

def _canonical_added_kept_from_predictors(predictor_cols: Dict[str, List[str]]) -> Dict[str, List[str]]:
    S0 = set(predictor_cols.get("F0", []))
    S1 = set(predictor_cols.get("F1", []))
    S2 = set(predictor_cols.get("F2", []))
    return {
        "f1_added_kept": sorted(S1 - S0),
        "f2_added_kept": sorted(S2 - S1),
    }

def _audit_invariants(diag: AdmissionDiagnostics) -> None:
    dropped = set(diag.dropped_constant_cols)
    S0 = set(diag.predictor_cols["F0"])
    S1 = set(diag.predictor_cols["F1"])
    S2 = set(diag.predictor_cols["F2"])

    if (S0 & dropped) or (S1 & dropped) or (S2 & dropped):
        raise RuntimeError("[AUDIT] predictor_cols overlaps dropped_constant_cols (should never happen).")

    canon = _canonical_added_kept_from_predictors(diag.predictor_cols)
    if set(diag.f1_added_kept) != set(canon["f1_added_kept"]):
        raise RuntimeError("[AUDIT] f1_added_kept != canonical (F1\\F0) from predictor_cols.")
    if set(diag.f2_added_kept) != set(canon["f2_added_kept"]):
        raise RuntimeError("[AUDIT] f2_added_kept != canonical (F2\\F1) from predictor_cols.")


def build_feature_sets(
    df: pd.DataFrame,
    scenario: str,
    admission_version: str,
    strict_schema: bool = False,
) -> Tuple[FeatureSets, AdmissionDiagnostics]:
    cols = list(df.columns)
    assert_no_forbidden_cols(cols, where=f"input_{scenario}")

    must_cols = KEY_COLS + [LABEL_COL] + SEASON_COLS
    miss = [c for c in must_cols if c not in df.columns]
    if miss:
        raise KeyError(f"[{scenario}] Missing required columns: {miss}")

    strata = list_strata_cols(df)
    static = list_static_cols(df)
    weights_present = list_weight_cols_present(df)
    cv_blocks = list_cv_block_cols(df)  # CV-only keep cols

    if scenario == "lead1":
        assert_lead1_no_nonlag_dyn_quality(df)
        core_dyn = expected_lead1_core_dyn()
        core_q0 = expected_lead1_core_quality_no_n()
        core_qn = expected_lead1_core_quality_n_only()
    elif scenario == "nowcast":
        core_dyn = expected_nowcast_core_dyn()
        core_q0 = expected_nowcast_core_quality_no_n()
        core_qn = expected_nowcast_core_quality_n_only()
    else:
        raise ValueError("scenario must be one of: lead1, nowcast")

    missing_core_dyn = [c for c in core_dyn if c not in df.columns]
    missing_core_q0 = [c for c in core_q0 if c not in df.columns]
    missing_qn = [c for c in core_qn if c not in df.columns]

    if strict_schema and (missing_core_dyn or missing_core_q0):
        raise RuntimeError(
            f"[{scenario}] strict_schema failed: missing core dynamic/quality(F0 policy). "
            f"missing_dyn={missing_core_dyn[:20]} missing_quality={missing_core_q0[:20]}"
        )

    qc_only_cols = [c for c in core_q0 if c in df.columns and c in QC_ONLY_QUALITY_COLS]
    core_q0_predictor = [c for c in core_q0 if c in df.columns and c not in QC_ONLY_QUALITY_COLS]

    predictor_F0_raw = _dedup_keep_order(
        df,
        SEASON_COLS
        + static
        + [c for c in core_dyn if c in df.columns]
        + core_q0_predictor
    )

    f1_added = infer_step6_memory_cols(df, scenario=scenario)
    predictor_F1_raw = _dedup_keep_order(df, predictor_F0_raw + f1_added)

    qn_present = [c for c in core_qn if c in df.columns]
    f2_added_ext = infer_extended_cols(df, already_in_f1=predictor_F1_raw)
    predictor_F2_raw = _dedup_keep_order(df, predictor_F1_raw + qn_present + f2_added_ext)

    f1_added_raw = sorted(set([c for c in predictor_F1_raw if c not in predictor_F0_raw]))
    f2_added_raw = sorted(set([c for c in predictor_F2_raw if c not in predictor_F1_raw]))

    predictor_F0_kept, dropped0 = drop_constant_columns(df, predictor_F0_raw)
    predictor_F1_kept, dropped1 = drop_constant_columns(df, predictor_F1_raw)
    predictor_F2_kept, dropped2 = drop_constant_columns(df, predictor_F2_raw)

    dropped_by_set = {
        "F0": sorted(set(dropped0)),
        "F1": sorted(set(dropped1)),
        "F2": sorted(set(dropped2)),
    }
    dropped_all = sorted(set(dropped0 + dropped1 + dropped2))

    predictor_cols = {
        "F0": predictor_F0_kept,
        "F1": predictor_F1_kept,
        "F2": predictor_F2_kept,
    }

    # ✅ HARD rule: predictors must exclude QC-only columns
    assert_predictors_exclude_qc_only(predictor_cols, where=f"predictor_policy_{scenario}")

    canon_added = _canonical_added_kept_from_predictors(predictor_cols)
    f1_added_kept = canon_added["f1_added_kept"]
    f2_added_kept = canon_added["f2_added_kept"]

    # base_frame: kept in dataset, not necessarily predictors
    base_frame = KEY_COLS + [LABEL_COL] + strata + cv_blocks + weights_present + qc_only_cols

    F0 = _dedup_keep_order(df, base_frame + predictor_F0_kept)
    F1 = _dedup_keep_order(df, base_frame + predictor_F1_kept)
    F2 = _dedup_keep_order(df, base_frame + predictor_F2_kept)

    notes = {
        "memory_features_found": bool(len(f1_added) > 0),
        "no_memory_features_found_reason": (
            "expected before Step6; F1==F0 is OK" if len(f1_added) == 0 else ""
        ),
        "cv_blocks_kept": cv_blocks,
    }

    diag = AdmissionDiagnostics(
        scenario=scenario,
        admission_version=admission_version,
        f0_missing_core_dyn=missing_core_dyn,
        f0_missing_core_quality=missing_core_q0,
        f2_missing_quality_n=missing_qn,
        f1_added_raw=f1_added_raw,
        f2_added_raw=f2_added_raw,
        f1_added_kept=f1_added_kept,
        f2_added_kept=f2_added_kept,
        dropped_constant_cols_by_set=dropped_by_set,
        dropped_constant_cols=dropped_all,
        weight_cols_present=weights_present,
        predictor_cols=predictor_cols,
        notes=notes,
    )

    _audit_invariants(diag)
    return FeatureSets(F0=F0, F1=F1, F2=F2), diag


# =========================
# Panel integrity checks
# =========================

def assert_month_start_dates(df: pd.DataFrame, scenario: str) -> None:
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        raise RuntimeError(f"[{scenario}] date must be datetime64 after coercion, got {df['date'].dtype}")
    bad = df[df["date"].dt.day != 1]
    if len(bad) > 0:
        ex = bad[["cell_id", "date"]].head(10).to_dict(orient="records")
        raise RuntimeError(f"[{scenario}] date must be month-start (day==1). Examples: {ex}")

def assert_y_binary(df: pd.DataFrame, scenario: str) -> None:
    if df[LABEL_COL].isna().any():
        raise RuntimeError(f"[{scenario}] y has NaN values.")
    vals = set(pd.unique(df[LABEL_COL]))
    if not vals.issubset({0, 1}):
        raise RuntimeError(f"[{scenario}] y must be binary {{0,1}}. Found: {sorted(list(vals))[:20]}")

def assert_panel_integrity(df: pd.DataFrame, scenario: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    assert_month_start_dates(df, scenario)
    assert_y_binary(df, scenario)

    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)

    dup_n = int(df.duplicated(subset=["cell_id", "date"]).sum())
    out["dup_cell_date_n"] = dup_n
    if dup_n > 0:
        raise RuntimeError(f"[{scenario}] duplicated (cell_id, date) rows: n={dup_n}")

    diffs = df.groupby("cell_id", sort=False)["date"].diff()
    bad_mon = df.loc[diffs.notna() & (diffs <= pd.Timedelta(0)), ["cell_id", "date"]].head(20)
    out["non_monotonic_rows_n"] = int((diffs.notna() & (diffs <= pd.Timedelta(0))).sum())
    if out["non_monotonic_rows_n"] > 0:
        ex = bad_mon.to_dict(orient="records")
        raise RuntimeError(f"[{scenario}] date is not strictly increasing within some cells. Examples: {ex}")

    exp = 119 if scenario == "lead1" else 120
    cnt = df.groupby("cell_id")["date"].nunique()
    bad = cnt[cnt != exp]

    out["n_rows"] = int(df.shape[0])
    out["n_cells"] = int(df["cell_id"].nunique())
    out["date_min"] = str(df["date"].min().date())
    out["date_max"] = str(df["date"].max().date())
    out["months_expected_per_cell"] = exp
    out["months_per_cell_min"] = int(cnt.min())
    out["months_per_cell_median"] = float(cnt.median())
    out["months_per_cell_max"] = int(cnt.max())
    out["bad_cells_n"] = int(len(bad))

    if len(bad) > 0:
        raise RuntimeError(f"[{scenario}] Months per cell != {exp}. examples={bad.head(10).to_dict()}")

    return out


# =========================
# QC: weights + labels
# =========================

def qc_weights(df: pd.DataFrame, scenario: str) -> List[Dict[str, object]]:
    rows = []
    for c in WEIGHT_COLS_ALL:
        if c in df.columns:
            stats = numeric_summary(df[c])
            stats["n_unique"] = float(n_unique_nonnull(df[c]))
            stats["is_constant"] = float(1.0 if (n_unique_nonnull(df[c]) <= 1) else 0.0)
            rows.append({"scenario": scenario, "col": c, **stats})
    return rows

def qc_labels(df: pd.DataFrame, scenario: str) -> Dict[str, object]:
    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    return {
        "y_rate_overall": float(y.mean()),
        "y_pos_n": int((y == 1).sum()),
        "y_neg_n": int((y == 0).sum()),
        "y_rate_by_year": {str(int(k)): float(v) for k, v in df.groupby("year")[LABEL_COL].mean().to_dict().items()},
        "y_rate_by_month": {str(int(k)): float(v) for k, v in df.groupby("month")[LABEL_COL].mean().to_dict().items()},
    }


# =========================
# QC: lead1 alignment (optional; needs nowcast)
# =========================

def _is_int_like_series(s: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(s.dtype):
        return True
    x = pd.to_numeric(s, errors="coerce")
    xf = x.dropna()
    if len(xf) == 0:
        return False
    return bool(np.all(np.isclose(xf.to_numpy(), np.round(xf.to_numpy()), atol=1e-9, rtol=0.0)))

def qc_lead1_alignment(
    lead1_df: pd.DataFrame,
    nowcast_df: pd.DataFrame,
    out_csv: str,
    out_md: str,
    sample_cells: int = 200,
    sample_months: int = 24,
    min_match: float = 0.999,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> Dict[str, object]:
    req = {"cell_id", "date"}
    if not req.issubset(set(lead1_df.columns)) or not req.issubset(set(nowcast_df.columns)):
        raise RuntimeError("[align] lead1/nowcast missing required key columns cell_id/date")

    lag_cols = sorted([c for c in lead1_df.columns if c.endswith("_lag1")])
    if not lag_cols:
        raise RuntimeError("[align] no *_lag1 columns found in lead1")

    base_cols = [c[:-5] for c in lag_cols]
    present_pairs = [(lc, bc) for lc, bc in zip(lag_cols, base_cols) if bc in nowcast_df.columns]
    if not present_pairs:
        raise RuntimeError("[align] no matching base columns found in nowcast for lead1 *_lag1 columns")

    cells = sorted(lead1_df["cell_id"].astype(str).unique().tolist())
    if sample_cells and sample_cells > 0:
        cells = cells[: min(sample_cells, len(cells))]

    l = lead1_df.loc[lead1_df["cell_id"].astype(str).isin(cells), ["cell_id", "date"] + [p[0] for p in present_pairs]].copy()
    n = nowcast_df.loc[nowcast_df["cell_id"].astype(str).isin(cells), ["cell_id", "date"] + [p[1] for p in present_pairs]].copy()

    n["date"] = (pd.to_datetime(n["date"]) + pd.offsets.MonthBegin(1))

    j = l.merge(n, on=["cell_id", "date"], how="inner", suffixes=("", "_base"))
    if j.empty:
        raise RuntimeError("[align] alignment join produced 0 rows (check date ranges)")

    j = j.sort_values(["cell_id", "date"]).reset_index(drop=True)
    if sample_months and sample_months > 0:
        j["_rk"] = j.groupby("cell_id", sort=False).cumcount(ascending=False)
        j = j.loc[j["_rk"] < sample_months].drop(columns=["_rk"])

    rows = []
    examples_lines = []
    n_pairs_checked = int(j[["cell_id", "date"]].drop_duplicates().shape[0])

    for lag_c, base_c in present_pairs:
        a = j[lag_c]
        b = j[base_c]

        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")

        mask = a_num.notna() & b_num.notna()
        if mask.sum() == 0:
            rows.append({
                "col_lag1": lag_c,
                "col_base": base_c,
                "n_compared": 0,
                "match_rate": float("nan"),
                "max_abs_diff": float("nan"),
                "n_mismatch": 0,
                "note": "no_overlap_nonnull",
            })
            continue

        is_int_like = _is_int_like_series(a_num[mask]) and _is_int_like_series(b_num[mask])

        if is_int_like:
            eq = (np.round(a_num[mask].to_numpy()) == np.round(b_num[mask].to_numpy()))
            diff = np.abs(np.round(a_num[mask].to_numpy()) - np.round(b_num[mask].to_numpy()))
        else:
            eq = np.isclose(a_num[mask].to_numpy(), b_num[mask].to_numpy(), rtol=rtol, atol=atol, equal_nan=False)
            diff = np.abs(a_num[mask].to_numpy() - b_num[mask].to_numpy())

        n_comp = int(mask.sum())
        n_mis = int((~eq).sum())
        match_rate = float(eq.mean())
        max_abs_diff = float(np.nanmax(diff)) if n_comp > 0 else float("nan")

        rows.append({
            "col_lag1": lag_c,
            "col_base": base_c,
            "n_compared": n_comp,
            "match_rate": match_rate,
            "max_abs_diff": max_abs_diff,
            "n_mismatch": n_mis,
            "is_int_like": int(is_int_like),
        })

        if n_mis > 0:
            mis_idx = np.where(~eq)[0][:3]
            sub = j.loc[mask].iloc[mis_idx][["cell_id", "date"]].copy()
            sub["lead1"] = a_num[mask].iloc[mis_idx].to_numpy()
            sub["nowcast_t-1"] = b_num[mask].iloc[mis_idx].to_numpy()
            examples_lines.append(f"- **{lag_c}** mismatches: {sub.to_dict(orient='records')}")

    df_out = pd.DataFrame(rows).sort_values(["match_rate", "col_lag1"], ascending=[True, True])
    df_out.to_csv(out_csv, index=False)

    bad = df_out[df_out["match_rate"].notna() & (df_out["match_rate"] < min_match)]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Lead1 alignment QC report\n\n")
        f.write(f"- n_pairs_checked: {n_pairs_checked}\n")
        f.write(f"- sample_cells: {sample_cells}\n")
        f.write(f"- sample_months: {sample_months}\n")
        f.write(f"- min_match: {min_match}\n")
        f.write(f"- rtol/atol: {rtol}/{atol}\n\n")

        if len(bad) > 0:
            f.write("## Worst columns (match_rate)\n\n")
            f.write(bad.to_markdown(index=False))
            f.write("\n\n## Examples\n\n")
            f.write("\n".join(examples_lines[:50]) if examples_lines else "No mismatch examples captured.\n")
        else:
            f.write("## Summary (lowest match_rate first)\n\n")
            f.write(df_out.head(30).to_markdown(index=False))
            f.write("\n\n## Mismatch examples\n\n")
            f.write("\n".join(examples_lines[:50]) if examples_lines else "No mismatches.\n")

    if len(bad) > 0:
        bad_cols = bad[["col_lag1", "match_rate", "max_abs_diff", "n_mismatch"]].head(30).to_dict(orient="records")
        raise RuntimeError(f"[align] lead1 alignment failed. bad_cols={bad_cols}")

    return {
        "n_pairs_checked": n_pairs_checked,
        "n_cols_checked": int(df_out.shape[0]),
        "min_match_rate": float(df_out["match_rate"].min()),
        "max_abs_diff_max": float(df_out["max_abs_diff"].max()),
    }


# =========================
# metadata_cells (CV-only)
# =========================

def maybe_write_metadata_cells(
    panel_base_path: Optional[str],
    out_path: str,
    include_strata: bool = True,
) -> Optional[pd.DataFrame]:
    if not panel_base_path or (not os.path.exists(panel_base_path)):
        return None
    dfb = pd.read_parquet(panel_base_path)
    if "cell_id" not in dfb.columns:
        return None

    cols = ["cell_id"]
    # keep lon/lat only in metadata_cells (NOT in main panel/featuresets)
    if "lon" in dfb.columns and "lat" in dfb.columns:
        cols += ["lon", "lat"]

    # keep spatial block id if present (or derive later downstream)
    for c in CV_BLOCK_COLS_CAND:
        if c in dfb.columns:
            cols.append(c)

    if include_strata:
        for c in STRATA_COLS_CAND:
            if c in dfb.columns:
                cols.append(c)

    dfm = dfb[cols].copy()
    agg = {c: "first" for c in dfm.columns if c != "cell_id"}
    dfm = dfm.groupby("cell_id", as_index=False).agg(agg)
    dfm.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    return dfm

def qc_metadata_cells(dfm: pd.DataFrame, expected_n_cells: int) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["n_rows"] = int(dfm.shape[0])
    out["n_cells"] = int(dfm["cell_id"].nunique())
    out["expected_n_cells"] = int(expected_n_cells)
    out["n_cells_matches_expected"] = bool(out["n_cells"] == out["expected_n_cells"])

    dup = int(dfm.duplicated(subset=["cell_id"]).sum())
    out["dup_cell_id_n"] = dup
    if dup > 0:
        raise RuntimeError(f"[metadata_cells] duplicated cell_id rows: n={dup}")

    if "lon" in dfm.columns and "lat" in dfm.columns:
        out["lon_nan_rate"] = float(dfm["lon"].isna().mean())
        out["lat_nan_rate"] = float(dfm["lat"].isna().mean())
        out["lon_out_of_range_n"] = int(((dfm["lon"] < -180) | (dfm["lon"] > 180)).sum(skipna=True))
        out["lat_out_of_range_n"] = int(((dfm["lat"] < -90) | (dfm["lat"] > 90)).sum(skipna=True))

    # qc block id if present
    for c in CV_BLOCK_COLS_CAND:
        if c in dfm.columns:
            out[f"{c}_nan_rate"] = float(dfm[c].isna().mean())

    strata_cols = [c for c in STRATA_COLS_CAND if c in dfm.columns]
    out["strata_cols"] = strata_cols
    for c in strata_cols:
        out[f"{c}_nan_rate"] = float(dfm[c].isna().mean())
    return out


# =========================
# Artifacts
# =========================

def write_leakage_blacklist_txt(path: str) -> None:
    lines = []
    lines.append("# Leakage blacklist (Step5) — audit artifact\n\n")
    lines.append("## Forbidden exact columns\n")
    for c in sorted(FORBIDDEN_EXACT):
        lines.append(f"- {c}\n")
    lines.append("\n## Forbidden substrings / regex patterns (defensive)\n")
    for p in FORBIDDEN_SUBSTR:
        lines.append(f"- {p}\n")
    lines.append("\n## Burned-area rule\n")
    lines.append("- Any column containing 'burned_area' is forbidden unless explicitly windowed past-only fire-history.\n")
    lines.append("\n## Allowed past-only fire-history patterns\n")
    for p in ALLOWED_FIREHIST_PATTERNS:
        lines.append(f"- {p}\n")
    lines.append("\n## Predictor hard exclusion (QC-only)\n")
    lines.append("- Predictors must NOT include: pi/weight/d_weight*/valid*/pix_count*\n")
    lines.append(f"- Regex: {QC_ONLY_PREDICTOR_RE.pattern}\n")
    lines.append("\n## CV-only columns\n")
    lines.append("- CV-only spatial block id columns are kept in dataset but never predictors.\n")
    for c in CV_BLOCK_COLS_CAND:
        lines.append(f"- {c}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def _yaml_dump_minimal(obj: dict) -> str:
    def esc(s: str) -> str:
        if re.search(r"[:#\-\{\}\[\],&\*\!\|\>\'\"\%@`]", s) or s.strip() != s or " " in s:
            return '"' + s.replace('"', '\\"') + '"'
        return s

    def dump(v, indent: int = 0) -> str:
        sp = "  " * indent
        if isinstance(v, dict):
            out = ""
            for k, vv in v.items():
                if isinstance(vv, (dict, list)):
                    out += f"{sp}{k}:\n{dump(vv, indent + 1)}"
                else:
                    out += f"{sp}{k}: {esc(str(vv))}\n"
            return out
        if isinstance(v, list):
            out = ""
            for item in v:
                out += f"{sp}- {esc(str(item))}\n"
            return out
        return f"{sp}{esc(str(v))}\n"

    return dump(obj)

def write_feature_sets_yaml(
    path: str,
    admission_version: str,
    lead1_predictors: Dict[str, List[str]],
    nowcast_predictors: Optional[Dict[str, List[str]]] = None,
) -> None:
    cfg = {
        "admission_version": admission_version,
        "notes": {
            "mainline_modeling": "Use lead1_* only (predict month t using <=t-1 dynamic info via *_lag1).",
            "nowcast": "Diagnostic only; do NOT use for mainline claims.",
            "n_policy": "F0/F1 exclude *_n; F2 may include *_n as quality_n.",
            "pix_count_policy": "pix_count is QC-only (kept in dataset, never predictors).",
            "valid_core_policy": "valid_core_pass/raw are QC/weights only; never predictors.",
            "predictor_qc_only_policy": "predictors exclude pi/weight/d_weight*/valid*/pix_count* (hard fail).",
            "F1_note": "If Step6 memory/anom/acc/firehist columns do not exist yet, F1==F0 is expected.",
            "cv_block_policy": "grid10km_id (or similar) is kept for Step7 P2; NEVER used as predictor.",
        },
        "lead1": {
            "F0": lead1_predictors.get("F0", []),
            "F1": lead1_predictors.get("F1", []),
            "F2": lead1_predictors.get("F2", []),
        },
    }
    if nowcast_predictors:
        cfg["nowcast_diagnostic"] = {
            "F0": nowcast_predictors.get("F0", []),
            "F1": nowcast_predictors.get("F1", []),
            "F2": nowcast_predictors.get("F2", []),
        }
    with open(path, "w", encoding="utf-8") as f:
        f.write(_yaml_dump_minimal(cfg))


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead1", default="data/processed/panel_lead1.parquet")
    parser.add_argument("--nowcast", default="")
    parser.add_argument("--out_root", default=".")
    parser.add_argument("--hash_algo", choices=["sha256", "md5"], default="sha256")
    parser.add_argument("--admission_version", default="v1.0.6")
    parser.add_argument("--strict_schema", action="store_true")
    parser.add_argument("--panel_base", default="data/interim/panel_base.parquet")
    parser.add_argument("--export_metadata_cells", action="store_true")
    parser.add_argument("--metadata_include_strata", action="store_true")

    # spatial block id options
    parser.add_argument("--spatial_block_policy", choices=["off", "auto", "require"], default="auto")
    parser.add_argument("--spatial_block_out_col", default="grid10km_id")
    parser.add_argument("--grid_km", type=float, default=10.0)

    # alignment qc knobs
    parser.add_argument("--align_sample_cells", type=int, default=200)
    parser.add_argument("--align_sample_months", type=int, default=24)
    parser.add_argument("--align_min_match", type=float, default=0.999)
    parser.add_argument("--align_rtol", type=float, default=1e-6)
    parser.add_argument("--align_atol", type=float, default=1e-8)

    args = parser.parse_args()

    out_processed = os.path.join(args.out_root, "data", "processed")
    out_interim = os.path.join(args.out_root, "data", "interim")
    out_artifacts = os.path.join(args.out_root, "artifacts")
    out_configs = os.path.join(args.out_root, "configs")
    out_fs = os.path.join(out_processed, "featuresets")

    ensure_dir(out_fs)
    ensure_dir(out_interim)
    ensure_dir(out_artifacts)
    ensure_dir(out_configs)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ---- load lead1
    if not os.path.exists(args.lead1):
        raise FileNotFoundError(f"lead1 panel not found: {args.lead1}")
    df_lead1 = pd.read_parquet(args.lead1)
    df_lead1 = coerce_date_column(df_lead1, "lead1")

    df_lead1, vc_info_lead1 = maybe_merge_valid_core_from_base(args.panel_base, df_lead1)

    # merge spatial block id (CV-only)
    df_lead1, sb_info_lead1 = maybe_merge_spatial_block_from_base(
        args.panel_base,
        df_lead1,
        out_col=args.spatial_block_out_col,
        policy=args.spatial_block_policy,
        grid_km=float(args.grid_km),
    )

    # drop leak-prone proxy if present
    df_lead1 = df_lead1.drop(columns=["valid_weight"], errors="ignore")

    df_lead1 = enforce_weight_columns(df_lead1)

    lead1_int = assert_panel_integrity(df_lead1, "lead1")
    fs_lead1, diag_lead1 = build_feature_sets(
        df_lead1, scenario="lead1", admission_version=args.admission_version, strict_schema=args.strict_schema
    )

    # ---- optional nowcast
    df_now = None
    now_int = None
    fs_now = None
    diag_now = None
    vc_info_now = None
    sb_info_now = None

    if args.nowcast:
        if not os.path.exists(args.nowcast):
            raise FileNotFoundError(f"nowcast panel not found: {args.nowcast}")
        df_now = pd.read_parquet(args.nowcast)
        df_now = coerce_date_column(df_now, "nowcast")
        df_now, vc_info_now = maybe_merge_valid_core_from_base(args.panel_base, df_now)

        df_now, sb_info_now = maybe_merge_spatial_block_from_base(
            args.panel_base,
            df_now,
            out_col=args.spatial_block_out_col,
            policy=args.spatial_block_policy,
            grid_km=float(args.grid_km),
        )

        df_now = df_now.drop(columns=["valid_weight"], errors="ignore")
        df_now = enforce_weight_columns(df_now)
        now_int = assert_panel_integrity(df_now, "nowcast")
        fs_now, diag_now = build_feature_sets(
            df_now, scenario="nowcast", admission_version=args.admission_version, strict_schema=args.strict_schema
        )

    # ---- optional alignment QC (only if nowcast provided)
    qc_align_path = ""
    qc_align_md = ""
    qc_align_summary: Dict[str, object] = {}
    if df_now is not None:
        qc_align_path = os.path.join(out_interim, "qc_lead1_alignment.csv")
        qc_align_md = os.path.join(out_artifacts, "qc_lead1_alignment_report.md")
        qc_align_summary = qc_lead1_alignment(
            lead1_df=df_lead1,
            nowcast_df=df_now,
            out_csv=qc_align_path,
            out_md=qc_align_md,
            sample_cells=args.align_sample_cells,
            sample_months=args.align_sample_months,
            min_match=args.align_min_match,
            rtol=args.align_rtol,
            atol=args.align_atol,
        )

    # ---- write featuresets
    paths: Dict[str, str] = {}
    out_hashes: Dict[str, str] = {}

    def write_one(tag: str, df: pd.DataFrame, cols: List[str]) -> None:
        p = os.path.join(out_fs, f"{tag}.parquet")
        materialize(df, cols).to_parquet(p, index=False, engine="pyarrow", compression="snappy")
        paths[tag] = p
        out_hashes[tag] = file_hash(p, algo=args.hash_algo)

    write_one("lead1_F0", df_lead1, fs_lead1.F0)
    write_one("lead1_F1", df_lead1, fs_lead1.F1)
    write_one("lead1_F2", df_lead1, fs_lead1.F2)
    if df_now is not None and fs_now is not None:
        write_one("nowcast_F0", df_now, fs_now.F0)
        write_one("nowcast_F1", df_now, fs_now.F1)
        write_one("nowcast_F2", df_now, fs_now.F2)

    # ---- QC outputs
    qc_rows: List[Dict[str, object]] = []

    def qc_row(scenario: str, df: pd.DataFrame, diag: AdmissionDiagnostics, featureset: str) -> Dict[str, object]:
        cols = diag.predictor_cols[featureset]
        if scenario == "lead1":
            dyn_count = sum(is_dynamic_lead1_value(c) for c in cols)
        else:
            dyn_count = sum(is_dynamic_nowcast_value(c) and (not is_quality_col(c)) for c in cols)
        q_count = sum(is_quality_col(c) for c in cols)
        qn_count = sum(is_quality_n_col(c) for c in cols)
        fs_obj = fs_lead1 if scenario == "lead1" else fs_now
        return {
            "scenario": scenario,
            "featureset": featureset,
            "admission_version": diag.admission_version,
            "run_id": run_id,
            "n_rows": int(df.shape[0]),
            "n_cols_dataset": int(len(getattr(fs_obj, featureset))),
            "n_predictors": int(len(cols)),
            "n_dynamic_predictors": int(dyn_count),
            "n_quality_predictors": int(q_count),
            "n_quality_n_predictors": int(qn_count),
            "missing_core_dyn_n": int(len(diag.f0_missing_core_dyn)) if featureset == "F0" else 0,
            "missing_core_quality_no_n_n": int(len(diag.f0_missing_core_quality)) if featureset == "F0" else 0,
            "missing_quality_n_n": int(len(diag.f2_missing_quality_n)) if featureset == "F2" else 0,
            "f1_added_kept_n": int(len(diag.f1_added_kept)) if featureset == "F1" else 0,
            "f2_added_kept_n": int(len(diag.f2_added_kept)) if featureset == "F2" else 0,
            "f1_added_raw_n": int(len(diag.f1_added_raw)) if featureset == "F1" else 0,
            "f2_added_raw_n": int(len(diag.f2_added_raw)) if featureset == "F2" else 0,
            "dropped_constant_cols_n": int(len(diag.dropped_constant_cols)),
            "weights_present": "|".join(diag.weight_cols_present),
            "F1_equals_F0": int(set(diag.predictor_cols["F1"]) == set(diag.predictor_cols["F0"])),
            "F2_equals_F0": int(set(diag.predictor_cols["F2"]) == set(diag.predictor_cols["F0"])),
            "memory_features_found": int(bool(diag.notes.get("memory_features_found", False))),
            "cv_block_cols_kept": "|".join(diag.notes.get("cv_blocks_kept", [])),
        }

    qc_rows += [
        qc_row("lead1", df_lead1, diag_lead1, "F0"),
        qc_row("lead1", df_lead1, diag_lead1, "F1"),
        qc_row("lead1", df_lead1, diag_lead1, "F2"),
    ]
    if df_now is not None and diag_now is not None and fs_now is not None:
        qc_rows += [
            qc_row("nowcast", df_now, diag_now, "F0"),
            qc_row("nowcast", df_now, diag_now, "F1"),
            qc_row("nowcast", df_now, diag_now, "F2"),
        ]
    qc_featuresets_path = os.path.join(out_interim, "qc_featuresets.csv")
    pd.DataFrame(qc_rows).to_csv(qc_featuresets_path, index=False)

    rows_w = qc_weights(df_lead1, "lead1")
    if df_now is not None:
        rows_w += qc_weights(df_now, "nowcast")
    qc_weights_path = os.path.join(out_interim, "qc_weights.csv")
    pd.DataFrame(rows_w).to_csv(qc_weights_path, index=False)

    rows_l = []
    lab_lead1 = qc_labels(df_lead1, "lead1")
    rows_l.append({"scenario": "lead1", "run_id": run_id, **{k: lab_lead1[k] for k in ["y_rate_overall", "y_pos_n", "y_neg_n"]}})
    if df_now is not None:
        lab_now = qc_labels(df_now, "nowcast")
        rows_l.append({"scenario": "nowcast", "run_id": run_id, **{k: lab_now[k] for k in ["y_rate_overall", "y_pos_n", "y_neg_n"]}})
    qc_labels_path = os.path.join(out_interim, "qc_labels.csv")
    pd.DataFrame(rows_l).to_csv(qc_labels_path, index=False)

    def diag_to_row(d: AdmissionDiagnostics) -> Dict[str, object]:
        return {
            "scenario": d.scenario,
            "run_id": run_id,
            "admission_version": d.admission_version,
            "missing_core_dyn": "|".join(d.f0_missing_core_dyn),
            "missing_core_quality_no_n": "|".join(d.f0_missing_core_quality),
            "missing_quality_n": "|".join(d.f2_missing_quality_n),
            "f1_added_raw": "|".join(d.f1_added_raw),
            "f1_added_kept": "|".join(d.f1_added_kept),
            "f2_added_raw": "|".join(d.f2_added_raw),
            "f2_added_kept": "|".join(d.f2_added_kept),
            "dropped_constant_cols_F0": "|".join(d.dropped_constant_cols_by_set.get("F0", [])),
            "dropped_constant_cols_F1": "|".join(d.dropped_constant_cols_by_set.get("F1", [])),
            "dropped_constant_cols_F2": "|".join(d.dropped_constant_cols_by_set.get("F2", [])),
            "dropped_constant_cols_union": "|".join(d.dropped_constant_cols),
            "weights_present": "|".join(d.weight_cols_present),
            "predictor_cols_F0_n": len(d.predictor_cols["F0"]),
            "predictor_cols_F1_n": len(d.predictor_cols["F1"]),
            "predictor_cols_F2_n": len(d.predictor_cols["F2"]),
            "memory_features_found": int(bool(d.notes.get("memory_features_found", False))),
            "memory_note": d.notes.get("no_memory_features_found_reason", ""),
            "cv_block_cols_kept": "|".join(d.notes.get("cv_blocks_kept", [])),
        }

    qc_admission_path = os.path.join(out_interim, "qc_admission.csv")
    pd.DataFrame([diag_to_row(diag_lead1)] + ([diag_to_row(diag_now)] if diag_now else [])).to_csv(qc_admission_path, index=False)

    const_rows: List[Dict[str, object]] = []
    for scenario, diag in [("lead1", diag_lead1), ("nowcast", diag_now)]:
        if diag is None:
            continue
        for fs_name in ["F0", "F1", "F2"]:
            for c in diag.dropped_constant_cols_by_set.get(fs_name, []):
                const_rows.append({"scenario": scenario, "featureset": fs_name, "col": c, "run_id": run_id})
    qc_constants_path = os.path.join(out_interim, "qc_constants.csv")
    pd.DataFrame(const_rows).to_csv(qc_constants_path, index=False)

    # metadata_cells (optional)
    meta_path = ""
    qc_metadata_path = ""
    meta_hash = ""
    meta_qc = None
    if args.export_metadata_cells:
        meta_path = os.path.join(out_interim, "metadata_cells.parquet")
        dfm = maybe_write_metadata_cells(args.panel_base, meta_path, include_strata=args.metadata_include_strata)
        if dfm is not None:
            meta_qc = qc_metadata_cells(dfm, expected_n_cells=lead1_int["n_cells"])
            qc_metadata_path = os.path.join(out_interim, "qc_metadata.csv")
            pd.DataFrame([meta_qc]).to_csv(qc_metadata_path, index=False)
            meta_hash = file_hash(meta_path, algo=args.hash_algo)
        else:
            meta_path = ""
            qc_metadata_path = ""

    # required artifacts
    blacklist_path = os.path.join(out_artifacts, "leakage_blacklist.txt")
    feature_sets_yaml_path = os.path.join(out_configs, "feature_sets.yaml")
    write_leakage_blacklist_txt(blacklist_path)
    write_feature_sets_yaml(
        feature_sets_yaml_path,
        admission_version=args.admission_version,
        lead1_predictors=diag_lead1.predictor_cols,
        nowcast_predictors=(diag_now.predictor_cols if diag_now else None),
    )

    canon_added_lead1 = _canonical_added_kept_from_predictors(diag_lead1.predictor_cols)
    canon_added_now = _canonical_added_kept_from_predictors(diag_now.predictor_cols) if diag_now else {"f1_added_kept": [], "f2_added_kept": []}

    manifest = {
        "run": {
            "run_id": run_id,
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "args": vars(args),
        "inputs": {
            "lead1_path": args.lead1,
            "lead1_hash": file_hash(args.lead1, algo=args.hash_algo),
            "nowcast_path": args.nowcast if args.nowcast else "",
            "nowcast_hash": file_hash(args.nowcast, algo=args.hash_algo) if args.nowcast else "",
            "hash_algo": args.hash_algo,
            "panel_base_path": args.panel_base,
            "panel_base_hash": file_hash(args.panel_base, algo=args.hash_algo) if (args.panel_base and os.path.exists(args.panel_base)) else "",
        },
        "outputs": {
            **paths,
            "qc_featuresets": qc_featuresets_path,
            "qc_weights": qc_weights_path,
            "qc_labels": qc_labels_path,
            "qc_admission": qc_admission_path,
            "qc_constants": qc_constants_path,
            "qc_metadata": qc_metadata_path,
            "metadata_cells": meta_path,
            "leakage_blacklist": blacklist_path,
            "feature_sets_yaml": feature_sets_yaml_path,
            "qc_lead1_alignment": qc_align_path,
            "qc_lead1_alignment_report": qc_align_md,
        },
        "output_hashes": {
            **out_hashes,
            "leakage_blacklist": file_hash(blacklist_path, algo=args.hash_algo),
            "feature_sets_yaml": file_hash(feature_sets_yaml_path, algo=args.hash_algo),
            "metadata_cells": meta_hash,
            "qc_metadata": (file_hash(qc_metadata_path, algo=args.hash_algo) if qc_metadata_path else ""),
            "qc_lead1_alignment": (file_hash(qc_align_path, algo=args.hash_algo) if qc_align_path else ""),
            "qc_lead1_alignment_report": (file_hash(qc_align_md, algo=args.hash_algo) if qc_align_md else ""),
        },
        "integrity": {"lead1": lead1_int, "nowcast": now_int if now_int else {}},
        "valid_core_handling": {"lead1": vc_info_lead1, "nowcast": vc_info_now if vc_info_now else {}},
        "spatial_block_handling": {"lead1": sb_info_lead1, "nowcast": sb_info_now if sb_info_now else {}},
        "predictor_cols": {"lead1": diag_lead1.predictor_cols, "nowcast": diag_now.predictor_cols if diag_now else {}},
        "diagnostics": {
            "lead1": {
                "f1_added_kept": canon_added_lead1["f1_added_kept"],
                "f2_added_kept": canon_added_lead1["f2_added_kept"],
                "dropped_constant_cols_by_set": diag_lead1.dropped_constant_cols_by_set,
            },
            "nowcast": {
                "f1_added_kept": canon_added_now["f1_added_kept"],
                "f2_added_kept": canon_added_now["f2_added_kept"],
                "dropped_constant_cols_by_set": diag_now.dropped_constant_cols_by_set if diag_now else {},
            },
        },
        "qc_metadata_summary": meta_qc if meta_qc else {},
        "qc_lead1_alignment_summary": qc_align_summary,
    }

    man_path = os.path.join(out_artifacts, "featuresets_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("✅ Step5 done (v1.0.6).")
    print(f"- run_id: {run_id}")
    print(f"- admission_version: {args.admission_version}")
    print(f"- spatial_block_policy: {args.spatial_block_policy} out_col={args.spatial_block_out_col} grid_km={args.grid_km}")
    print(f"- lead1_F0/F1/F2: {paths['lead1_F0']}, {paths['lead1_F1']}, {paths['lead1_F2']}")
    if args.nowcast:
        print(f"- nowcast_F0/F1/F2: {paths['nowcast_F0']}, {paths['nowcast_F1']}, {paths['nowcast_F2']} (diagnostic only)")
        print(f"- qc_lead1_alignment: {qc_align_path}")
        print(f"- qc_lead1_alignment_report: {qc_align_md}")
    print(f"- qc_featuresets: {qc_featuresets_path}")
    print(f"- qc_admission:   {qc_admission_path}")
    print(f"- qc_weights:     {qc_weights_path}")
    print(f"- qc_labels:      {qc_labels_path}")
    print(f"- qc_constants:   {qc_constants_path}")
    if qc_metadata_path:
        print(f"- qc_metadata:    {qc_metadata_path}")
    print(f"- manifest:       {man_path}")
    print(f"- blacklist:      {blacklist_path}")
    print(f"- feature_sets:   {feature_sets_yaml_path}")
    if meta_path:
        print(f"- metadata_cells: {meta_path} (hash={meta_hash}) (CV-only)")
    for k, v in paths.items():
        print(f"- {k}: {v} (hash={out_hashes.get(k,'')})")


if __name__ == "__main__":
    main()