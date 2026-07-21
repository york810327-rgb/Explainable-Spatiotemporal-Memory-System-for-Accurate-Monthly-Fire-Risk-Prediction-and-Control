#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_featuresets (Step5.5) — Feature set admission + versioning (F0/F1/F2)

Design goals (paper-friendly):
- Step5.5 ONLY does: column admission + versioning + manifest/QC.
- No training-only transforms; no anomaly/climatology; no imputation.

Key enhancement (for Step7 P2/P3):
- Keep CV-only spatial block cols in output dataset (NEVER predictors):
  grid10km_id / grid50km_id / grid100km_id / spatial_block_id / tile_id

New additions (to match your Step7 input naming):
- Keep baseline outputs unchanged:
    lead1_{F0,F1,F2}.parquet
    nowcast_{F0,F1,F2}.parquet
- Optionally ALSO write extra lead1 outputs for Step7:
    lead1_{F0,F1,F2}_wgrid50_100.parquet

Hard rules:
A) QC-only columns MUST NOT be in predictors (fail by default):
   pi, weight, d_weight*, valid_*, pix_count*
B) CV-only block columns MUST NOT be predictors (fail by default):
   grid10km_id, grid50km_id, grid100km_id, spatial_block_id, tile_id
C) Leakage blacklist enforcement on predictors.
D) --drop_constants drops constant predictors from predictor list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import yaml


# -------------------------
# CV-only / QC-only rules
# -------------------------
CV_ONLY_BLOCK_COLS = {"grid10km_id", "grid50km_id", "grid100km_id", "spatial_block_id", "tile_id"}


# -------------------------
# utils
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

def read_text_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def atomic_write_parquet(df: pd.DataFrame, out_path: str, *, compression: str = "snappy") -> None:
    out_path = str(out_path)
    out_dir = os.path.dirname(out_path)
    ensure_dir(out_dir)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_step55_", suffix=".parquet", dir=out_dir if out_dir else None)
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

def dedup_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def parse_csv_cols(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    return dedup_preserve_order(parts)


# -------------------------
# read any table
# -------------------------
def _read_table_any(path: str) -> pd.DataFrame:
    if not path:
        raise ValueError("empty path")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".feather"]:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table format: {path} (use parquet/csv/feather)")


# -------------------------
# merge cell-level meta (lon/lat/BIOME/STRATUM...)
# -------------------------
def merge_cell_meta(panel: pd.DataFrame, meta_path: str, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge specified columns from a cell-level meta table by cell_id.

    meta must have: cell_id + requested cols (subset allowed; missing cols will be reported).
    If panel already has a column, we will NOT overwrite it.
    """
    m = _read_table_any(meta_path)
    if "cell_id" not in m.columns:
        raise KeyError(f"cell_meta missing required column: cell_id | path={meta_path}")

    cols = dedup_preserve_order([c.strip() for c in cols if c.strip()])
    cols_avail = [c for c in cols if c in m.columns]
    cols_missing = [c for c in cols if c not in m.columns]

    use_cols = ["cell_id"] + cols_avail
    m2 = m.loc[:, use_cols].drop_duplicates(subset=["cell_id"]).copy()

    before_has = {c: int(c in panel.columns) for c in cols}

    merge_cols = ["cell_id"] + [c for c in cols_avail if c not in panel.columns]
    out = panel
    if len(merge_cols) > 1:
        out = panel.merge(m2.loc[:, merge_cols], on="cell_id", how="left")

    meta_info = {
        "meta_path": meta_path,
        "requested_cols": cols,
        "meta_cols_available": cols_avail,
        "meta_cols_missing_in_meta": cols_missing,
        "merged_cols": [c for c in cols_avail if c not in panel.columns],
        "before_has": before_has,
        "after_has": {c: int(c in out.columns) for c in cols},
        "meta_rows": int(len(m2)),
        "panel_rows": int(len(panel)),
    }
    return out, meta_info


# -------------------------
# derive grid10km from lon/lat
# -------------------------
def derive_grid10km_id_from_lonlat(panel: pd.DataFrame, lon_col: str, lat_col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Derive grid10km_id from lon/lat using EPSG:3857 Web Mercator, then 10km floor.
    grid10km_id := "{x10}_{y10}" where x10=floor(x/10000), y10=floor(y/10000)
    x,y in meters.
    """
    if lon_col not in panel.columns or lat_col not in panel.columns:
        raise KeyError(f"derive_grid10km_from_lonlat requires columns: {lon_col}, {lat_col}")

    R = 6378137.0
    lon = pd.to_numeric(panel[lon_col], errors="coerce")
    lat = pd.to_numeric(panel[lat_col], errors="coerce")

    bad = int(lon.isna().sum() + lat.isna().sum())
    if bad > 0:
        raise RuntimeError(f"lon/lat contains NaN after coercion: bad_count={bad}")

    lat_clamped = lat.clip(lower=-85.05112878, upper=85.05112878)

    lon_rad = np.deg2rad(lon.to_numpy(dtype=np.float64))
    lat_rad = np.deg2rad(lat_clamped.to_numpy(dtype=np.float64))

    x = R * lon_rad
    y = R * np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))

    x10 = np.floor(x / 10000.0).astype(np.int64)
    y10 = np.floor(y / 10000.0).astype(np.int64)

    # IMPORTANT: use pandas string concat (avoid numpy ufunc add type errors)
    sx = pd.Series(x10, index=panel.index, dtype="int64").astype("string")
    sy = pd.Series(y10, index=panel.index, dtype="int64").astype("string")
    grid10km_id = (sx + "_" + sy).astype("string")

    out = panel.copy()
    out["grid10km_id"] = grid10km_id

    meta = {
        "lon_col": lon_col,
        "lat_col": lat_col,
        "derived": True,
        "n_rows": int(len(out)),
        "grid10km_id_unique": int(out["grid10km_id"].nunique(dropna=True)),
    }
    return out, meta


# -------------------------
# derive grid50km/grid100km from grid10km_id
# -------------------------
def derive_grid50_100_from_grid10(panel: pd.DataFrame, grid10_col: str = "grid10km_id") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Derive grid50km_id and grid100km_id from grid10km_id formatted as "x10_y10" (x10,y10 are int grid indices).
    grid50km: x50=floor(x10/5), y50=floor(y10/5)
    grid100km: x100=floor(x10/10), y100=floor(y10/10)
    """
    if grid10_col not in panel.columns:
        raise KeyError(f"derive_grid50_100_from_grid10 requires column: {grid10_col}")

    s = panel[grid10_col].astype("string")
    parts = s.str.split("_", n=1, expand=True)
    if parts.shape[1] != 2:
        bad = s.dropna().head(10).tolist()
        raise RuntimeError(f"{grid10_col} is not in 'x_y' format. examples={bad}")

    x10 = pd.to_numeric(parts[0], errors="coerce")
    y10 = pd.to_numeric(parts[1], errors="coerce")
    if x10.isna().any() or y10.isna().any():
        bad = panel.loc[x10.isna() | y10.isna(), [grid10_col]].head(10).to_dict(orient="records")
        raise RuntimeError(f"{grid10_col} parse to numeric failed. examples={bad}")

    x50 = (x10 // 5).astype("int64")
    y50 = (y10 // 5).astype("int64")
    x100 = (x10 // 10).astype("int64")
    y100 = (y10 // 10).astype("int64")

    grid50km_id = (x50.astype("string") + "_" + y50.astype("string")).astype("string")
    grid100km_id = (x100.astype("string") + "_" + y100.astype("string")).astype("string")

    out = panel.copy()
    out["grid50km_id"] = grid50km_id
    out["grid100km_id"] = grid100km_id

    meta = {
        "grid10_col": grid10_col,
        "derived": True,
        "n_rows": int(len(out)),
        "grid10_unique": int(out[grid10_col].nunique(dropna=True)),
        "grid50_unique": int(out["grid50km_id"].nunique(dropna=True)),
        "grid100_unique": int(out["grid100km_id"].nunique(dropna=True)),
    }
    return out, meta


# -------------------------
# leakage blacklist
# -------------------------
def parse_blacklist(path: str) -> Tuple[List[str], List[re.Pattern]]:
    """
    leakage_blacklist.txt format:
      - exact forbidden cols under "## Forbidden exact columns"
      - regex/substrings under "## Forbidden substrings / regex patterns (defensive)"
      Lines are "- item".
    """
    lines = read_text_lines(path)
    exact: List[str] = []
    patterns: List[str] = []

    mode = None
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("##"):
            if "Forbidden exact columns" in s:
                mode = "exact"
            elif "Forbidden substrings / regex patterns" in s:
                mode = "pattern"
            else:
                mode = None
            continue
        if s.startswith("- "):
            item = s[2:].strip()
            if mode == "exact":
                exact.append(item)
            elif mode == "pattern":
                patterns.append(item)

    compiled: List[re.Pattern] = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error:
            compiled.append(re.compile(re.escape(p)))
    return exact, compiled

def check_no_leakage(cols: List[str], blacklist_exact: List[str], blacklist_patterns: List[re.Pattern]) -> List[str]:
    bad: List[str] = []
    sset = set(cols)
    for x in blacklist_exact:
        if x in sset:
            bad.append(x)
    for c in cols:
        for rx in blacklist_patterns:
            if rx.search(str(c)):
                bad.append(c)
                break
    return sorted(set(bad))


# -------------------------
# column roles
# -------------------------
def infer_id_cols(df: pd.DataFrame) -> List[str]:
    must = []
    for c in ["cell_id", "date", "y", "year", "month"]:
        if c in df.columns:
            must.append(c)
    return must

def infer_weight_cols(df: pd.DataFrame) -> List[str]:
    cand = ["pi", "d_weight", "pi_clip", "d_weight_clip", "weight", "valid_core_pass", "valid_core"]
    return [c for c in cand if c in df.columns]

def compile_qc_only_regex(default_regex: str, user_regex: str = "") -> re.Pattern:
    pat = user_regex.strip() or default_regex
    return re.compile(pat)

def compile_cv_only_regex(user_regex: str = "") -> re.Pattern:
    pat = user_regex.strip() or r"^(grid10km_id|grid50km_id|grid100km_id|spatial_block_id|tile_id)$"
    return re.compile(pat)


# -------------------------
# temporal audit
# -------------------------
def temporal_leakage_audit(predictors: List[str], scenario: str) -> Dict[str, object]:
    preds = [str(c) for c in predictors]
    lag0_hits = sorted({c for c in preds if "_lag0" in c})

    same_month_hits: List[str] = []
    if scenario == "lead1":
        for c in preds:
            if "_mon" in c and ("_lag" not in c):
                same_month_hits.append(c)
        same_month_hits = sorted(set(same_month_hits))

    return {
        "lag0_present": bool(lag0_hits),
        "same_month_dynamic_present": bool(same_month_hits) if scenario == "lead1" else False,
        "lag0_hits": lag0_hits[:200],
        "same_month_dynamic_hits": same_month_hits[:200] if scenario == "lead1" else [],
    }

def enforce_temporal_audit(audit: Dict[str, object], scenario: str, fs_name: str, mode: str) -> None:
    if mode == "off":
        return

    lag0 = bool(audit.get("lag0_present", False))
    sm = bool(audit.get("same_month_dynamic_present", False)) if scenario == "lead1" else False

    if not (lag0 or sm):
        return

    msg = (
        f"[temporal_audit {scenario}/{fs_name}] lag0_present={lag0}, "
        f"same_month_dynamic_present={sm}. "
        f"lag0_hits={audit.get('lag0_hits', [])[:20]} "
        f"same_month_hits={audit.get('same_month_dynamic_hits', [])[:20]}"
    )

    if mode == "fail":
        raise RuntimeError(msg)
    else:
        print("⚠️  " + msg)


# -------------------------
# predictor normalization
# -------------------------
def drop_constant_predictors(df: pd.DataFrame, predictor_cols: List[str]) -> Tuple[List[str], List[str]]:
    dropped: List[str] = []
    kept: List[str] = []
    for c in predictor_cols:
        if c not in df.columns:
            kept.append(c)
            continue
        s = df[c]
        if s.isna().all():
            dropped.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            v = _safe_numeric(s)
            if v.nunique(dropna=True) <= 1:
                dropped.append(c)
                continue
        else:
            if s.nunique(dropna=True) <= 1:
                dropped.append(c)
                continue
        kept.append(c)
    return kept, dropped

def normalize_predictors(
    df: pd.DataFrame,
    scenario: str,
    fs_name: str,
    predictors_raw: List[str],
    *,
    qc_only_rx: re.Pattern,
    qc_only_mode: str,    # fail|drop
    cv_only_rx: re.Pattern,
    cv_only_mode: str,    # fail|drop
    id_mode: str,         # fail|drop
    strict: bool,
) -> Tuple[List[str], Dict[str, object]]:
    info: Dict[str, object] = {
        "raw_n": int(len(predictors_raw)),
        "raw_head": predictors_raw[:20],
        "dedup_removed_n": 0,
        "qc_only_hits": [],
        "cv_only_hits": [],
        "id_col_hits": [],
        "missing_predictors": [],
    }

    preds = [str(c) for c in predictors_raw]
    preds_dedup = dedup_preserve_order(preds)
    info["dedup_removed_n"] = int(len(preds) - len(preds_dedup))
    preds = preds_dedup

    id_cols = set(infer_id_cols(df))
    qc_hits = [c for c in preds if qc_only_rx.match(c)]
    cv_hits = [c for c in preds if (cv_only_rx.match(c) or c in CV_ONLY_BLOCK_COLS)]
    id_hits = [c for c in preds if c in id_cols]

    info["qc_only_hits"] = qc_hits
    info["cv_only_hits"] = cv_hits
    info["id_col_hits"] = id_hits

    if qc_hits and qc_only_mode == "fail":
        raise RuntimeError(f"[{scenario}/{fs_name}] QC-only cols forbidden in predictors. hits={qc_hits[:50]}")
    if cv_hits and cv_only_mode == "fail":
        raise RuntimeError(f"[{scenario}/{fs_name}] CV-only block cols forbidden in predictors. hits={cv_hits[:50]}")
    if id_hits and id_mode == "fail":
        raise RuntimeError(f"[{scenario}/{fs_name}] ID/label/time cols must not be in predictors. hits={id_hits[:50]}")

    if qc_hits and qc_only_mode == "drop":
        preds = [c for c in preds if c not in set(qc_hits)]
    if cv_hits and cv_only_mode == "drop":
        preds = [c for c in preds if c not in set(cv_hits)]
    if id_hits and id_mode == "drop":
        preds = [c for c in preds if c not in set(id_hits)]

    miss = [c for c in preds if c not in df.columns]
    info["missing_predictors"] = miss
    if miss and strict:
        raise KeyError(f"[{scenario}/{fs_name}] predictors not found in panel (strict): {miss[:50]}")
    if miss and (not strict):
        preds = [c for c in preds if c in df.columns]

    return preds, info


# -------------------------
# Step6-derived extras for F1
# -------------------------
LEAD1_STEP6_F1_EXTRA = [
    "y_lag1",
    "y_lag12",
    "fire_count_12m",
    "months_since_last_fire",
    "fire_tslf",
    "P_sum_mon_lag1_roll3m_sum",
    "PET_sum_mon_lag1_roll3m_sum",
    "VPD_mean_mon_lag1_roll3m_mean",
    "NDVI_mean_mon_lag1_roll3m_mean",
]

NOWCAST_STEP6_F1_EXTRA = [
    "y_lag1",
    "y_lag12",
    "fire_count_12m",
    "months_since_last_fire",
    "fire_tslf",
    "P_sum_mon_roll3m_sum",
    "PET_sum_mon_roll3m_sum",
    "VPD_mean_mon_roll3m_mean",
    "NDVI_mean_mon_roll3m_mean",
]

def apply_f1_step6_extras(
    predictors: List[str],
    *,
    scenario: str,
    panel_cols: List[str],
    policy: str,   # off|auto|require
) -> Tuple[List[str], Dict[str, List[str]]]:
    if policy not in {"off", "auto", "require"}:
        raise ValueError(f"Invalid f1_step6_policy: {policy}")

    extras_wanted = LEAD1_STEP6_F1_EXTRA if scenario == "lead1" else NOWCAST_STEP6_F1_EXTRA
    cols_set = set(panel_cols)

    extras_present = [c for c in extras_wanted if c in cols_set]
    extras_missing = [c for c in extras_wanted if c not in cols_set]

    if policy == "require" and extras_missing:
        raise RuntimeError(f"[{scenario}/F1] f1_step6_policy=require but extras missing: {extras_missing}")

    if policy == "off":
        return predictors, {
            "extras_wanted": extras_wanted,
            "extras_present_added": [],
            "extras_missing": extras_missing,
        }

    s = set(predictors)
    out = list(predictors)
    added: List[str] = []
    for c in extras_present:
        if c not in s:
            out.append(c)
            s.add(c)
            added.append(c)

    return out, {
        "extras_wanted": extras_wanted,
        "extras_present_added": added,
        "extras_missing": extras_missing,
    }


# -------------------------
# core build
# -------------------------
@dataclass
class OneFSResult:
    run_id: str
    out_key: str                 # unique label for this output (e.g. lead1_F1, lead1_F1_wgrid50_100)
    scenario: str
    featureset: str
    out_path: str
    hash: str
    n_rows: int
    n_cols_dataset: int

    predictors_raw_n: int
    predictors_kept_n: int
    predictor_cols: str

    dropped_constant_predictors_n: int
    dropped_constant_predictors: str

    leakage_bad_cols_n: int
    leakage_bad_cols: str

    qc_only_hits_n: int
    qc_only_hits: str

    cv_only_hits_n: int
    cv_only_hits: str

    id_col_hits_n: int
    id_col_hits: str

    missing_predictors_n: int
    missing_predictors: str
    dedup_removed_n: int

    keep_extra_cols_req: str
    keep_extra_cols_kept: str
    keep_extra_cols_missing: str

    keep_cv_block_cols: int
    cv_block_cols_kept: str
    cv_block_cols_missing_in_input: str

    f1_step6_policy: str
    f1_step6_extras_added: str
    f1_step6_extras_missing: str

    temporal_lag0_present: int
    temporal_same_month_dynamic_present: int
    temporal_lag0_hits: str
    temporal_same_month_dynamic_hits: str


def build_one(
    df: pd.DataFrame,
    scenario: str,
    fs_name: str,
    predictors_raw: List[str],
    out_path: str,
    *,
    out_key: str,
    run_id: str,
    blacklist_exact: List[str],
    blacklist_patterns: List[re.Pattern],
    strict: bool,
    drop_constants: bool,
    qc_only_rx: re.Pattern,
    qc_only_mode: str,
    cv_only_rx: re.Pattern,
    cv_only_mode: str,
    id_mode: str,
    temporal_audit_mode: str,   # off|warn|fail
    keep_extra_cols: List[str],
    keep_cv_block_cols: bool,
    f1_step6_policy: str,       # off|auto|require
) -> Tuple[OneFSResult, Dict[str, object]]:

    required = ["cell_id", "date", "y"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"[{scenario}/{fs_name}] missing required cols in panel: {miss}")

    f1_meta = {"extras_present_added": [], "extras_missing": []}
    predictors_effective = [str(x) for x in predictors_raw]
    if fs_name == "F1":
        predictors_effective, f1_meta = apply_f1_step6_extras(
            predictors_effective,
            scenario=scenario,
            panel_cols=list(df.columns),
            policy=f1_step6_policy,
        )

    predictors, info = normalize_predictors(
        df, scenario, fs_name, predictors_effective,
        qc_only_rx=qc_only_rx,
        qc_only_mode=qc_only_mode,
        cv_only_rx=cv_only_rx,
        cv_only_mode=cv_only_mode,
        id_mode=id_mode,
        strict=strict,
    )

    bad_leak = check_no_leakage(predictors, blacklist_exact, blacklist_patterns)
    if bad_leak and strict:
        raise RuntimeError(f"[{scenario}/{fs_name}] leakage blacklist hit in predictors: {bad_leak[:50]}")

    dropped_const: List[str] = []
    predictors_final = list(predictors)
    if drop_constants:
        predictors_final, dropped_const = drop_constant_predictors(df, predictors_final)

    predictors_final_exist = [c for c in predictors_final if c in df.columns]
    audit = temporal_leakage_audit(predictors_final_exist, scenario=scenario)
    enforce_temporal_audit(audit, scenario=scenario, fs_name=fs_name, mode=temporal_audit_mode)

    id_cols = infer_id_cols(df)
    w_cols = infer_weight_cols(df)

    keep_extra_cols = dedup_preserve_order([str(c) for c in keep_extra_cols])
    extra_kept = [c for c in keep_extra_cols if c in df.columns]
    extra_missing = [c for c in keep_extra_cols if c not in df.columns]

    cv_kept: List[str] = []
    cv_missing_in_input: List[str] = [c for c in sorted(CV_ONLY_BLOCK_COLS) if c not in df.columns]
    if keep_cv_block_cols:
        cv_kept = [c for c in sorted(CV_ONLY_BLOCK_COLS) if c in df.columns]

    keep: List[str] = []
    for c in id_cols + w_cols + predictors_final + extra_kept + cv_kept:
        if c in df.columns and c not in keep:
            keep.append(c)

    out = df.loc[:, keep].copy()

    ensure_dir(os.path.dirname(out_path))
    atomic_write_parquet(out, out_path, compression="snappy")
    h = file_hash(out_path)

    res = OneFSResult(
        run_id=run_id,
        out_key=out_key,
        scenario=scenario,
        featureset=fs_name,
        out_path=out_path,
        hash=h,
        n_rows=int(out.shape[0]),
        n_cols_dataset=int(out.shape[1]),

        predictors_raw_n=int(info.get("raw_n", len(predictors_effective))),
        predictors_kept_n=int(len([c for c in predictors_final if c in df.columns])),
        predictor_cols="|".join([c for c in predictors_final if c in df.columns][:2000]),

        dropped_constant_predictors_n=int(len(dropped_const)),
        dropped_constant_predictors="|".join(dropped_const[:500]),

        leakage_bad_cols_n=int(len(bad_leak)),
        leakage_bad_cols="|".join(bad_leak[:500]),

        qc_only_hits_n=int(len(info.get("qc_only_hits", []))),
        qc_only_hits="|".join(info.get("qc_only_hits", [])[:500]),

        cv_only_hits_n=int(len(info.get("cv_only_hits", []))),
        cv_only_hits="|".join(info.get("cv_only_hits", [])[:200]),

        id_col_hits_n=int(len(info.get("id_col_hits", []))),
        id_col_hits="|".join(info.get("id_col_hits", [])[:50]),

        missing_predictors_n=int(len(info.get("missing_predictors", []))),
        missing_predictors="|".join(info.get("missing_predictors", [])[:500]),
        dedup_removed_n=int(info.get("dedup_removed_n", 0)),

        keep_extra_cols_req="|".join(keep_extra_cols[:200]),
        keep_extra_cols_kept="|".join(extra_kept[:200]),
        keep_extra_cols_missing="|".join(extra_missing[:200]),

        keep_cv_block_cols=int(bool(keep_cv_block_cols)),
        cv_block_cols_kept="|".join(cv_kept[:50]),
        cv_block_cols_missing_in_input="|".join(cv_missing_in_input[:50]),

        f1_step6_policy=(f1_step6_policy if fs_name == "F1" else ""),
        f1_step6_extras_added="|".join(f1_meta.get("extras_present_added", [])[:200]) if fs_name == "F1" else "",
        f1_step6_extras_missing="|".join(f1_meta.get("extras_missing", [])[:200]) if fs_name == "F1" else "",

        temporal_lag0_present=int(bool(audit.get("lag0_present", False))),
        temporal_same_month_dynamic_present=int(bool(audit.get("same_month_dynamic_present", False))) if scenario == "lead1" else 0,
        temporal_lag0_hits="|".join(audit.get("lag0_hits", [])[:200]),
        temporal_same_month_dynamic_hits="|".join(audit.get("same_month_dynamic_hits", [])[:200]) if scenario == "lead1" else "",
    )

    return res, audit


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--lead1_panel", required=True)
    ap.add_argument("--nowcast_panel", default="")

    ap.add_argument("--feature_sets_yaml", required=True)
    ap.add_argument("--blacklist", default="artifacts/leakage_blacklist.txt")

    ap.add_argument("--out_dir", default="data/processed/featuresets")
    ap.add_argument("--qc_out", default="data/interim/qc_featuresets_step55.csv")
    ap.add_argument("--manifest_out", default="artifacts/featuresets_manifest_step55.json")

    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--drop_constants", action="store_true")

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backup_on_overwrite", action="store_true")

    # predictor admission hard rules
    ap.add_argument("--qc_only_mode", choices=["fail", "drop"], default="fail")
    ap.add_argument("--id_mode", choices=["fail", "drop"], default="fail")
    ap.add_argument("--qc_only_regex", default=r"^(pi|weight|d_weight.*|valid_.*|pix_count.*)$")

    ap.add_argument("--cv_only_mode", choices=["fail", "drop"], default="fail")
    ap.add_argument("--cv_only_regex", default=r"^(grid10km_id|grid50km_id|grid100km_id|spatial_block_id|tile_id)$")

    ap.add_argument("--temporal_audit_mode", choices=["off", "warn", "fail"], default="fail")

    ap.add_argument(
        "--keep_extra_cols",
        default="",
        help="Comma-separated columns to keep in dataset for QC/reporting (NEVER predictors). Example: 'BIOME,STRATUM'.",
    )

    # keep CV block cols in dataset
    ap.add_argument("--keep_cv_block_cols", action="store_true")
    ap.add_argument("--no_keep_cv_block_cols", action="store_true")

    # merge cell meta before derivations
    ap.add_argument("--cell_meta", default="", help="Optional cell-level meta table to merge by cell_id (parquet/csv/feather).")
    ap.add_argument("--cell_meta_cols", default="lon,lat,BIOME,STRATUM", help="Columns to merge from cell_meta (comma-separated).")

    # derive blocks
    ap.add_argument("--derive_grid10km_from_lonlat", action="store_true")
    ap.add_argument("--lon_col", default="lon")
    ap.add_argument("--lat_col", default="lat")

    ap.add_argument("--derive_grid50_100_from_grid10", action="store_true")

    ap.add_argument("--f1_step6_policy", choices=["off", "auto", "require"], default="auto")

    # extra lead1 outputs for Step7 input naming
    ap.add_argument(
        "--also_write_lead1_wgrid50_100",
        action="store_true",
        help="Additionally write lead1_{F0,F1,F2}_wgrid50_100.parquet (for Step7 input).",
    )
    ap.add_argument("--wgrid_suffix", default="_wgrid50_100")

    args = ap.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    keep_extra_cols = parse_csv_cols(args.keep_extra_cols)

    keep_cv_block_cols = True
    if args.keep_cv_block_cols:
        keep_cv_block_cols = True
    if args.no_keep_cv_block_cols:
        keep_cv_block_cols = False

    maybe_overwrite_path(args.qc_out, args.overwrite, args.backup_on_overwrite, "qc_out")
    maybe_overwrite_path(args.manifest_out, args.overwrite, args.backup_on_overwrite, "manifest_out")

    base_cfg = load_yaml(args.feature_sets_yaml)
    if not isinstance(base_cfg, dict):
        raise ValueError("feature_sets_yaml must parse to a dict")

    blacklist_exact, blacklist_patterns = ([], [])
    if args.blacklist and os.path.exists(args.blacklist):
        blacklist_exact, blacklist_patterns = parse_blacklist(args.blacklist)

    if not os.path.exists(args.lead1_panel):
        raise FileNotFoundError(args.lead1_panel)
    lead1 = pd.read_parquet(args.lead1_panel)

    now: Optional[pd.DataFrame] = None
    if args.nowcast_panel:
        if not os.path.exists(args.nowcast_panel):
            raise FileNotFoundError(args.nowcast_panel)
        now = pd.read_parquet(args.nowcast_panel)

    # ---- merge cell meta (lon/lat/BIOME/STRATUM...) BEFORE derive ----
    cell_meta_merge_info: Dict[str, Any] = {"enabled": False}
    if args.cell_meta.strip():
        cols = parse_csv_cols(args.cell_meta_cols)
        lead1, m1 = merge_cell_meta(lead1, args.cell_meta.strip(), cols)
        cell_meta_merge_info = {"enabled": True, "lead1": m1}
        if now is not None:
            now, m2 = merge_cell_meta(now, args.cell_meta.strip(), cols)
            cell_meta_merge_info["nowcast"] = m2

    # ---- derive grid10km if asked ----
    derive10_meta: Dict[str, Any] = {"enabled": False}
    if args.derive_grid10km_from_lonlat:
        if "grid10km_id" not in lead1.columns:
            lead1, dmeta1 = derive_grid10km_id_from_lonlat(lead1, args.lon_col, args.lat_col)
            derive10_meta = {"enabled": True, "lead1": dmeta1}
        if now is not None and ("grid10km_id" not in now.columns):
            now, dmeta2 = derive_grid10km_id_from_lonlat(now, args.lon_col, args.lat_col)
            derive10_meta.setdefault("enabled", True)
            derive10_meta["nowcast"] = dmeta2

    # ---- derive grid50/100 from grid10 if asked ----
    derive50_100_meta: Dict[str, Any] = {"enabled": False}
    if args.derive_grid50_100_from_grid10:
        if "grid10km_id" not in lead1.columns:
            raise RuntimeError("derive_grid50_100_from_grid10 requested but grid10km_id not present in lead1 (merge/derive grid10km first).")
        lead1, gmeta1 = derive_grid50_100_from_grid10(lead1, "grid10km_id")
        derive50_100_meta = {"enabled": True, "lead1": gmeta1}
        if now is not None:
            if "grid10km_id" not in now.columns:
                raise RuntimeError("derive_grid50_100_from_grid10 requested but grid10km_id not present in nowcast (merge/derive grid10km first).")
            now, gmeta2 = derive_grid50_100_from_grid10(now, "grid10km_id")
            derive50_100_meta["nowcast"] = gmeta2

    qc_only_rx = compile_qc_only_regex(
        default_regex=r"^(pi|weight|d_weight.*|valid_.*|pix_count.*)$",
        user_regex=args.qc_only_regex
    )
    cv_only_rx = compile_cv_only_regex(user_regex=args.cv_only_regex)

    results: List[OneFSResult] = []
    audits_by_key: Dict[str, Dict[str, Any]] = {}

    def _validate_spec(spec: dict, scenario_key: str) -> None:
        if not isinstance(spec, dict):
            raise ValueError(f"{scenario_key} must be a dict in YAML")
        for fs_name in ["F0", "F1", "F2"]:
            if fs_name not in spec:
                raise KeyError(f"{scenario_key}.{fs_name} missing in YAML")
            if not isinstance(spec[fs_name], list):
                raise ValueError(f"{scenario_key}.{fs_name} must be a list")

    # ---- lead1 ----
    lead1_spec = base_cfg.get("lead1", {})
    _validate_spec(lead1_spec, "lead1")

    for fs_name in ["F0", "F1", "F2"]:
        preds = lead1_spec.get(fs_name, [])

        # (A) baseline output (unchanged naming)
        out_key = f"lead1_{fs_name}"
        out_path = os.path.join(args.out_dir, f"lead1_{fs_name}.parquet")
        maybe_overwrite_path(out_path, args.overwrite, args.backup_on_overwrite, f"out({out_path})")

        r, audit = build_one(
            lead1, "lead1", fs_name, preds, out_path,
            out_key=out_key,
            run_id=run_id,
            blacklist_exact=blacklist_exact,
            blacklist_patterns=blacklist_patterns,
            strict=args.strict,
            drop_constants=args.drop_constants,
            qc_only_rx=qc_only_rx,
            qc_only_mode=args.qc_only_mode,
            cv_only_rx=cv_only_rx,
            cv_only_mode=args.cv_only_mode,
            id_mode=args.id_mode,
            temporal_audit_mode=args.temporal_audit_mode,
            keep_extra_cols=keep_extra_cols,
            keep_cv_block_cols=keep_cv_block_cols,
            f1_step6_policy=args.f1_step6_policy,
        )
        results.append(r)
        audits_by_key[out_key] = audit

        # (B) extra output for Step7 input naming
        if args.also_write_lead1_wgrid50_100:
            out_key2 = f"lead1_{fs_name}{args.wgrid_suffix}"
            out_path2 = os.path.join(args.out_dir, f"lead1_{fs_name}{args.wgrid_suffix}.parquet")
            maybe_overwrite_path(out_path2, args.overwrite, args.backup_on_overwrite, f"out({out_path2})")

            r2, audit2 = build_one(
                lead1, "lead1", fs_name, preds, out_path2,
                out_key=out_key2,
                run_id=run_id,
                blacklist_exact=blacklist_exact,
                blacklist_patterns=blacklist_patterns,
                strict=args.strict,
                drop_constants=args.drop_constants,
                qc_only_rx=qc_only_rx,
                qc_only_mode=args.qc_only_mode,
                cv_only_rx=cv_only_rx,
                cv_only_mode=args.cv_only_mode,
                id_mode=args.id_mode,
                temporal_audit_mode=args.temporal_audit_mode,
                keep_extra_cols=keep_extra_cols,
                keep_cv_block_cols=keep_cv_block_cols,
                f1_step6_policy=args.f1_step6_policy,
            )
            results.append(r2)
            audits_by_key[out_key2] = audit2

    # ---- nowcast ----
    if now is not None:
        now_spec = base_cfg.get("nowcast_diagnostic", {})
        _validate_spec(now_spec, "nowcast_diagnostic")

        for fs_name in ["F0", "F1", "F2"]:
            preds = now_spec.get(fs_name, [])
            out_key = f"nowcast_{fs_name}"
            out_path = os.path.join(args.out_dir, f"nowcast_{fs_name}.parquet")
            maybe_overwrite_path(out_path, args.overwrite, args.backup_on_overwrite, f"out({out_path})")

            r, audit = build_one(
                now, "nowcast", fs_name, preds, out_path,
                out_key=out_key,
                run_id=run_id,
                blacklist_exact=blacklist_exact,
                blacklist_patterns=blacklist_patterns,
                strict=args.strict,
                drop_constants=args.drop_constants,
                qc_only_rx=qc_only_rx,
                qc_only_mode=args.qc_only_mode,
                cv_only_rx=cv_only_rx,
                cv_only_mode=args.cv_only_mode,
                id_mode=args.id_mode,
                temporal_audit_mode=args.temporal_audit_mode,
                keep_extra_cols=keep_extra_cols,
                keep_cv_block_cols=keep_cv_block_cols,
                f1_step6_policy=args.f1_step6_policy,
            )
            results.append(r)
            audits_by_key[out_key] = audit

    # ---- QC ----
    ensure_dir(os.path.dirname(args.qc_out))
    qc = pd.DataFrame([asdict(r) for r in results])
    qc.to_csv(args.qc_out, index=False)

    def _split_cols(s: str) -> List[str]:
        return [x for x in s.split("|") if x] if s else []

    # ---- manifest ----
    manifest_outputs = []
    for r in results:
        audit = audits_by_key.get(r.out_key, {})
        manifest_outputs.append({
            "out_key": r.out_key,
            "scenario": r.scenario,
            "featureset": r.featureset,
            "path": r.out_path,
            "hash": r.hash,
            "n_rows": r.n_rows,
            "n_cols_dataset": r.n_cols_dataset,
            "n_predictors": r.predictors_kept_n,
            "predictor_cols": _split_cols(r.predictor_cols),
            "dropped_constant_predictors": _split_cols(r.dropped_constant_predictors),
            "leakage_bad_cols": _split_cols(r.leakage_bad_cols),
            "qc_only_hits": _split_cols(r.qc_only_hits),
            "cv_only_hits": _split_cols(r.cv_only_hits),
            "id_col_hits": _split_cols(r.id_col_hits),
            "missing_predictors": _split_cols(r.missing_predictors),
            "dedup_removed_n": r.dedup_removed_n,
            "keep_extra_cols_req": _split_cols(r.keep_extra_cols_req),
            "keep_extra_cols_kept": _split_cols(r.keep_extra_cols_kept),
            "keep_extra_cols_missing": _split_cols(r.keep_extra_cols_missing),
            "keep_cv_block_cols": bool(r.keep_cv_block_cols),
            "cv_block_cols_kept": _split_cols(r.cv_block_cols_kept),
            "cv_block_cols_missing_in_input": _split_cols(r.cv_block_cols_missing_in_input),
            "f1_step6_policy": r.f1_step6_policy,
            "f1_step6_extras_added": _split_cols(r.f1_step6_extras_added),
            "f1_step6_extras_missing": _split_cols(r.f1_step6_extras_missing),
            "temporal_audit": audit,
        })

    script_path = os.path.abspath(__file__)
    manifest = {
        "run_id": run_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "step": "Step5.5",
            "statement": "Column selection + versioning ONLY. No training-only transforms; no anomaly/climatology; no imputation.",
            "qc_only_predictor_rule": {"regex": args.qc_only_regex, "mode": args.qc_only_mode},
            "cv_only_predictor_rule": {"regex": args.cv_only_regex, "mode": args.cv_only_mode, "cols": sorted(list(CV_ONLY_BLOCK_COLS))},
            "id_predictor_rule": {"mode": args.id_mode},
            "keep_extra_cols_rule": {"requested": keep_extra_cols},
            "keep_cv_block_cols_rule": {"enabled": bool(keep_cv_block_cols)},
            "cell_meta_merge": {
                "cell_meta": args.cell_meta.strip(),
                "cell_meta_cols": parse_csv_cols(args.cell_meta_cols),
                "merge_info": cell_meta_merge_info,
            },
            "derive_grid10km_from_lonlat": {
                "enabled": bool(args.derive_grid10km_from_lonlat),
                "lon_col": args.lon_col,
                "lat_col": args.lat_col,
                "meta": derive10_meta,
            },
            "derive_grid50_100_from_grid10": {
                "enabled": bool(args.derive_grid50_100_from_grid10),
                "meta": derive50_100_meta,
            },
            "extra_step7_outputs": {
                "also_write_lead1_wgrid50_100": bool(args.also_write_lead1_wgrid50_100),
                "wgrid_suffix": args.wgrid_suffix,
            },
            "f1_step6_rule": {"policy": args.f1_step6_policy},
            "temporal_audit_rule": {"mode": args.temporal_audit_mode},
        },
        "inputs": {
            "lead1_panel": args.lead1_panel,
            "lead1_panel_hash": file_hash(args.lead1_panel),
            "nowcast_panel": args.nowcast_panel or "",
            "nowcast_panel_hash": file_hash(args.nowcast_panel) if args.nowcast_panel else "",
            "feature_sets_yaml": args.feature_sets_yaml,
            "feature_sets_yaml_hash": file_hash(args.feature_sets_yaml),
            "blacklist": args.blacklist if (args.blacklist and os.path.exists(args.blacklist)) else "",
            "blacklist_hash": file_hash(args.blacklist) if (args.blacklist and os.path.exists(args.blacklist)) else "",
            "script": script_path,
            "script_hash": file_hash(script_path),
            "args": vars(args),
        },
        "outputs": manifest_outputs,
        "notes": {
            "mainline": "Use lead1_* for claims. nowcast_* is diagnostic only.",
            "drop_constants": bool(args.drop_constants),
            "strict": bool(args.strict),
        },
    }

    ensure_dir(os.path.dirname(args.manifest_out))
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("✅ build_featuresets (Step5.5) done.")
    print(f"- run_id: {run_id}")
    print(f"- out_dir: {args.out_dir}")
    print(f"- qc: {args.qc_out}")
    print(f"- manifest: {args.manifest_out}")
    print(f"- keep_cv_block_cols: {keep_cv_block_cols}")
    if args.cell_meta.strip():
        print(f"- cell_meta merged: {args.cell_meta.strip()} cols={parse_csv_cols(args.cell_meta_cols)}")
    if args.derive_grid10km_from_lonlat:
        print(f"- derive_grid10km_from_lonlat: True (lon={args.lon_col}, lat={args.lat_col})")
    if args.derive_grid50_100_from_grid10:
        print(f"- derive_grid50_100_from_grid10: True")
    if args.also_write_lead1_wgrid50_100:
        print(f"- extra lead1 outputs: True (suffix={args.wgrid_suffix})")
    for r in results:
        print(f"  - {r.out_key}: {r.out_path} (hash={r.hash})")


if __name__ == "__main__":
    main()