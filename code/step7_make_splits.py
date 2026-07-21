#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step7 (v3.2): Spatio-temporal CV protocol (P1/P2/P2'/P3/P4) — explicit folds

Design principles (Top-journal friendly):
- Step7 ONLY produces split assignments (train/val/test/inner_train/inner_val). NO transforms.
- Explicitly expands folds (no "val_fold" convention).
- Strict anti-leakage checks for spatial blocks.

Protocols included:
- P1_time_holdout:
    train: [p1_train_start, p1_train_end] excluding val window
    val:   [p1_val_start, p1_val_end]
    test:  [p1_test_start, p1_test_end]
  Default: train=2015–2021, val=2022, test=2023–2024

- P2_spatial_{block}_timeblocked (appendix):
    train/val window: [p2_train_start, p2_train_end]  (default 2015–2022)
    within train/val window: spatial block K-fold -> val by block fold, rest train
    test window: [p2_test_start, p2_test_end] (default 2023–2024) duplicated across folds

- P2_prime_nested_{block}_timeblocked (advanced):
    outer split:
      outer_train: [p2p_outer_train_start, p2p_outer_train_end] (default 2015–2021)
      outer_val:   [p2p_outer_val_start, p2p_outer_val_end]     (default 2022)
      outer_test:  [p2p_outer_test_start, p2p_outer_test_end]   (default 2023–2024)
    nested inner CV for tuning inside outer_train only:
      assign spatial block K-fold on outer_train blocks -> inner_val by fold, rest inner_train
    Output is expanded by inner folds. split_role ∈ {inner_train, inner_val, val, test}

- P3_spatiotemporal_holdout_{block} (strict generalization):
    For each fold:
      heldout_blocks = a subset of blocks (approximately 1/k of all blocks; deterministic via seed)
      test: date in [p3_test_start,p3_test_end] AND block in heldout_blocks
      train: date in [p3_train_start,p3_train_end] AND block NOT in heldout_blocks AND NOT in val window
      val:   date in [p3_val_start,p3_val_end] AND block NOT in heldout_blocks
    Default: train=2015–2021, val=2022, test=2023–2024

- P4_random_row_kfold / P4_random_cell_kfold:
    Random splits (optimistic baseline) to show "virtual high" performance without spatiotemporal blocking.

Outputs:
- split parquet columns:
  protocol, fold_id, split_role, cell_id, date, y, [BIOME], [block_col]
- qc csv: overall + by_BIOME counts, y_rate, date_min/max
- manifest json: params + hashes + realized_windows from output table

Notes:
- We require date to be month-start (day==1).
- We require y binary {0,1}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Prefer coarser blocks first (better for spatial generalization)
BLOCK_PREF = ["grid100km_id", "grid50km_id", "grid10km_id", "spatial_block_id", "tile_id"]


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
    return out

def assert_required(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"[{name}] missing required columns: {miss}")

def safe_int_y(y: pd.Series) -> pd.Series:
    yy = pd.to_numeric(y, errors="coerce")
    if yy.isna().any():
        raise RuntimeError("y has NaN after coercion")
    vals = set(pd.unique(yy.astype(int)))
    if not vals.issubset({0, 1}):
        raise RuntimeError(f"y must be binary {{0,1}}, got values={sorted(list(vals))[:20]}")
    return yy.astype(np.int8)

def parse_ym(s: str) -> pd.Timestamp:
    return pd.Period(s, freq="M").to_timestamp()

def ym_str(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")

def in_range(d: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (d >= start) & (d <= end)

def choose_block_col(df: pd.DataFrame) -> Optional[str]:
    for c in BLOCK_PREF:
        if c in df.columns:
            return c
    return None

def select_meta_cols(df: pd.DataFrame, biome_col: str, block_col: str) -> pd.DataFrame:
    cols = ["cell_id", "date"]
    if biome_col and biome_col in df.columns:
        cols.append(biome_col)
    if block_col and block_col in df.columns:
        cols.append(block_col)
    return df[cols].drop_duplicates(["cell_id", "date"])

def attach_meta(base: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Robust join on (cell_id,date) without creating *_x/*_y or overwriting existing columns.
    If base already has a column, we do NOT re-merge it from meta.
    """
    if meta is None or len(meta) == 0:
        return base
    join_keys = ["cell_id", "date"]
    extra_cols = [c for c in meta.columns if c not in join_keys]
    extra_cols = [c for c in extra_cols if c not in base.columns]
    if len(extra_cols) == 0:
        return base
    m = meta[join_keys + extra_cols]
    return base.merge(m, on=join_keys, how="left")

def fmt_date_minmax(g: pd.DataFrame) -> Dict[str, str]:
    dmin = pd.to_datetime(g["date"]).min()
    dmax = pd.to_datetime(g["date"]).max()
    return {"date_min": ym_str(dmin), "date_max": ym_str(dmax)}

def realized_windows(split_df: pd.DataFrame, protocol: str, fold_id: int) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    g = split_df[(split_df["protocol"] == protocol) & (split_df["fold_id"] == fold_id)]
    for role, gg in g.groupby("split_role"):
        out[str(role)] = fmt_date_minmax(gg)
    return out


# -------------------------
# protocol specs
# -------------------------

@dataclass
class FoldSpec:
    protocol: str
    fold_id: int
    notes: str = ""


# -------------------------
# P1: time-blocked holdout
# -------------------------

def make_p1_holdout(
    df_core: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, List[FoldSpec]]:
    d = df_core["date"]
    m_test = in_range(d, test_start, test_end)
    m_val = in_range(d, val_start, val_end)
    m_train = in_range(d, train_start, train_end) & (~m_val)

    keep = m_test | m_val | m_train
    x = df_core.loc[keep, ["cell_id", "date", "y"]].copy()
    x["protocol"] = "P1_time_holdout"
    x["fold_id"] = 0
    x["split_role"] = np.where(m_test[keep], "test", np.where(m_val[keep], "val", "train"))

    folds = [FoldSpec(protocol="P1_time_holdout", fold_id=0, notes="Mainline: strict time-blocked holdout.")]
    return x.reset_index(drop=True), folds


# -------------------------
# P2: spatial-block K-fold within train/val window + fixed time test
# -------------------------

def make_p2_spatial_blocked(
    df_core: pd.DataFrame,
    *,
    block_col: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    k_blocks: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[FoldSpec]]:
    if block_col not in df_core.columns:
        raise KeyError(f"P2 requires spatial block col '{block_col}' in dataset.")

    d = df_core["date"]
    m_test = in_range(d, test_start, test_end)
    m_trainwin = in_range(d, train_start, train_end)

    blocks = pd.Series(df_core.loc[m_trainwin, block_col].dropna().unique()).astype(str).values
    if len(blocks) == 0:
        raise RuntimeError(f"P2: block_col={block_col} has no valid blocks in train window.")

    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    b2f = {b: int(i % k_blocks) for i, b in enumerate(blocks)}

    tmp = df_core[["cell_id", "date", "y", block_col]].copy()
    tmp["_block_str"] = tmp[block_col].astype(str)
    tmp["_block_fold"] = tmp["_block_str"].map(b2f)

    trainwin_rows = tmp.loc[m_trainwin, ["cell_id", "date", "y", block_col, "_block_fold"]].copy()
    if trainwin_rows["_block_fold"].isna().any():
        bad = trainwin_rows.loc[trainwin_rows["_block_fold"].isna()].head(10).to_dict(orient="records")
        raise RuntimeError(f"P2: rows in train window missing block->fold mapping. examples={bad}")

    parts = []
    folds = []

    proto = f"P2_spatial_{block_col}_timeblocked"

    # fixed test duplicated across folds
    test_rows = tmp.loc[m_test, ["cell_id", "date", "y", block_col]].copy()
    for f in range(k_blocks):
        z = test_rows.copy()
        z["protocol"] = proto
        z["fold_id"] = f
        z["split_role"] = "test"
        parts.append(z)

    for f in range(k_blocks):
        z = trainwin_rows.copy()
        z["protocol"] = proto
        z["fold_id"] = f
        z["split_role"] = np.where(z["_block_fold"].astype(int) == f, "val", "train")
        z = z.drop(columns=["_block_fold"])
        parts.append(z)
        folds.append(FoldSpec(protocol=proto, fold_id=f, notes="Appendix: spatial-block CV in train window + fixed time test."))

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out.reset_index(drop=True), folds


# -------------------------
# P2': outer time split + inner spatial nested CV inside outer-train
# Output split_role: inner_train, inner_val, val, test
# -------------------------

def make_p2_prime_nested(
    df_core: pd.DataFrame,
    *,
    block_col: str,
    outer_train_start: pd.Timestamp,
    outer_train_end: pd.Timestamp,
    outer_val_start: pd.Timestamp,
    outer_val_end: pd.Timestamp,
    outer_test_start: pd.Timestamp,
    outer_test_end: pd.Timestamp,
    k_inner: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[FoldSpec]]:
    if block_col not in df_core.columns:
        raise KeyError(f"P2' requires spatial block col '{block_col}' in dataset.")

    d = df_core["date"]
    m_test = in_range(d, outer_test_start, outer_test_end)
    m_val = in_range(d, outer_val_start, outer_val_end)
    m_outer_train = in_range(d, outer_train_start, outer_train_end)

    proto = f"P2_prime_nested_{block_col}_timeblocked"

    # blocks from outer-train only
    blocks = pd.Series(df_core.loc[m_outer_train, block_col].dropna().unique()).astype(str).values
    if len(blocks) == 0:
        raise RuntimeError(f"P2': block_col={block_col} has no valid blocks in outer-train window.")

    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    b2f = {b: int(i % k_inner) for i, b in enumerate(blocks)}

    tmp = df_core[["cell_id", "date", "y", block_col]].copy()
    tmp["_block_str"] = tmp[block_col].astype(str)
    tmp["_inner_fold"] = tmp["_block_str"].map(b2f)

    outer_train_rows = tmp.loc[m_outer_train, ["cell_id", "date", "y", block_col, "_inner_fold"]].copy()
    if outer_train_rows["_inner_fold"].isna().any():
        bad = outer_train_rows.loc[outer_train_rows["_inner_fold"].isna()].head(10).to_dict(orient="records")
        raise RuntimeError(f"P2': rows in outer-train missing block->inner_fold mapping. examples={bad}")

    val_rows = tmp.loc[m_val, ["cell_id", "date", "y", block_col]].copy()
    test_rows = tmp.loc[m_test, ["cell_id", "date", "y", block_col]].copy()

    parts = []
    folds = []

    # Expand by inner folds so Step8 can do tuning inside each fold_id
    # split_role meanings per fold_id:
    #   inner_train/inner_val: used for hyperparam search
    #   val: outer temporal validation (2022)
    #   test: final temporal test (2023–2024)
    for f in range(k_inner):
        z_tr = outer_train_rows.copy()
        z_tr["protocol"] = proto
        z_tr["fold_id"] = f
        z_tr["split_role"] = np.where(z_tr["_inner_fold"].astype(int) == f, "inner_val", "inner_train")
        z_tr = z_tr.drop(columns=["_inner_fold"])
        parts.append(z_tr)

        z_val = val_rows.copy()
        z_val["protocol"] = proto
        z_val["fold_id"] = f
        z_val["split_role"] = "val"
        parts.append(z_val)

        z_te = test_rows.copy()
        z_te["protocol"] = proto
        z_te["fold_id"] = f
        z_te["split_role"] = "test"
        parts.append(z_te)

        folds.append(FoldSpec(protocol=proto, fold_id=f, notes="Advanced: outer time split + nested spatial CV inside outer-train for tuning."))

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out.reset_index(drop=True), folds


# -------------------------
# P3: spatiotemporal holdout
# test: future years & heldout blocks
# train/val: other blocks only, date<=val_end (default 2022)
# -------------------------

def make_p3_spatiotemporal_holdout(
    df_core: pd.DataFrame,
    *,
    block_col: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    k_blocks: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[FoldSpec]]:
    if block_col not in df_core.columns:
        raise KeyError(f"P3 requires spatial block col '{block_col}' in dataset.")

    proto = f"P3_spatiotemporal_holdout_{block_col}"

    blocks_all = pd.Series(df_core[block_col].dropna().unique()).astype(str).values
    if len(blocks_all) == 0:
        raise RuntimeError(f"P3: block_col={block_col} has no blocks at all.")

    rng = np.random.default_rng(seed)
    rng.shuffle(blocks_all)
    b2f = {b: int(i % k_blocks) for i, b in enumerate(blocks_all)}

    d = df_core["date"]
    m_test_time = in_range(d, test_start, test_end)
    m_val_time = in_range(d, val_start, val_end)
    m_train_time = in_range(d, train_start, train_end) & (~m_val_time)

    tmp = df_core[["cell_id", "date", "y", block_col]].copy()
    tmp["_block_str"] = tmp[block_col].astype(str)
    tmp["_block_fold"] = tmp["_block_str"].map(b2f)

    if tmp["_block_fold"].isna().any():
        bad = tmp.loc[tmp["_block_fold"].isna()].head(10).to_dict(orient="records")
        raise RuntimeError(f"P3: block->fold mapping has NaN. examples={bad}")

    parts = []
    folds = []

    for f in range(k_blocks):
        heldout = (tmp["_block_fold"].astype(int) == f)

        # Spec:
        # test: future & heldout blocks
        m_test = m_test_time & heldout

        # train/val: non-heldout blocks ONLY, date<=val_end (here enforced by m_train_time/m_val_time)
        m_train = m_train_time & (~heldout)
        m_val = m_val_time & (~heldout)

        keep = m_test | m_train | m_val
        z = tmp.loc[keep, ["cell_id", "date", "y", block_col]].copy()
        z["protocol"] = proto
        z["fold_id"] = f
        z["split_role"] = np.where(m_test[keep], "test", np.where(m_val[keep], "val", "train"))
        parts.append(z)

        folds.append(FoldSpec(protocol=proto, fold_id=f, notes="Strict: test requires (future years) AND (heldout blocks). Train/val use other blocks only."))

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out.reset_index(drop=True), folds


# -------------------------
# P4: random kfold (row / cell)
# -------------------------

def make_random_row_kfold_expanded(df_core: pd.DataFrame, *, k: int, seed: int) -> Tuple[pd.DataFrame, List[FoldSpec]]:
    n = len(df_core)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=np.int32)
    fold_id[perm] = (np.arange(n) % k).astype(np.int32)

    base = df_core[["cell_id", "date", "y"]].copy()
    base["_fold"] = fold_id

    parts = []
    folds = []
    for f in range(k):
        z = base.copy()
        z["protocol"] = "P4_random_row_kfold"
        z["fold_id"] = f
        z["split_role"] = np.where(z["_fold"] == f, "val", "train")
        z = z.drop(columns=["_fold"])
        parts.append(z)
        folds.append(FoldSpec(protocol="P4_random_row_kfold", fold_id=f, notes="Optimistic baseline: random ROW-wise K-fold."))
    return pd.concat(parts, axis=0, ignore_index=True), folds

def make_random_cell_kfold_expanded(df_core: pd.DataFrame, *, k: int, seed: int) -> Tuple[pd.DataFrame, List[FoldSpec]]:
    cells = df_core["cell_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    cell2fold = {cid: int(i % k) for i, cid in enumerate(cells)}

    base = df_core[["cell_id", "date", "y"]].copy()
    base["_fold"] = base["cell_id"].map(cell2fold).astype(np.int32)

    parts = []
    folds = []
    for f in range(k):
        z = base.copy()
        z["protocol"] = "P4_random_cell_kfold"
        z["fold_id"] = f
        z["split_role"] = np.where(z["_fold"] == f, "val", "train")
        z = z.drop(columns=["_fold"])
        parts.append(z)
        folds.append(FoldSpec(protocol="P4_random_cell_kfold", fold_id=f, notes="Optimistic baseline: random CELL-wise K-fold (no time blocking)."))
    return pd.concat(parts, axis=0, ignore_index=True), folds


# -------------------------
# QC + checks
# -------------------------

def qc_basic(df_split: pd.DataFrame, biome_col: str) -> pd.DataFrame:
    rows = []

    def _add(g: pd.DataFrame, tag: Dict[str, str]) -> None:
        y_rate = float(g["y"].mean()) if len(g) else float("nan")
        rec = {
            **tag,
            "n_rows": int(len(g)),
            "n_cells": int(g["cell_id"].nunique()),
            "y_rate": y_rate,
            "date_min": str(pd.to_datetime(g["date"]).min().date()),
            "date_max": str(pd.to_datetime(g["date"]).max().date()),
        }
        rows.append(rec)

    grp_cols = ["protocol", "fold_id", "split_role"]
    for keys, g in df_split.groupby(grp_cols, dropna=False):
        tag = dict(zip(grp_cols, [str(x) for x in keys]))
        _add(g, tag)

    out = pd.DataFrame(rows)
    out["qc_level"] = "overall"

    if biome_col and biome_col in df_split.columns:
        rows2 = []
        grp_cols2 = ["protocol", "fold_id", "split_role", biome_col]
        for keys, g in df_split.groupby(grp_cols2, dropna=False):
            tag = dict(zip(grp_cols2, [str(x) for x in keys]))
            y_rate = float(g["y"].mean()) if len(g) else float("nan")
            rows2.append({
                **tag,
                "n_rows": int(len(g)),
                "n_cells": int(g["cell_id"].nunique()),
                "y_rate": y_rate,
                "date_min": str(pd.to_datetime(g["date"]).min().date()),
                "date_max": str(pd.to_datetime(g["date"]).max().date()),
            })
        out2 = pd.DataFrame(rows2)
        out2["qc_level"] = f"by_{biome_col}"
        out = pd.concat([out, out2], axis=0, ignore_index=True)

    return out

def assert_no_overlap_within_protocol_fold(df_split: pd.DataFrame) -> None:
    key_cols = ["protocol", "fold_id", "cell_id", "date"]
    tmp = df_split[key_cols + ["split_role"]].copy()
    c = tmp.groupby(key_cols, dropna=False)["split_role"].nunique()
    bad = c[c > 1]
    if len(bad) > 0:
        ex = bad.head(10)
        raise RuntimeError(f"Split overlap detected within (protocol,fold): examples={ex.to_dict()}")

def assert_blocks_disjoint(split_df: pd.DataFrame, protocol: str, block_col: str, role_a: str, role_b: str) -> None:
    """
    For each fold: blocks(role_a) ∩ blocks(role_b) must be empty.
    """
    g_all = split_df[split_df["protocol"] == protocol].copy()
    if len(g_all) == 0:
        return
    if block_col not in g_all.columns:
        raise RuntimeError(f"Block disjoint check failed: missing block_col={block_col} in split_df for protocol={protocol}.")

    for f in sorted(g_all["fold_id"].unique()):
        g = g_all[g_all["fold_id"] == f]
        a = set(g[g["split_role"] == role_a][block_col].dropna().astype(str).unique())
        b = set(g[g["split_role"] == role_b][block_col].dropna().astype(str).unique())
        inter = a & b
        if len(inter) > 0:
            sample = list(sorted(inter))[:5]
            raise RuntimeError(
                f"Block leakage: protocol={protocol} fold={f} roles({role_a},{role_b}) share blocks "
                f"(n_overlap={len(inter)}). examples={sample}"
            )

def assert_p3_heldout_logic(split_df: pd.DataFrame, protocol: str, block_col: str,
                           test_start: pd.Timestamp, test_end: pd.Timestamp) -> None:
    """
    P3 required logic sanity:
    - test rows must be within [test_start,test_end]
    - for each fold, no block appears in both test and train/val
    """
    g_all = split_df[split_df["protocol"] == protocol].copy()
    if len(g_all) == 0:
        return
    for f in sorted(g_all["fold_id"].unique()):
        g = g_all[g_all["fold_id"] == f]
        gt = g[g["split_role"] == "test"]
        if len(gt) > 0:
            dmin = pd.to_datetime(gt["date"]).min()
            dmax = pd.to_datetime(gt["date"]).max()
            if dmin < test_start or dmax > test_end:
                raise RuntimeError(f"P3 test time range violated: fold={f} got [{dmin},{dmax}] expected within [{test_start},{test_end}]")

        test_blocks = set(gt[block_col].dropna().astype(str).unique())
        tv_blocks = set(g[g["split_role"].isin(["train", "val"])][block_col].dropna().astype(str).unique())
        inter = test_blocks & tv_blocks
        if len(inter) > 0:
            raise RuntimeError(f"P3 leakage: fold={f} test blocks overlap train/val blocks. n={len(inter)}")

def summarize_protocols(df_split: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for p, g in df_split.groupby("protocol"):
        out[p] = {
            "n_folds": int(g["fold_id"].nunique()),
            "n_rows": int(len(g)),
            "n_cells": int(g["cell_id"].nunique()),
        }
    return out


# -------------------------
# main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--featureset_in", required=True)
    ap.add_argument("--out_split_parquet", required=True)
    ap.add_argument("--out_manifest_json", required=True)
    ap.add_argument("--out_qc_csv", required=True)

    ap.add_argument("--biome_col", default="BIOME")

    # choose / force spatial block
    ap.add_argument("--block_col", default="", help="Spatial block col for P2/P2'/P3. If empty, auto choose by preference.")

    # enable protocols
    ap.add_argument("--enable_p1", action="store_true")
    ap.add_argument("--enable_p2", action="store_true")
    ap.add_argument("--enable_p2_prime", action="store_true")
    ap.add_argument("--enable_p3", action="store_true")
    ap.add_argument("--enable_p4", action="store_true")

    # P1 windows (defaults per your new spec)
    ap.add_argument("--p1_train_start", default="2015-01")
    ap.add_argument("--p1_train_end",   default="2021-12")
    ap.add_argument("--p1_val_start",   default="2022-01")
    ap.add_argument("--p1_val_end",     default="2022-12")
    ap.add_argument("--p1_test_start",  default="2023-01")
    ap.add_argument("--p1_test_end",    default="2024-12")

    # P2 windows (train window includes 2022, spatial CV inside it)
    ap.add_argument("--p2_train_start", default="2015-01")
    ap.add_argument("--p2_train_end",   default="2022-12")
    ap.add_argument("--p2_test_start",  default="2023-01")
    ap.add_argument("--p2_test_end",    default="2024-12")
    ap.add_argument("--p2_k_blocks", type=int, default=5)
    ap.add_argument("--p2_seed", type=int, default=0)

    # P2' outer time windows + inner K
    ap.add_argument("--p2p_outer_train_start", default="2015-01")
    ap.add_argument("--p2p_outer_train_end",   default="2021-12")
    ap.add_argument("--p2p_outer_val_start",   default="2022-01")
    ap.add_argument("--p2p_outer_val_end",     default="2022-12")
    ap.add_argument("--p2p_outer_test_start",  default="2023-01")
    ap.add_argument("--p2p_outer_test_end",    default="2024-12")
    ap.add_argument("--p2p_k_inner", type=int, default=5)
    ap.add_argument("--p2p_seed", type=int, default=0)

    # P3 windows + K blocks
    ap.add_argument("--p3_train_start", default="2015-01")
    ap.add_argument("--p3_train_end",   default="2021-12")
    ap.add_argument("--p3_val_start",   default="2022-01")
    ap.add_argument("--p3_val_end",     default="2022-12")
    ap.add_argument("--p3_test_start",  default="2023-01")
    ap.add_argument("--p3_test_end",    default="2024-12")
    ap.add_argument("--p3_k_blocks", type=int, default=5)
    ap.add_argument("--p3_seed", type=int, default=0)

    # P4 random
    ap.add_argument("--p4_k", type=int, default=5)
    ap.add_argument("--p4_seed", type=int, default=0)

    ap.add_argument("--hash_algo", choices=["sha256", "md5"], default="sha256")

    args = ap.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not os.path.exists(args.featureset_in):
        raise FileNotFoundError(args.featureset_in)

    df = pd.read_parquet(args.featureset_in)
    df = coerce_date(df, "featureset")
    assert_required(df, ["cell_id", "date", "y"], "featureset")
    df["y"] = safe_int_y(df["y"])

    biome_col = (args.biome_col or "").strip()
    if biome_col and biome_col not in df.columns:
        print("⚠️  BIOME column not found. QC will be overall only.")
        biome_col = ""

    # block col (for protocols that need it)
    block_col = (args.block_col or "").strip()
    need_block = args.enable_p2 or args.enable_p2_prime or args.enable_p3
    if need_block:
        if not block_col:
            block_col = choose_block_col(df) or ""
        if not block_col or block_col not in df.columns:
            raise RuntimeError(
                "Spatial-block protocol requested but no usable block col found. "
                f"Provided='{args.block_col}', tried preference={BLOCK_PREF}."
            )

    # meta for BIOME (and block if you want kept)
    meta = select_meta_cols(df, biome_col=biome_col, block_col=block_col if need_block else "")

    # df_core
    core_cols = ["cell_id", "date", "y"] + ([block_col] if need_block else [])
    df_core = df[core_cols].drop_duplicates(["cell_id", "date"]).copy()

    all_splits = []
    all_folds = []

    # ---- P1 ----
    if args.enable_p1:
        p1_split, p1_folds = make_p1_holdout(
            df_core[["cell_id", "date", "y"]],
            train_start=parse_ym(args.p1_train_start),
            train_end=parse_ym(args.p1_train_end),
            val_start=parse_ym(args.p1_val_start),
            val_end=parse_ym(args.p1_val_end),
            test_start=parse_ym(args.p1_test_start),
            test_end=parse_ym(args.p1_test_end),
        )
        p1_split = attach_meta(p1_split, meta)
        all_splits.append(p1_split)
        all_folds.extend([asdict(f) for f in p1_folds])

    # ---- P2 ----
    p2_protocol = ""
    if args.enable_p2:
        p2_split, p2_folds = make_p2_spatial_blocked(
            df_core[["cell_id", "date", "y", block_col]],
            block_col=block_col,
            train_start=parse_ym(args.p2_train_start),
            train_end=parse_ym(args.p2_train_end),
            test_start=parse_ym(args.p2_test_start),
            test_end=parse_ym(args.p2_test_end),
            k_blocks=int(args.p2_k_blocks),
            seed=int(args.p2_seed),
        )
        p2_split = attach_meta(p2_split, meta)  # will skip block_col if already present
        all_splits.append(p2_split)
        all_folds.extend([asdict(f) for f in p2_folds])
        p2_protocol = f"P2_spatial_{block_col}_timeblocked"

    # ---- P2' ----
    p2p_protocol = ""
    if args.enable_p2_prime:
        p2p_split, p2p_folds = make_p2_prime_nested(
            df_core[["cell_id", "date", "y", block_col]],
            block_col=block_col,
            outer_train_start=parse_ym(args.p2p_outer_train_start),
            outer_train_end=parse_ym(args.p2p_outer_train_end),
            outer_val_start=parse_ym(args.p2p_outer_val_start),
            outer_val_end=parse_ym(args.p2p_outer_val_end),
            outer_test_start=parse_ym(args.p2p_outer_test_start),
            outer_test_end=parse_ym(args.p2p_outer_test_end),
            k_inner=int(args.p2p_k_inner),
            seed=int(args.p2p_seed),
        )
        p2p_split = attach_meta(p2p_split, meta)
        all_splits.append(p2p_split)
        all_folds.extend([asdict(f) for f in p2p_folds])
        p2p_protocol = f"P2_prime_nested_{block_col}_timeblocked"

    # ---- P3 ----
    p3_protocol = ""
    if args.enable_p3:
        p3_split, p3_folds = make_p3_spatiotemporal_holdout(
            df_core[["cell_id", "date", "y", block_col]],
            block_col=block_col,
            train_start=parse_ym(args.p3_train_start),
            train_end=parse_ym(args.p3_train_end),
            val_start=parse_ym(args.p3_val_start),
            val_end=parse_ym(args.p3_val_end),
            test_start=parse_ym(args.p3_test_start),
            test_end=parse_ym(args.p3_test_end),
            k_blocks=int(args.p3_k_blocks),
            seed=int(args.p3_seed),
        )
        p3_split = attach_meta(p3_split, meta)
        all_splits.append(p3_split)
        all_folds.extend([asdict(f) for f in p3_folds])
        p3_protocol = f"P3_spatiotemporal_holdout_{block_col}"

    # ---- P4 ----
    if args.enable_p4:
        p4_row, p4_row_folds = make_random_row_kfold_expanded(
            df_core[["cell_id", "date", "y"]], k=int(args.p4_k), seed=int(args.p4_seed)
        )
        p4_cell, p4_cell_folds = make_random_cell_kfold_expanded(
            df_core[["cell_id", "date", "y"]], k=int(args.p4_k), seed=int(args.p4_seed)
        )
        p4_row = attach_meta(p4_row, meta)
        p4_cell = attach_meta(p4_cell, meta)
        all_splits.extend([p4_row, p4_cell])
        all_folds.extend([asdict(f) for f in p4_row_folds])
        all_folds.extend([asdict(f) for f in p4_cell_folds])

    if len(all_splits) == 0:
        raise RuntimeError("No protocol enabled. Use --enable_p1/--enable_p2/--enable_p2_prime/--enable_p3/--enable_p4.")

    split_df = pd.concat(all_splits, axis=0, ignore_index=True)
    assert_required(split_df, ["protocol", "fold_id", "split_role", "cell_id", "date", "y"], "split_df")
    assert_no_overlap_within_protocol_fold(split_df)

    # Strict block disjoint checks
    if args.enable_p2:
        assert_blocks_disjoint(split_df, p2_protocol, block_col, role_a="train", role_b="val")
    if args.enable_p2_prime:
        assert_blocks_disjoint(split_df, p2p_protocol, block_col, role_a="inner_train", role_b="inner_val")
    if args.enable_p3:
        assert_p3_heldout_logic(
            split_df, p3_protocol, block_col,
            test_start=parse_ym(args.p3_test_start), test_end=parse_ym(args.p3_test_end)
        )

    # outputs
    ensure_dir(os.path.dirname(args.out_split_parquet))
    split_df.to_parquet(args.out_split_parquet, index=False, engine="pyarrow", compression="snappy")

    qc = qc_basic(split_df, biome_col=biome_col)
    ensure_dir(os.path.dirname(args.out_qc_csv))
    qc.to_csv(args.out_qc_csv, index=False)

    # realized windows
    realized = {}
    for proto in sorted(split_df["protocol"].unique().tolist()):
        realized[proto] = {
            str(f): realized_windows(split_df, proto, int(f))
            for f in sorted(split_df.loc[split_df["protocol"] == proto, "fold_id"].unique())
        }

    manifest = {
        "run_id": run_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "featureset_in": args.featureset_in,
        "featureset_hash": file_hash(args.featureset_in, algo=args.hash_algo),
        "out_split_parquet": args.out_split_parquet,
        "out_split_hash": file_hash(args.out_split_parquet, algo=args.hash_algo),
        "protocols_included": sorted(split_df["protocol"].unique().tolist()),
        "protocol_summary": summarize_protocols(split_df),
        "biome_col_used": biome_col,
        "block_col_used": block_col if need_block else "",
        "declared_windows": {
            "p1": {"train": [args.p1_train_start, args.p1_train_end],
                   "val": [args.p1_val_start, args.p1_val_end],
                   "test": [args.p1_test_start, args.p1_test_end]},
            "p2": {"trainwin": [args.p2_train_start, args.p2_train_end],
                   "test": [args.p2_test_start, args.p2_test_end]},
            "p2_prime": {"outer_train": [args.p2p_outer_train_start, args.p2p_outer_train_end],
                         "outer_val": [args.p2p_outer_val_start, args.p2p_outer_val_end],
                         "outer_test": [args.p2p_outer_test_start, args.p2p_outer_test_end]},
            "p3": {"train": [args.p3_train_start, args.p3_train_end],
                   "val": [args.p3_val_start, args.p3_val_end],
                   "test": [args.p3_test_start, args.p3_test_end]},
        },
        "realized_windows": realized,
        "p2": {"k_blocks": int(args.p2_k_blocks), "seed": int(args.p2_seed)},
        "p2_prime": {"k_inner": int(args.p2p_k_inner), "seed": int(args.p2p_seed)},
        "p3": {"k_blocks": int(args.p3_k_blocks), "seed": int(args.p3_seed)},
        "p4": {"k": int(args.p4_k), "seed": int(args.p4_seed)},
        "fold_specs": all_folds,
        "notes": {
            "step": "Step7_v3_2",
            "statement": "Splitting ONLY. No fold-wise transforms here (belongs to Step8).",
            "explicit_folds": "Expanded explicitly (no val_fold convention).",
            "top_journal_note": "P1 mainline, P2 appendix spatial CV, P2' nested tuning, P3 strict spatiotemporal holdout, P4 optimistic baselines.",
        },
    }

    ensure_dir(os.path.dirname(args.out_manifest_json))
    with open(args.out_manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("✅ Step7 v3.2 done (explicit folds generated).")
    print(f"- run_id: {run_id}")
    print(f"- split_parquet: {args.out_split_parquet}")
    print(f"- qc_csv: {args.out_qc_csv}")
    print(f"- manifest_json: {args.out_manifest_json}")
    print(f"- protocols: {manifest['protocols_included']}")
    print(f"- block_col_used: {manifest['block_col_used']}")


if __name__ == "__main__":
    main()