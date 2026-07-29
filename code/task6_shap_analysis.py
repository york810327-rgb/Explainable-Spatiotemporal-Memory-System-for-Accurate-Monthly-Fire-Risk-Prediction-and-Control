"""Regenerate the requested comparison, P3, and BIOME-specific SHAP figures.

This script deliberately reuses the fitted P1 model and the already-produced
metric tables. It does not retrain models and it does not download data.

Inputs
------
SCI/data/01_p1_high_metrics.csv
SCI/data/03_official_p1_p3_multiscale_metrics.csv
SCI/results/p1_lgbm_strict_model.joblib
SCI/data/p3_weight_dataset.parquet
SCI/data/full_features.csv

Generated CSV tables are written to SCI/data/. Generated figures are written
to SCI/results/.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 20260729
MODEL_PATH = RESULTS / "p1_lgbm_strict_model.joblib"
DATA_PATH = DATA / "p3_weight_dataset.parquet"
FEATURE_PATH = DATA / "full_features.csv"
P1_METRICS_PATH = DATA / "01_p1_high_metrics.csv"
P3_METRICS_PATH = DATA / "03_official_p1_p3_multiscale_metrics.csv"

MODEL_ORDER = [
    "LightGBM (depth-capped)",
    "Logistic L2 baseline",
    "Logistic L1 baseline",
    "ElasticNet baseline",
]
MODEL_LABELS = {
    "LightGBM (depth-capped)": "LightGBM",
    "Logistic L2 baseline": "Logistic L2",
    "Logistic L1 baseline": "Logistic L1",
    "ElasticNet baseline": "ElasticNet",
}
COLORS = {
    "LightGBM (depth-capped)": "#087E8B",
    "Logistic L2 baseline": "#F4A261",
    "Logistic L1 baseline": "#8D6A9F",
    "ElasticNet baseline": "#C8553D",
}

BIOME_NAMES = {
    1: ("Tropical & Subtropical Moist Broadleaf Forests", "TSMBF"),
    4: ("Temperate Broadleaf & Mixed Forests", "TBMF"),
    5: ("Temperate Conifer Forests", "TCF"),
    6: ("Boreal Forests / Taiga", "BFT"),
    8: ("Temperate Grasslands, Savannas & Shrublands", "TGSS"),
    9: ("Flooded Grasslands & Savannas", "FGS"),
    10: ("Montane Grasslands & Shrublands", "MGS"),
    11: ("Unclassified / N/A", "N/A"),
    13: ("Deserts & Xeric Shrublands", "DXS"),
}

FEATURE_GROUPS = {
    "Fire history": [
        "y_lag1",
        "y_lag12",
        "fire_count_12m",
        "months_since_last_fire",
    ],
    "Drought / moisture": [
        "PET_sum_mon_lag1",
        "P_sum_mon_lag1",
        "VPD_mean_mon_lag1",
        "VPD_mean_mon_lag1_roll3m_mean",
        "SM1_mean_mon_lag1",
        "RH_mean_mon_lag1",
    ],
    "Wind": [
        "WD_R_mon_lag1",
        "WD_u_mon_lag1",
        "WD_v_mon_lag1",
        "WS_max_mon_lag1",
        "WS_mean_mon_lag1",
        "WS_strong_frac_lag1",
    ],
    "Vegetation": [
        "NDVI_mean_mon_lag1",
        "EVI_mean_mon_lag1",
        "treecover_2015",
        "frac_forest",
    ],
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
    }
)


def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(RESULTS / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tree_vs_baselines() -> pd.DataFrame:
    metrics = pd.read_csv(P1_METRICS_PATH)
    metrics["Model"] = pd.Categorical(
        metrics["Model"], categories=MODEL_ORDER, ordered=True
    )
    metrics = metrics.sort_values("Model").dropna(subset=["Model"]).copy()

    specs = [
        ("AUPRC", "higher is better"),
        ("ROC-AUC", "higher is better"),
        ("F1", "higher is better"),
        ("Brier score", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.3))
    for ax, (metric, direction) in zip(axes, specs):
        values = metrics[metric].astype(float).to_numpy()
        names = metrics["Model"].astype(str).tolist()
        bars = ax.bar(
            range(len(metrics)),
            values,
            color=[COLORS[name] for name in names],
            width=0.68,
        )
        pad = max(values.max() * 0.035, 0.002)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + pad,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"{metric}\n{direction}", fontsize=10)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(
            [MODEL_LABELS[name] for name in names], rotation=32, ha="right"
        )
        ax.set_ylim(0, values.max() * 1.18)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Test score")

    fig.suptitle(
        "P1 time extrapolation (train 2015–2021, validation 2022, test 2023–2024)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.04,
        "LightGBM leads on ranking and threshold metrics; its class-weighted probabilities "
        "have a higher (worse) Brier score than the linear baselines.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    _save(fig, "02_p1_tree_vs_baselines.png")
    return metrics


def plot_p3_generalization_gap() -> pd.DataFrame:
    raw = pd.read_csv(P3_METRICS_PATH)
    p1 = raw.loc[raw["Protocol"].astype(str).str.lower().eq("p1")].iloc[0]
    p3 = raw.loc[
        raw["Protocol"].astype(str).str.contains(r"p3_\d+km", regex=True)
    ].copy()
    p3["Scale_km"] = (
        p3["Protocol"].astype(str).str.extract(r"p3_(\d+)km")[0].astype(int)
    )
    p3 = p3.sort_values("Scale_km")

    metric_specs = [
        ("AUPRC", "AUPRC", True),
        ("ROC-AUC", "ROC-AUC", True),
        ("Brier_Score", "Brier score", False),
    ]
    gap_rows = []
    for col, label, higher_better in metric_specs:
        for _, row in p3.iterrows():
            raw_delta = float(row[col] - p1[col])
            gap_rows.append(
                {
                    "Scale_km": int(row["Scale_km"]),
                    "Metric": label,
                    "P1_reference": float(p1[col]),
                    "P3_score": float(row[col]),
                    "P3_minus_P1": raw_delta,
                    "performance_change_vs_P1": (
                        raw_delta if higher_better else -raw_delta
                    ),
                }
            )
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(
        DATA / "03_p3_generalization_gap.csv",
        index=False,
        float_format="%.6f",
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    scales = p3["Scale_km"].to_numpy()
    for ax, (col, label, higher_better) in zip(axes, metric_specs):
        scores = p3[col].astype(float).to_numpy()
        reference = float(p1[col])
        ax.plot(scales, scores, marker="o", lw=2.4, color="#C14924")
        ax.axhline(reference, ls="--", lw=1.8, color="#2A6FBB", label="P1 reference")
        for x, score in zip(scales, scores):
            delta = score - reference
            perf_delta = delta if higher_better else -delta
            ax.annotate(
                f"{score:.4f}\nΔperf {perf_delta:+.4f}",
                (x, score),
                xytext=(0, 9 if score >= reference else -28),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        values = np.r_[scores, reference]
        spread = max(values.max() - values.min(), 0.001)
        ax.set_ylim(values.min() - 0.22 * spread, values.max() + 0.25 * spread)
        ax.set_title(f"{label}\n({'higher' if higher_better else 'lower'} is better)")
        ax.set_xlabel("Spatial blocking scale (km)")
        ax.set_xticks(scales)
        ax.set_ylabel("Score")
        ax.legend(loc="best")

    fig.suptitle(
        "P3 spatial generalization: score and performance change relative to P1",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.035,
        "AUPRC remains below P1 at every scale, but improves from 10 km to 100 km; "
        "there is no monotonic cross-scale degradation in the available results.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    _save(fig, "03_p3_multiscale_degradation.png")
    return gaps


def _model_inputs() -> tuple[object, pd.DataFrame, list[str], pd.Series]:
    model = joblib.load(MODEL_PATH)
    features = pd.read_csv(FEATURE_PATH)["feature"].astype(str).tolist()
    required = list(dict.fromkeys(features + ["date", "y", "BIOME"]))
    df = pd.read_parquet(DATA_PATH, columns=required)
    year = pd.to_datetime(df["date"]).dt.year
    test = year >= 2023
    x_test = df.loc[test, features].replace([np.inf, -np.inf], np.nan)
    meta = df.loc[test, ["BIOME", "y"]].copy()

    if getattr(model, "n_features_in_", len(features)) != len(features):
        raise ValueError(
            f"Model expects {model.n_features_in_} features, but {FEATURE_PATH.name} "
            f"contains {len(features)}."
        )
    if len(x_test) != len(meta):
        raise AssertionError("Test feature and metadata rows are misaligned.")
    return model, x_test, features, meta


def _group_indices(features: list[str]) -> dict[str, list[int]]:
    indices = {
        group: [features.index(name) for name in names if name in features]
        for group, names in FEATURE_GROUPS.items()
    }
    missing_groups = [group for group, idx in indices.items() if not idx]
    if missing_groups:
        raise ValueError(f"No model features found for groups: {missing_groups}")
    return indices


def plot_global_shap_and_dependence(
    model: object, x_test: pd.DataFrame, features: list[str]
) -> None:
    rng = np.random.default_rng(SEED)
    n = min(3000, len(x_test))
    sample_pos = rng.choice(len(x_test), size=n, replace=False)
    xs = x_test.iloc[sample_pos].copy()
    shap_values = model.predict(xs, pred_contrib=True)[:, :-1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:][::-1]

    pd.DataFrame(
        {
            "feature": [features[i] for i in top_idx],
            "mean_abs_shap": mean_abs[top_idx],
        }
    ).to_csv(
        DATA / "04_shap_feature_importance.csv",
        index=False,
        float_format="%.8f",
    )

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for row, feature_idx in enumerate(top_idx):
        values = xs.iloc[:, feature_idx].to_numpy(dtype=float)
        finite = np.isfinite(values)
        fill = float(np.nanmedian(values[finite])) if finite.any() else 0.0
        values = np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)
        lo, hi = np.percentile(values, [2, 98])
        color = np.clip((values - lo) / (hi - lo + 1e-12), 0, 1)
        jitter = rng.normal(0, 0.115, n)
        ax.scatter(
            shap_values[:, feature_idx],
            len(top_idx) - 1 - row + jitter,
            c=color,
            cmap="coolwarm",
            s=8,
            alpha=0.5,
            linewidths=0,
        )
    ax.axvline(0, color="#777777", lw=0.9)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([features[i] for i in top_idx][::-1], fontsize=8)
    ax.set_xlabel("SHAP contribution to model log-odds")
    ax.set_title("P1 LightGBM global SHAP summary (3,000 test observations)")
    scalar = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, pad=0.02, label="Feature value (low to high)")
    fig.tight_layout()
    _save(fig, "04_shap_beeswarm.png")

    selected = [
        name
        for name in [
            "months_since_last_fire",
            "VPD_mean_mon_lag1_roll3m_mean",
            "P_sum_mon_lag1_roll3m_sum",
        ]
        if name in features
    ]
    fig, axes = plt.subplots(1, len(selected), figsize=(5.1 * len(selected), 4.5))
    axes = np.atleast_1d(axes)
    trigger_rows = []
    for ax, name in zip(axes, selected):
        j = features.index(name)
        x_values = xs[name].to_numpy(dtype=float)
        y_values = shap_values[:, j]
        finite = np.isfinite(x_values)
        ax.scatter(
            x_values[finite],
            y_values[finite],
            s=9,
            alpha=0.3,
            color="#087E8B",
            linewidths=0,
        )
        ax.axhline(0, color="#777777", lw=0.9)
        ax.set_title(name)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("SHAP contribution")
        q10, q90 = np.quantile(x_values[finite], [0.10, 0.90])
        low = float(y_values[finite & (x_values <= q10)].mean())
        high = float(y_values[finite & (x_values >= q90)].mean())
        trigger_rows.append(
            {
                "feature": name,
                "P10_value": q10,
                "P90_value": q90,
                "mean_SHAP_low_decile": low,
                "mean_SHAP_high_decile": high,
                "high_minus_low_SHAP": high - low,
            }
        )
    fig.suptitle("Nonlinear physical-trigger diagnostics", fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_shap_dependence.png")
    pd.DataFrame(trigger_rows).to_csv(
        DATA / "05_nonlinear_trigger_summary.csv",
        index=False,
        float_format="%.8f",
    )


def _top_fraction_metrics(y: np.ndarray, p: np.ndarray, fraction: float) -> tuple[float, float]:
    if y.sum() == 0:
        return 0.0, np.nan
    k = max(1, int(np.ceil(len(y) * fraction)))
    top_y = y[np.argsort(-p, kind="stable")[:k]]
    return float(top_y.mean()), float(top_y.sum() / max(1, y.sum()))


def plot_biome_shap(
    model: object,
    x_test: pd.DataFrame,
    features: list[str],
    meta: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    group_indices = _group_indices(features)
    contribution_rows = []
    performance_rows = []

    for biome in sorted(meta["BIOME"].dropna().astype(int).unique()):
        mask = meta["BIOME"].astype("Int64").eq(biome).to_numpy(dtype=bool)
        positions = np.flatnonzero(mask)
        if not len(positions):
            continue

        sample_positions = (
            positions
            if len(positions) <= 2000
            else rng.choice(positions, size=2000, replace=False)
        )
        xs = x_test.iloc[sample_positions]
        contrib = model.predict(xs, pred_contrib=True)[:, :-1]
        mean_abs = np.abs(contrib).mean(axis=0)
        sums = {
            group: float(mean_abs[idx].sum())
            for group, idx in group_indices.items()
        }
        total = sum(sums.values())
        biome_name, biome_short = BIOME_NAMES.get(
            biome, (f"BIOME {biome}", f"B{biome}")
        )
        for group, value in sums.items():
            contribution_rows.append(
                {
                    "BIOME": biome,
                    "biome_name": biome_name,
                    "biome_short": biome_short,
                    "feature_group": group,
                    "mean_abs_shap_sum": value,
                    "share_within_four_groups": value / total if total else np.nan,
                    "shap_sample_n": len(sample_positions),
                }
            )

        x_all = x_test.iloc[positions]
        y = meta.iloc[positions]["y"].astype(int).to_numpy()
        p = model.predict_proba(x_all)[:, 1]
        pred = p >= threshold
        top5_precision, top5_recall = _top_fraction_metrics(y, p, 0.05)
        has_both_classes = np.unique(y).size == 2
        performance_rows.append(
            {
                "BIOME": biome,
                "biome_name": biome_name,
                "biome_short": biome_short,
                "test_n": len(y),
                "fire_n": int(y.sum()),
                "prevalence": float(y.mean()),
                "AUPRC": average_precision_score(y, p) if y.sum() else np.nan,
                "ROC_AUC": roc_auc_score(y, p) if has_both_classes else np.nan,
                "Brier_score": brier_score_loss(y, p),
                "F1": f1_score(y, pred, zero_division=0),
                "Precision": precision_score(y, pred, zero_division=0),
                "Recall": recall_score(y, pred, zero_division=0),
                "Top_5pct_precision": top5_precision,
                "Top_5pct_recall": top5_recall,
                "threshold_from_2022_validation": threshold,
            }
        )

    contributions = pd.DataFrame(contribution_rows)
    performance = pd.DataFrame(performance_rows)
    contributions.to_csv(
        DATA / "06_biome_shap_group_contributions.csv",
        index=False,
        float_format="%.8f",
    )
    performance.to_csv(
        DATA / "06_biome_performance.csv",
        index=False,
        float_format="%.8f",
    )
    pd.DataFrame(
        [
            {"BIOME": biome, "biome_name": name, "biome_short": short}
            for biome, (name, short) in BIOME_NAMES.items()
        ]
    ).to_csv(DATA / "06_biome_mapping.csv", index=False)

    groups = list(FEATURE_GROUPS)
    theta = np.linspace(0, 2 * np.pi, len(groups), endpoint=False)
    theta_closed = np.r_[theta, theta[0]]
    biomes = (
        performance.loc[performance["BIOME"].ne(11)]
        .sort_values("test_n", ascending=False)["BIOME"]
        .astype(int)
        .tolist()
    )
    ncols = 3
    nrows = int(np.ceil(len(biomes) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13.2, 4.1 * nrows),
        subplot_kw={"polar": True},
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, biome in zip(axes, biomes):
        subset = (
            contributions.loc[contributions["BIOME"].eq(biome)]
            .set_index("feature_group")
            .reindex(groups)
        )
        values = subset["share_within_four_groups"].to_numpy(dtype=float)
        closed = np.r_[values, values[0]]
        ax.plot(theta_closed, closed, color="#C14924", lw=2)
        ax.fill(theta_closed, closed, color="#C14924", alpha=0.20)
        ax.set_xticks(theta)
        ax.set_xticklabels(
            ["Fire history", "Drought /\nmoisture", "Wind", "Vegetation"],
            fontsize=8,
        )
        ax.set_ylim(0, max(0.55, np.nanmax(values) * 1.08))
        ax.set_yticklabels([])
        name = subset["biome_short"].dropna().iloc[0]
        n = int(performance.loc[performance["BIOME"].eq(biome), "test_n"].iloc[0])
        fires = int(performance.loc[performance["BIOME"].eq(biome), "fire_n"].iloc[0])
        ax.set_title(f"{name} (BIOME {biome})\nn={n:,}, fires={fires}", pad=16)
    for ax in axes[len(biomes) :]:
        ax.set_visible(False)
    fig.suptitle(
        "BIOME-specific SHAP composition across four physical feature groups",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.01,
        "Each panel sums to 100% across the four prespecified groups; "
        "BIOME is used only for post-hoc grouping, never as a model input.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.98))
    _save(fig, "06_biome_shap_radar.png")
    return contributions, performance


def main() -> None:
    metrics = plot_tree_vs_baselines()
    gaps = plot_p3_generalization_gap()
    model, x_test, features, meta = _model_inputs()
    plot_global_shap_and_dependence(model, x_test, features)
    threshold = float(
        metrics.loc[
            metrics["Model"].astype(str).eq("LightGBM (depth-capped)"),
            "Threshold (validation-selected)",
        ].iloc[0]
    )
    contributions, performance = plot_biome_shap(
        model, x_test, features, meta, threshold
    )

    print("Generated CSV tables in:", DATA)
    print("Generated figures in:", RESULTS)
    print(
        metrics[
            ["Model", "AUPRC", "ROC-AUC", "Brier score", "F1"]
        ].to_string(index=False)
    )
    print("\nP3 performance changes relative to P1:")
    print(gaps.to_string(index=False))
    print("\nBIOME SHAP rows:", len(contributions))
    print(
        performance[
            ["BIOME", "biome_short", "test_n", "fire_n", "AUPRC", "Brier_score"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
