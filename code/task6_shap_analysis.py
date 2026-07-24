"""Final LY deliverable: strict P1 evaluation, P3 generalization and SHAP.

Run from the project folder with the geo-ebm Conda environment:
    python run_ly_analysis.py

P1 protocol: train 2015-2021, validation 2022, test 2023-2024.
No cell/grid IDs, BIOME, STRATUM, dates, labels or sampling weights are model inputs.
"""
from pathlib import Path
import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 20260721
TOP_FRACTIONS = (0.01, 0.05, 0.10)
sns.set_theme(style="whitegrid", context="talk")


def choose_f1_threshold(y, p):
    """Choose operating threshold only on validation data, never on test data."""
    candidates = np.unique(np.quantile(p, np.linspace(.01, .99, 199)))
    f1s = [f1_score(y, p >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def metric_row(y, p, threshold):
    pred = p >= threshold
    row = {
        "AUPRC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "Brier score": brier_score_loss(y, p),
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Threshold (validation-selected)": threshold,
    }
    order = np.argsort(-p, kind="stable")
    for fraction in TOP_FRACTIONS:
        k = max(1, int(np.ceil(len(y) * fraction)))
        top_y = np.asarray(y)[order[:k]]
        tag = f"Top-{int(fraction*100)}%"
        row[f"{tag} precision"] = top_y.mean()
        row[f"{tag} recall"] = top_y.sum() / max(1, np.asarray(y).sum())
        row[f"{tag} hits"] = int(top_y.sum())
    return row


def bootstrap_std(y, p, threshold, n_boot=100):
    """Bootstrap test-set uncertainty; fixed seed makes the output reproducible."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y); p = np.asarray(p)
    rows = []
    for _ in range(n_boot):
        ind = rng.integers(0, len(y), len(y))
        # Resample until both labels are present; needed for ROC-AUC.
        if np.unique(y[ind]).size < 2:
            continue
        rows.append(metric_row(y[ind], p[ind], threshold))
    boot = pd.DataFrame(rows)
    return boot.mean(numeric_only=True).add_suffix(" mean (bootstrap)"), boot.std(numeric_only=True, ddof=1).add_suffix(" std (bootstrap)")


def make_linear_model(features, penalty, c=0.1, l1_ratio=None):
    # Solver choice is deliberate: full P1 has ~250k training records; saga is
    # required only for ElasticNet and is unnecessarily slow for pure L1/L2.
    if penalty == "elasticnet":
        classifier = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=1e-4,
                                   l1_ratio=l1_ratio, max_iter=1200, tol=1e-3,
                                   random_state=SEED, early_stopping=True,
                                   validation_fraction=.1, n_iter_no_change=8)
        return Pipeline([
            ("preprocess", ColumnTransformer([
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), features)
            ])),
            ("model", classifier),
        ])
    solver = {"l2": "lbfgs", "l1": "liblinear"}[penalty]
    kwargs = dict(penalty=penalty, C=c, solver=solver, max_iter=500,
                  tol=1e-3, random_state=SEED)
    if l1_ratio is not None:
        kwargs["l1_ratio"] = l1_ratio
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), features)
        ])),
        ("model", LogisticRegression(**kwargs)),
    ])


def save_metric_table(records):
    rows = []
    for name, y, p, threshold in records:
        direct = pd.Series(metric_row(y, p, threshold))
        means, stds = bootstrap_std(y, p, threshold)
        merged = pd.concat([direct, means, stds])
        merged["Model"] = name
        merged["Protocol"] = "P1: train 2015-2021; validation 2022; test 2023-2024"
        rows.append(merged)
    result = pd.DataFrame(rows)
    front = ["Model", "Protocol", "AUPRC", "ROC-AUC", "Brier score", "F1", "Precision", "Recall",
             "Top-1% precision", "Top-1% recall", "Top-1% hits", "Top-5% precision", "Top-5% recall", "Top-5% hits",
             "Top-10% precision", "Top-10% recall", "Top-10% hits", "Threshold (validation-selected)"]
    result = result[[c for c in front if c in result] + [c for c in result if c not in front]]
    result.to_csv(OUT / "01_p1_high_metrics.csv", index=False, float_format="%.6f")
    return result


def plot_p1_comparison(metrics):
    display = metrics.melt(id_vars="Model", value_vars=["AUPRC", "ROC-AUC", "Brier score", "F1"],
                           var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(15, 6.5))
    sns.barplot(data=display, x="Metric", y="Value", hue="Model", ax=ax,
                palette=["#157f7b", "#e68a2e", "#9270ad", "#b1492f"])
    ax.set_title("P1 time-extrapolation: tree model versus linear baselines")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.legend(title="Model", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "02_p1_tree_vs_baselines.png", dpi=260); plt.close(fig)


def plot_p3_scales():
    p1 = pd.read_csv(ROOT / "lgbm_p1_metrics.csv")
    p3 = pd.concat([pd.read_csv(p) for p in sorted(ROOT.glob("lgbm_p3_*km_metrics.csv"))], ignore_index=True)
    p3["Scale (km)"] = p3["Protocol"].str.extract(r"p3_(\d+)km").astype(int)
    combined = pd.concat([p1.assign(Scale="P1"), p3.assign(Scale=p3["Scale (km)"])], ignore_index=True)
    combined.to_csv(OUT / "03_official_p1_p3_multiscale_metrics.csv", index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["AUPRC", "ROC-AUC", "Brier_Score"]):
        sp = p3.sort_values("Scale (km)")
        ax.plot(sp["Scale (km)"], sp[metric], marker="o", lw=2.5, color="#b1492f")
        for _, r in sp.iterrows():
            ax.annotate(f"{r[metric]:.3f}", (r["Scale (km)"], r[metric]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=10)
        ax.axhline(float(p1[metric].iloc[0]), ls="--", color="#276fbf", label="P1 reference")
        ax.set_title(metric); ax.set_xlabel("Spatial blocking scale (km)"); ax.legend(fontsize=9)
    fig.suptitle("P3 spatial generalization across blocking scales", y=1.03)
    fig.tight_layout(); fig.savefig(OUT / "03_p3_multiscale_degradation.png", dpi=260, bbox_inches="tight"); plt.close(fig)
    return p1, p3


def shap_outputs(tree, x_test, features):
    rng = np.random.default_rng(SEED)
    ix = rng.choice(len(x_test), size=min(3000, len(x_test)), replace=False)
    xs = x_test.iloc[ix].copy()
    contrib = tree.predict(xs, pred_contrib=True)[:, :-1]
    mean_abs = np.abs(contrib).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:][::-1]
    top_features = [features[i] for i in top_idx]
    importance = pd.DataFrame({"feature": top_features, "mean_abs_shap": mean_abs[top_idx]})
    importance.to_csv(OUT / "04_shap_feature_importance.csv", index=False, float_format="%.8f")

    fig, ax = plt.subplots(figsize=(11, 8))
    for row, j in enumerate(top_idx):
        values = xs.iloc[:, j].to_numpy(dtype=float); values = np.nan_to_num(values, nan=np.nanmedian(values))
        sv = contrib[:, j]; jitter = rng.normal(0, .12, len(sv))
        lo, hi = np.percentile(values, [2, 98]); color = np.clip((values-lo)/(hi-lo+1e-12), 0, 1)
        ax.scatter(sv, len(top_idx)-1-row+jitter, c=color, cmap="coolwarm", s=8, alpha=.5, linewidths=0)
    ax.axvline(0, color="grey", lw=.8); ax.set_yticks(range(len(top_idx))); ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("SHAP value (effect on model log-odds)"); ax.set_title("P1 LightGBM SHAP summary (3,000 test observations)")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0,1)); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Feature value (low → high)")
    fig.tight_layout(); fig.savefig(OUT / "04_shap_beeswarm.png", dpi=260); plt.close(fig)

    # Prespecified physical variables make the nonlinear mechanism discussion auditable.
    dep_features = [f for f in ["months_since_last_fire", "VPD_mean_mon_lag1_roll3m_mean", "P_sum_mon_lag1_roll3m_sum"] if f in features]
    fig, axes = plt.subplots(1, len(dep_features), figsize=(5.4*len(dep_features), 4.8))
    if len(dep_features) == 1: axes = [axes]
    trigger_rows = []
    for ax, feature in zip(axes, dep_features):
        j = features.index(feature); values = xs[feature].to_numpy(dtype=float); sv = contrib[:, j]
        ax.scatter(values, sv, s=9, alpha=.35, color="#157f7b"); ax.axhline(0, color="grey", lw=.8)
        ax.set_title(feature); ax.set_xlabel("Feature value"); ax.set_ylabel("SHAP value")
        q10, q90 = np.nanquantile(values, [.10, .90])
        low = float(np.nanmean(sv[values <= q10])); high = float(np.nanmean(sv[values >= q90]))
        trigger_rows.append({"feature": feature, "P10 value": q10, "P90 value": q90,
                             "mean SHAP at low decile": low, "mean SHAP at high decile": high,
                             "high_minus_low_SHAP": high-low})
    fig.suptitle("Nonlinear physical trigger diagnostics", y=1.03)
    fig.tight_layout(); fig.savefig(OUT / "05_shap_dependence.png", dpi=260, bbox_inches="tight"); plt.close(fig)
    triggers = pd.DataFrame(trigger_rows)
    triggers.to_csv(OUT / "05_nonlinear_trigger_summary.csv", index=False, float_format="%.8f")

    groups = {
        "Fire history": ["y_lag1", "y_lag12", "fire_count_12m", "months_since_last_fire"],
        "Drought / moisture": ["PET_sum_mon_lag1", "P_sum_mon_lag1", "VPD_mean_mon_lag1", "VPD_mean_mon_lag1_roll3m_mean", "SM1_mean_mon_lag1", "RH_mean_mon_lag1"],
        "Wind": ["WD_R_mon_lag1", "WD_u_mon_lag1", "WD_v_mon_lag1", "WS_max_mon_lag1", "WS_mean_mon_lag1", "WS_strong_frac_lag1"],
        "Vegetation": ["NDVI_mean_mon_lag1", "EVI_mean_mon_lag1", "treecover_2015", "frac_forest"],
        "Topography / human": ["elev_mean", "slope_mean", "dist_built_m", "frac_crop", "BUILT_mean"],
    }
    group_values = {group: float(sum(mean_abs[features.index(f)] for f in fs if f in features)) for group, fs in groups.items()}
    pd.DataFrame({"feature_group": group_values.keys(), "mean_abs_shap_sum": group_values.values()}).to_csv(OUT / "06_shap_group_contributions.csv", index=False)
    labels = list(group_values); values = np.array(list(group_values.values())); values = values / values.max()
    theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={"polar": True})
    ax.plot(np.r_[theta, theta[0]], np.r_[values, values[0]], color="#b1492f", lw=2.5)
    ax.fill(np.r_[theta, theta[0]], np.r_[values, values[0]], color="#b1492f", alpha=.22)
    ax.set_xticks(theta); ax.set_xticklabels(labels, fontsize=10); ax.set_yticklabels([]); ax.set_title("P1 grouped SHAP contribution", pad=24)
    fig.tight_layout(); fig.savefig(OUT / "06_shap_physical_driver_radar.png", dpi=260); plt.close(fig)
    return importance, triggers, group_values


def main():
    df = pd.read_parquet(ROOT / "p3_weight_dataset.parquet")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    features = pd.read_csv(ROOT / "full_features.csv")["feature"].tolist()
    assert set(features).issubset(df.columns)
    x = df[features].replace([np.inf, -np.inf], np.nan); y = df["y"].astype(int)
    train = df.year <= 2021; val = df.year == 2022; test = df.year >= 2023
    assert train.any() and val.any() and test.any()
    xtr, xv, xt = x.loc[train], x.loc[val], x.loc[test]
    ytr, yv, yt = y.loc[train], y.loc[val], y.loc[test]

    # Match the supplied P1 model family: balanced classes, depth capped at 5,
    # 500 trees and no test-set-driven tuning.  An earlier early-stopping trial
    # stopped at one tree on the highly imbalanced 2022 validation year, which is
    # not a meaningful fitted model.
    tree = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=.03,
        num_leaves=31, max_depth=5, min_child_samples=20, colsample_bytree=.8,
        subsample=.8, class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)
    tree.fit(xtr, ytr)
    joblib.dump(tree, OUT / "p1_lgbm_strict_model.joblib")

    models = [("LightGBM (depth-capped)", tree)]
    for name, penalty, ratio in [("Logistic L2 baseline", "l2", None), ("Logistic L1 baseline", "l1", None), ("ElasticNet baseline", "elasticnet", .5)]:
        model = make_linear_model(features, penalty, l1_ratio=ratio)
        model.fit(xtr, ytr); models.append((name, model))

    records, prediction = [], pd.DataFrame({"cell_id": df.loc[test, "cell_id"].to_numpy(), "date": df.loc[test, "date"].to_numpy(), "y": yt.to_numpy()})
    for name, model in models:
        pv = model.predict_proba(xv)[:,1]; pt = model.predict_proba(xt)[:,1]
        threshold = choose_f1_threshold(yv, pv)
        records.append((name, yt, pt, threshold)); prediction[name] = pt
    prediction.to_parquet(OUT / "p1_test_predictions.parquet", index=False)
    metrics = save_metric_table(records); plot_p1_comparison(metrics)
    p1_official, p3 = plot_p3_scales()
    tree_importance, triggers, groups = shap_outputs(tree, xt, features)

    best_base = metrics.loc[metrics.Model != "LightGBM (depth-capped)"].sort_values("AUPRC", ascending=False).iloc[0]
    tree_row = metrics.loc[metrics.Model == "LightGBM (depth-capped)"].iloc[0]
    p3_drop = float(p3.loc[p3["Scale (km)"] == 100, "AUPRC"].iloc[0] - p3.loc[p3["Scale (km)"] == 10, "AUPRC"].iloc[0])
    trigger_text = "\n".join(
        f"- **{r.feature}**：从低十分位（{r['P10 value']:.3g}）到高十分位（{r['P90 value']:.3g}）时，平均 SHAP 变化为 {r.high_minus_low_SHAP:.3f}。"
        for _, r in triggers.iterrows())
    readme = f"""# LY 最终交付说明

## 协议与防泄漏
P1 采用严格时间外推：2015–2021 训练、2022 验证、2023–2024 测试。模型输入仅为 `full_features.csv` 中的物理与历史特征；`cell_id`、所有空间阻断 ID、BIOME、STRATUM、日期、标签和抽样权重均未进入模型。LightGBM 最大深度固定为 5、树数固定为 500；F1 的操作阈值只由 2022 验证集确定，测试集不参与调参。

## P1 树模型与线性基线
最终树模型测试集 AUPRC 为 **{tree_row['AUPRC']:.4f}**，最佳线性基线（{best_base.Model}）为 **{best_base.AUPRC:.4f}**。完整的 AUPRC、ROC-AUC、Brier score、F1、Precision、Recall、Top-1/5/10% 命中指标及 100 次 Bootstrap 均值/标准差见 `01_p1_high_metrics.csv`。F1 阈值始终在 2022 验证集选定后固定应用于 2023–2024 测试集。

## P3 多尺度泛化
官方 P3 结果中，空间阻断尺度由 100 km 缩小至 10 km 时，AUPRC 从 0.2223 降至 0.2062，绝对下降 **{p3_drop:.4f}**（约 {p3_drop/0.22227477070901802:.1%}）；ROC-AUC 在 0.9435–0.9457 间基本稳定。这说明细尺度空间外推主要损失稀有起火样本的排序能力，因而需要重点关注局地气候/地貌组合与训练区域不同的网格。

## 非线性触发机制归因
{trigger_text}

上述数值来自 P1 最终树模型 3,000 个测试样本的精确树 SHAP 贡献（LightGBM `pred_contrib`）。蜂群图用于判断重要性和方向；依赖图与 `05_nonlinear_trigger_summary.csv` 用于核验上述低/高状态差异。机制上，火历史体现燃料与火发生的时间记忆；持续干燥度（降水、VPD、土壤湿度、PET）调节燃料可燃性；风速/风向调节扩散条件；NDVI/EVI、树冠覆盖和土地覆盖反映燃料数量与连通性；地形和人类活动共同塑造局地微气候与点火机会。非线性关系由树模型直接学习，不能把单一变量的 SHAP 方向误解为独立因果效应。

## 文件用途
见本目录中的结果图、指标表、预测概率和 `p1_lgbm_strict_model.joblib`。该模型文件可供组内继续进行个例解释或预警制图。
"""
    (OUT / "README_LY_FINAL.md").write_text(readme, encoding="utf-8")
    print(metrics[["Model", "AUPRC", "ROC-AUC", "Brier score", "F1", "Precision", "Recall", "Top-5% precision", "Top-5% recall"]].to_string(index=False))
    print(f"Final outputs: {OUT}")


if __name__ == "__main__":
    main()
