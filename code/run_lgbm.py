import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from evaluation import calculate_advanced_metrics
from weight_strategy import get_ablation_weight_strategy
def run_lgbm_experiment(split_path, feature_path, protocol_name, out_dir):
    """
    统一的 LightGBM 训练与测评管线
    """
    print(f"\n{'='*60}")
    print(f"开始执行 LightGBM 树模型实验: 【{protocol_name}】")
    print(f"{'='*60}")
    # ==========================================
    # 加载数据
    # ==========================================
    print(f"加载划分表: {split_path}")
    df_split = pd.read_parquet(split_path, engine="fastparquet")
    if "date" in df_split.columns:
        df_split["date"] = pd.to_datetime(df_split["date"], errors="coerce")
    print(f"加载特征大表: {feature_path}")
    df_feature = pd.read_parquet(feature_path, engine="fastparquet")
    if "date" in df_feature.columns:
        df_feature["date"] = pd.to_datetime(df_feature["date"], errors="coerce")
    # ==========================================
    # 空间特征无泄漏左连接
    # ==========================================
    print("正在执行 Left Join 双主键合并 (cell_id, date)...")
    overlap_cols = [c for c in df_feature.columns if c in df_split.columns and c not in ["cell_id", "date"]]
    print(f"发现重复的重叠列，将在合并前从右表剔除以防冲突: {overlap_cols}")
    # 将重复列从特征大表中剔除
    df_feature_clean = df_feature.drop(columns=overlap_cols)
    # 再次进行完美左连接
    df_merged = pd.merge(df_split, df_feature_clean, on=["cell_id", "date"], how="left")
    # 根据 split_role 划分训练集和验证/测试集
    train_df = df_merged[df_merged["split_role"] == "train"].copy()
    # 兼容 P1(看 val 验证集) 和 P3(看 test 测试集)
    if "val" in df_merged["split_role"].values:
        eval_df = df_merged[df_merged["split_role"] == "val"].copy()
        eval_name = "验证集(Val)"
    else:
        eval_df = df_merged[df_merged["split_role"] == "test"].copy()
        eval_name = "测试集(Test)"
    print(f"训练集样本数: {len(train_df):,} 行")
    print(f"{eval_name}样本数: {len(eval_df):,} 行")
    # ==========================================
    # 绝对禁止空间ID和泄露元数据进入特征
    # ==========================================
    exclude_cols = [
        "cell_id", "date", "y", "protocol", "fold_id", "split_role", "BIOME", 
        "grid10km_id", "grid50km_id", "grid100km_id", "spatial_block_id", "tile_id", "year"
    ]
    features = [c for c in train_df.columns if c not in exclude_cols]
    X_train, y_train = train_df[features], train_df["y"]
    X_eval, y_eval = eval_df[features], eval_df["y"]
    # ==========================================
    # 初始化树模型
    # ==========================================
    print("初始化 LightGBM 引擎...")
    #如果树过深，它会死记硬背 P1 的地理位置，P3泛化必定崩溃
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,               # 限制深度，保证空间泛化鲁棒性
        num_leaves=31,
        class_weight='balanced',   
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    print("开始拟合树模型 (监控 AUPRC 并启用 Early Stopping)...")
    # 监控验证集表现，30轮内 AUPRC 无提升则提前停止训练，防过拟合
    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        eval_metric="aucpr",  
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    # ==========================================
    # 推断与高阶测评
    # ==========================================
    print(f"在 {eval_name} 上提取预警概率...")
    y_prob = model.predict_proba(X_eval)[:, 1]
    # 调用计算核心指标
    auprc = average_precision_score(y_eval, y_prob)
    roc_auc = roc_auc_score(y_eval, y_prob)
    brier = brier_score_loss(y_eval, y_prob)
    print(f"\n{protocol_name}】 最新战报:")
    print(f"   - AUPRC      : {auprc:.6f}  (基准目标需超越 Logistic-L2 的 0.209984)")
    print(f"   - ROC-AUC    : {roc_auc:.6f}")
    print(f"   - Brier Score: {brier:.6f}  (越小越好)")
    # ==========================================
    # 结果与资产归档至 results 文件夹
    # ==========================================
    os.makedirs(out_dir, exist_ok=True)
    # 存下测评成绩单
    metrics_df = pd.DataFrame({
        "Model": ["LightGBM"],
        "Protocol": [protocol_name],
        "AUPRC": [auprc],
        "ROC-AUC": [roc_auc],
        "Brier_Score": [brier]
    })
    metrics_path = os.path.join(out_dir, f"lgbm_{protocol_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    # 存下模型权重供后期可视化与SHAP解释
    model_path = os.path.join(out_dir, f"lgbm_{protocol_name}_model.joblib")
    joblib.dump(model, model_path)
    print(f"模型与指标均已安全落盘至: {out_dir}\n")
    return metrics_df
# =====================================================================
# 中央控制台
# =====================================================================
if __name__ == "__main__":
    # 配置统一的标准工程路径
    FEATURE_PATH = "./data/lead1_F1_wgrid50_100.parquet"
    OUT_DIR = "./results"
    # 运行 P1 时间外推协议基准跑批
    P1_SPLIT_PATH = "./data/df_clean_local.parquet"
    if os.path.exists(P1_SPLIT_PATH):
        run_lgbm_experiment(P1_SPLIT_PATH, FEATURE_PATH, "p1", OUT_DIR)
    else:
        print(f"找不到 P1 数据：{P1_SPLIT_PATH}")
    # 运行 P3 多尺度空间泛化阻断极限测试
    # 只要 task1_p3_split.py 切好数据，这串循环会自动全部读取并跑出所有尺度的结果
    p3_scales = ["grid10km_id", "grid50km_id", "grid100km_id"]
    for scale in p3_scales:
        p3_split_path = f"./data/p3_split_{scale}.parquet"
        p3_name = scale.replace("grid", "p3_").replace("_id", "") 
        if os.path.exists(p3_split_path):
            run_lgbm_experiment(p3_split_path, FEATURE_PATH, p3_name, OUT_DIR)
        else:
            print(f"[跳过] 尚未找到 {scale} 的 P3 切分表 ({p3_split_path})，等切完重跑即可自动识别。")