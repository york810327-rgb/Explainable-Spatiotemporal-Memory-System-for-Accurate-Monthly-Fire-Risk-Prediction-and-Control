import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, brier_score_loss
import os
from weight_strategy import get_ablation_weight_strategy
def run_weight_ablation():
    print("开始执行模块三：极度不平衡数据的权重策略消融实验...")
    # 加载专门为模块三、四准备的底座数据
    print("加载 p3_weight_dataset.parquet 数据底座...")
    data_path = './data/p3_weight_dataset.parquet'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件：{data_path}")
    df = pd.read_parquet(data_path)
    # 根据 P1 协议划分训练集和验证集
    print("执行 P1 时间外推划分 (Train: 2015-2021, Val: 2022)...")
    train_data = df[df['year'] <= 2021].reset_index(drop=True)
    val_data = df[df['year'] == 2022].reset_index(drop=True)
    # 定义需要剔除的非特征列
    drop_cols = ['cell_id', 'date', 'year', 'month', 'y', 'pi', 'd_weight', 'weight', 
                 'BIOME', 'STRATUM', 'grid10km_id', 'grid50km_id', 'grid100km_id']
    features = [c for c in train_data.columns if c not in drop_cols]
    X_train, y_train = train_data[features], train_data['y']
    X_val, y_val = val_data[features], val_data['y']
    # 提取空间抽样设计权重
    sample_weight_train = train_data['d_weight'].values
    # 循环测试三组策略
    strategies = ['cw_none__sw_none', 'cw_balanced__sw_none', 'cw_none__sw_dweight']
    results = []
    for name in strategies:
        print(f"\n训练策略: {name}")
        cw, sw = get_ablation_weight_strategy(name, d_weight_array=sample_weight_train)
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            class_weight=cw,
            random_state=42,
            n_jobs=-1
        )
        # 拟合模型
        model.fit(X_train, y_train, sample_weight=sw)
        # 预测概率
        y_prob = model.predict_proba(X_val)[:, 1]
        # 计算核心指标: AUPRC 与 Brier Score
        auprc = average_precision_score(y_val, y_prob)
        brier = brier_score_loss(y_val, y_prob)
        print(f"[{name}] AUPRC: {auprc:.4f} | Brier Score: {brier:.4f}")
        results.append({
            'Strategy': name,
            'AUPRC': auprc,
            'Brier_Score': brier
        })
    os.makedirs('./results', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('./results/03_weight_ablation_metrics.csv', index=False)
    print("\n三组权重策略跑批完毕！指标已保存至 ./results/03_weight_ablation_metrics.csv")
if __name__ == "__main__":
    run_weight_ablation()