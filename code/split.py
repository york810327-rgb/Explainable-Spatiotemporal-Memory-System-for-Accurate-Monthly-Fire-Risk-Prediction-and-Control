import os
import pandas as pd
import numpy as np
print("正在使用 fastparquet 引擎读取已生成的干净数据集...")
CLEAN_DATA_PATH = "./data/df_clean_local.parquet"
if not os.path.exists(CLEAN_DATA_PATH):
    print(f"未找到该文件：{CLEAN_DATA_PATH}！请确保脚本已经成功运行并输出了文件。")
else:
    df_clean = pd.read_parquet(CLEAN_DATA_PATH, engine="fastparquet")
    print("成功加载本地清洗数据！")
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    df_clean["year"] = df_clean["date"].dt.year.astype("Int64")
    print("开始执行 P1 时间外推协议切分...")
    # 训练集（Train）：2015–2021年
    # 验证集（Val）：2022年
    # 测试集（Test）：2023–2024年
    train_local = df_clean[df_clean["year"] <= 2021].reset_index(drop=True)
    val_local = df_clean[df_clean["year"] == 2022].reset_index(drop=True)
    test_local = df_clean[(df_clean["year"] >= 2023) & (df_clean["year"] <= 2024)].reset_index(drop=True)
    print(f"本地 [训练集 (2015-2021)] 样本数: {len(train_local):,} 行")
    print(f"本地 [验证集 (2022)]       样本数: {len(val_local):,} 行")
    print(f"本地 [测试集 (2023-2024)]   样本数: {len(test_local):,} 行") 
    print("\n正在进行时空记忆物理特征组完整性抽查...")
    crucial_vars = ["VPD_mean_mon_lag1", "NDVI_mean_mon_lag1", "P_sum_mon_lag1"]
    print("核心物理消融特征抽检状态：")
    for var in crucial_vars:
        if var in df_clean.columns:
            print(f"[状态正常] 关键特征 {var} 完好存在。")
        else:
            print(f"[提示] 未在切分表中直接发现 {var}（由于这是专属切分表，该特征存放在另一张特征表里，属于正常现象）。")