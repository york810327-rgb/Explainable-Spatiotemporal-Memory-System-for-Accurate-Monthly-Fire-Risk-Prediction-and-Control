import os
import pandas as pd
import numpy as np
LOCAL_DATA_PATH = "./data/step7v3_2_P1_time_holdout_lead1F1.parquet" 
print("正在尝试读取本地的 Parquet 数据文件...")
if not os.path.exists(LOCAL_DATA_PATH):
    print(f"找不到文件！请检查路径和文件名是否完全匹配：{LOCAL_DATA_PATH}")
    print("提示：请确保该文件确实存放在 D 盘的 SCI 文件夹下。")
else:
    df_raw = pd.read_parquet(LOCAL_DATA_PATH, engine="fastparquet")
    print(f"成功读取文件！")
    print(f"数据大小: 共 {len(df_raw):,} 行， {df_raw.shape[1]} 个特征（列）\n")
    print("前 3 行数据预览：")
    print(df_raw.head(3))
print("\n正在启动学术防泄漏硬拦截审计...")
FORBIDDEN_EXACT = {"burned_area_m2", "burned_flag", "burned_frac", "ever_burned", "valid_weight"}
FORBIDDEN_PREFIX = ("label_aux_", "burned_", "valid_weight")
spy_columns = []
for col in df_raw.columns:
    if col in FORBIDDEN_EXACT:
        spy_columns.append(col)
    else:
        for prefix in FORBIDDEN_PREFIX:
            if col.startswith(prefix) and col != "y": 
                spy_columns.append(col)
                break
spy_columns = sorted(list(set(spy_columns)))
print(f"本地审计发现！共有 {len(spy_columns)} 个特征包含泄漏风险，必须拦截。")
print(f"危险列名单：{spy_columns}")
df_clean = df_raw.drop(columns=spy_columns, errors="ignore")
print(f"拦截清洗完毕！安全数据集的特征数由 {df_raw.shape[1]} 优化为 {df_clean.shape[1]}")
OUTPUT_CLEAN_PATH = "./data/df_clean_local.parquet"
df_clean.to_parquet(OUTPUT_CLEAN_PATH, index=False)
print(f"任务已完成！干净的数据已保存至本地磁盘：{OUTPUT_CLEAN_PATH}")