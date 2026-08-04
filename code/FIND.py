import pandas as pd
import joblib
import os

data_dir = "../../data/原始data/"

# 1. 加载官方指定底座和全量大表
print("正在读取官方底座和全量大表...")
df_base = pd.read_parquet(data_dir + 'p3_weight_dataset.parquet', engine='fastparquet')
df_master = pd.read_parquet(data_dir + 'lead1_F1_wgrid50_100.parquet', engine='fastparquet')

# 记录底座原始行数，用于防膨胀校验
original_row_count = len(df_base)

# 2. 安全 Merge：从全量大表中提取需要的辅助字段和缺失特征
print("正在执行安全的双主键合并 (Merge)...")
# 🔥 关键修改：把模型报错缺失的 pi_clip, d_weight_clip, fire_tslf 也加进来一起拿走
cols_to_extract = ['cell_id', 'date', 'valid_core', 'pi_clip', 'd_weight_clip', 'fire_tslf']
df_aux = df_master[cols_to_extract].drop_duplicates(subset=['cell_id', 'date'])

# 执行左连接，严格校验多对一或一对一关系，防止数据行数爆炸
df_test = pd.merge(
    left=df_base,
    right=df_aux,
    on=['cell_id', 'date'],
    how='left',
    validate='1:1' # 严格安全校验
)

# 校验行数是否发生错乱
assert len(df_test) == original_row_count, f"严重错误：合并后行数发生变化！原行数 {original_row_count}, 现行数 {len(df_test)}"
print("✅ 合并校验通过：行数未发生膨胀，主键对齐完美。")
#FIND Blind
# 3. 加载 AI 大脑（模型权重）
print("正在加载 P3-100km 模型...")
model = joblib.load(data_dir + 'lgbm_p3_100km_model.joblib')

# 4. 自动匹配特征（剔除刚才拼进来的非特征辅助列）
if hasattr(model, "feature_name_"):
    feature_cols = model.feature_name_
elif hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    # 自动剔除所有非特征列
    ignore_cols = ['cell_id', 'date', 'BIOME', 'biome', 'y', 'valid_core', 'fold', 'weight']
    feature_cols = [col for col in df_test.columns if col not in ignore_cols]

X_test = df_test[feature_cols]

# 5. 让 AI 开始做题，预测起火概率
print("正在生成每个网格的预测概率...")
df_test['pred_prob'] = model.predict_proba(X_test)[:, 1]

# 6. 寻找预警盲区（实际起火 y==1，且预测概率 < 0.1）
# 还可以根据班长要求，加上 valid_core == 1 的质控条件
blind_spots = df_test[
    (df_test['y'] == 1) &
    (df_test['pred_prob'] < 0.1) &
    (df_test['valid_core'] == 1) # 加入了质控字段过滤
]

print(f"\n🎯 官方标准下，总共找到 {len(blind_spots)} 个有效预警盲区网格！")

# 7. 按照生态区（BIOME）统计盲区分布
biome_col = 'BIOME' if 'BIOME' in df_test.columns else 'biome'
blind_spot_counts = blind_spots[biome_col].value_counts().sort_index()

print("\n🚨 官方标准 - 各生态区预警盲区（漏报）数量统计：")
print(blind_spot_counts)

# 把盲区数据保存下来
output_path = "../../data/processed/p3_100km_blind_spots_official.csv"
blind_spots.to_csv(output_path, index=False)
print(f"✅ 官方盲区明细已保存至 {output_path}")