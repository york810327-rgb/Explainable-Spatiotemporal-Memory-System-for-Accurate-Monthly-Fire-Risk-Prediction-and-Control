import os
import pandas as pd
import numpy as np
from step7_make_splits import (
    make_p3_spatiotemporal_holdout, 
    assert_p3_heldout_logic, 
    assert_blocks_disjoint
)
def generate_and_audit_p3_splits(df_core, block_cols=['grid10km_id', 'grid50km_id', 'grid100km_id']):
    """
    按 10km, 50km, 100km 尺度生成 P3 时空阻断划分，并执行严格防泄漏审计
    """
    print("开始生成 P3 多尺度空间阻断数据集...")
    # 严格遵照 P3 协议的时间双盲切分要求 (训练: 2015-2021, 验证: 2022, 测试: 2023-2024)
    t_start, t_end = pd.Timestamp('2015-01-01'), pd.Timestamp('2021-12-31')
    v_start, v_end = pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31')
    te_start, te_end = pd.Timestamp('2023-01-01'), pd.Timestamp('2024-12-31')
    p3_splits_dict = {}
    for block_col in block_cols:
        print(f"\n正在按【{block_col}】尺度执行空间阻断...")
        #调用同文件夹下 step7_make_splits.py 的 P3 核心拆分函数
        split_df, folds = make_p3_spatiotemporal_holdout(
            df_core=df_core, 
            block_col=block_col,
            train_start=t_start, train_end=t_end,
            val_start=v_start, val_end=v_end,
            test_start=te_start, test_end=te_end,
            k_blocks=5, 
            seed=42
        )
        #严格防泄漏断言审计
        print(f"[审计中] 正在执行 P3 测试集时间与区块绝对隔离断言检查...")
        #断言1：测试集时间范围符合要求，且测试区块不在训练集/验证集中
        assert_p3_heldout_logic(
            split_df, 
            protocol=f"P3_spatiotemporal_holdout_{block_col}", 
            block_col=block_col, 
            test_start=te_start, 
            test_end=te_end
        )
        #断言2：确保每一个 fold 内的 train 和 test 在空间上完全互斥，不发生“空间泄漏”
        assert_blocks_disjoint(
            split_df, 
            protocol=f"P3_spatiotemporal_holdout_{block_col}", 
            block_col=block_col, 
            role_a="train", 
            role_b="test"
        )
        print(f"[{block_col}] 防泄漏断言通过！数据绝对纯净，未发生空间跨界泄漏。")
        save_path = f"./data/p3_split_{block_col}.parquet"
        split_df.to_parquet(save_path, engine="fastparquet")
        p3_splits_dict[block_col] = save_path
        print(f"切分表已保存至: {save_path}")
    return p3_splits_dict
if __name__ == "__main__":
    FEATURE_DATA_PATH = "./data/lead1_F1_wgrid50_100.parquet"
    
    if not os.path.exists(FEATURE_DATA_PATH):
        print(f"❌ 未找到特征大表：{FEATURE_DATA_PATH}，请检查路径。")
    else:
        print(f"✅ 加载含有空间标签的大表：{FEATURE_DATA_PATH}")
        df_core = pd.read_parquet(FEATURE_DATA_PATH, engine="fastparquet")
        df_core["date"] = pd.to_datetime(df_core["date"], errors="coerce")
        
        # 执行切分
        generate_and_audit_p3_splits(df_core)