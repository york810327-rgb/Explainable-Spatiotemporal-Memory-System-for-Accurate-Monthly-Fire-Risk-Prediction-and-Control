import numpy as np
def get_ablation_weight_strategy(strategy_mode, d_weight_array=None):
    """
    极度不平衡数据的权重策略消融接口
    参数:
        strategy_mode: 'cw_none__sw_none', 'cw_balanced__sw_none', 'cw_none__sw_dweight'
        d_weight_array: 空间抽样设计权重数组 (可选)
    返回:
        class_weight, sample_weight
    """
    print(f"\n正在配置“极度不平衡对抗”权重消融策略: 【{strategy_mode}】")
    class_weight = None
    sample_weight = None
    if strategy_mode == 'cw_none__sw_none':
        print("策略 1: 不采用任何权重策略 (Baseline)。")
    elif strategy_mode == 'cw_balanced__sw_none':
        print("策略 2: 启用传统算法类别平衡 (class_weight = 'balanced')。")
        class_weight = 'balanced'
    elif strategy_mode == 'cw_none__sw_dweight':
        print("策略 3: 启用空间抽样设计权重法 (sample_weight = d_weight)。")
        if d_weight_array is None:
            print("警告：选择了 d_weight 策略，但未传入权重数据！")
        else:
            sample_weight = d_weight_array 
    else:
        raise ValueError(f"未知的消融策略：{strategy_mode}")
        
    return class_weight, sample_weight
if __name__ == "__main__":
    # 本地自测接口是否正常工作
    print("代码自测中...")
    cw, sw = get_ablation_weight_strategy('cw_balanced__sw_none')
    cw2, sw2 = get_ablation_weight_strategy('cw_none__sw_dweight', d_weight_array=np.array([1.2, 0.8, 1.0]))
    print("\n接口自测正常！")