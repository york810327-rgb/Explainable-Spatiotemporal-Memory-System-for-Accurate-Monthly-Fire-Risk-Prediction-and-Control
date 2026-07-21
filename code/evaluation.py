import numpy as np

def calculate_advanced_metrics(y_true, y_prob):
    """
    评估指标函数
    y_true: 真实的0-1标签
    y_prob: 模型预测的起火概率
    """
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    # 核心 AUPRC 计算逻辑 (针对极度不平衡火灾数据)
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    if tp[-1] == 0:
        auprc = 0.0
    else:
        precision = tp / np.maximum(tp + fp, 1)
        auprc = float(precision[y_true_sorted == 1].sum() / tp[-1])
        
    # Brier Score 计算 (评估概率校准质量)
    brier_score = float(np.mean((y_prob - y_true) ** 2))
    
    # 简单算一下伪 F1-Score (设定标准阈值 0.5)
    y_pred = (y_prob >= 0.5).astype(int)
    tp_f1 = np.sum((y_pred == 1) & (y_true == 1))
    prec_f1 = tp_f1 / np.maximum(np.sum(y_pred == 1), 1)
    rec_f1 = tp_f1 / np.maximum(np.sum(y_true == 1), 1)
    f1_score = 2 * (prec_f1 * rec_f1) / np.maximum(prec_f1 + rec_f1, 1e-8)
    
    print("================ 情况汇报 ================")
    print(f"核心 AUPRC (越高越好): {auprc:.2%}")
    print(f"Brier Score (越低越好): {brier_score:.4f}")
    print(f"基础 F1-Score (阈值0.5): {f1_score:.2%}")
    print("==================================================")
    return {"auprc": auprc, "brier": brier_score, "f1": f1_score}

# 本地自测
if __name__ == "__main__":
    np.random.seed(42)
    mock_true = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
    mock_prob = np.random.uniform(0, 0.3, size=1000)
    
    print("正在本地调试评估模块...")
    calculate_advanced_metrics(mock_true, mock_prob)