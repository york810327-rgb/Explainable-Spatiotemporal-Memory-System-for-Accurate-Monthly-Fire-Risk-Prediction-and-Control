import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
def plot_weight_ablation_metrics():
    print("Plotting Weight Strategy Ablation: AUPRC vs Brier Score...")
    metrics_path = './results/03_weight_ablation_metrics.csv'
    output_png = './results/03_weight_strategy_comparison.png'
    if not os.path.exists(metrics_path):
        print(f"ERROR: Result file not found: {metrics_path}")
        return  
    #读取数据
    df = pd.read_csv(metrics_path)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    #创建 1 行 2 列的子图，axes 是左图，axes[1] 是右图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    #策略名称映射
    strategy_mapping = {
        'cw_none__sw_none': 'Baseline\n(No Weight)',
        'cw_balanced__sw_none': 'Class Balanced\n(Algorithm)',
        'cw_none__sw_dweight': 'Spatial Design Weight\n(Ours)'
    }
    df['Strategy_Label'] = df['Strategy'].map(strategy_mapping).fillna(df['Strategy'])
    #绘制 AUPRC 对比图
    sns.barplot(data=df, x='Strategy_Label', y='AUPRC', ax=axes[0], hue='Strategy_Label', palette='Blues_d', legend=False)
    axes[0].set_title('AUPRC Comparison (Higher is Better)', fontweight='bold')
    axes[0].set_ylabel('AUPRC')
    axes[0].set_xlabel('')
    #绘制 Brier Score 对比图
    sns.barplot(data=df, x='Strategy_Label', y='Brier_Score', ax=axes[1], hue='Strategy_Label', palette='Reds_d', legend=False)
    axes[1].set_title('Brier Score Comparison (Lower is Better)', fontweight='bold')
    axes[1].set_ylabel('Brier Score')
    axes[1].set_xlabel('')
    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig(output_png, dpi=300)
    print(f"\nChart successfully generated and saved to {output_png}!")
if __name__ == "__main__":
    plot_weight_ablation_metrics()