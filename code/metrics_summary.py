import os
import pandas as pd
def aggregate_metrics_to_table(metrics_files):
    print("正在汇总所有模型与协议的测评指标表...")
    all_results = []
    for file_path in metrics_files:
        if os.path.exists(file_path):
            df_res = pd.read_csv(file_path)
            all_results.append(df_res)
            print(f"成功读取: {file_path}")
        else:
            print(f"[提示] 未找到结果文件，跳过：{file_path}")
    if all_results:
        # 将所有表格上下拼接
        summary_table = pd.concat(all_results, ignore_index=True)
        out_dir = "./results"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "all_models_metrics_summary.csv")
        summary_table.to_csv(out_path, index=False)
        print(f"\n测试结果汇总完毕！总表已归档至：{out_path}")
    else:
        print("\n没有找到任何结果文件，无法汇总。")
if __name__ == "__main__":
    files_to_merge = [
        './results/logistic_l2_metrics.csv',       
        './results/lgbm_p1_metrics.csv',           
        './results/lgbm_p3_10km_metrics.csv',      
        './results/lgbm_p3_50km_metrics.csv',      
        './results/lgbm_p3_100km_metrics.csv'      
    ]
    aggregate_metrics_to_table(files_to_merge)