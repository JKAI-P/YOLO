#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成YOLO训练指标图表的脚本
"""

import os
from ultralytics.utils.plotting import plot_results

def plot_ablation_cbam_results():
    """绘制ablation-cbam实验的训练结果图表"""
    # 指定results.csv文件路径
    results_csv_path = "runs/detect/runs/detect/ablation-cbam/results.csv"
    
    if not os.path.exists(results_csv_path):
        print(f"❌ 找不到results.csv文件: {results_csv_path}")
        return
    
    print(f"📊 正在绘制训练结果图表...")
    print(f"📁 数据源: {results_csv_path}")
    
    try:
        # 使用Ultralytics内置的plot_results函数
        plot_results(file=results_csv_path)
        print("✅ 图表已成功生成！")
        print("📁 图表保存在: runs/detect/runs/detect/ablation-cbam/ 目录下")
        
    except Exception as e:
        print(f"❌ 绘图过程中出现错误: {e}")
        print("💡 提示: 确保已正确安装matplotlib等依赖库")

if __name__ == "__main__":
    plot_ablation_cbam_results()
```