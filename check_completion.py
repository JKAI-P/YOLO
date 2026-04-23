#!/usr/bin/env python3
"""
自动检测实验完成状态
当所有6个实验都完成后，创建完成标记文件
"""

import os
import time
from datetime import datetime

def check_all_completed():
    """检查所有实验是否完成"""
    base_dir = "runs/train"
    experiments = [
        "1_baseline",
        "2_cbam", 
        "3_cbam_wf",
        "4_iyuav_det",
        "comp_yolov5s",
        "comp_yolov7"
    ]
    
    completed_count = 0
    total_experiments = len(experiments)
    
    for exp in experiments:
        results_file = os.path.join(base_dir, exp, "results.csv")
        if os.path.exists(results_file):
            # 检查是否有300+行数据（包含表头）
            with open(results_file, 'r') as f:
                line_count = len(f.readlines())
                if line_count >= 301:  # 300 epochs + 1 header
                    completed_count += 1
                    print(f"✅ {exp} 已完成 ({line_count-1}/300 epochs)")
                else:
                    print(f"🔄 {exp} 进行中 ({line_count-1}/300 epochs)")
        else:
            print(f"⏳ {exp} 尚未开始或初始化中")
    
    return completed_count == total_experiments

def main():
    """主检测循环"""
    print(f"🔍 开始监控实验完成状态 - {datetime.now()}")
    print("=" * 50)
    
    while True:
        if check_all_completed():
            # 创建完成标记文件
            with open("ALL_EXPERIMENTS_COMPLETED.txt", "w") as f:
                f.write(f"所有实验已于 {datetime.now()} 完成！\n")
                f.write("结果位置: runs/train/\n")
                f.write("论文数据源: 各目录下的 results.csv 文件\n")
            
            print("\n🎉 🎉 🎉 所有实验已完成！🎉 🎉 🎉")
            print("📄 完成标记文件已创建: ALL_EXPERIMENTS_COMPLETED.txt")
            print("📊 现在可以提取真实数据填入论文了！")
            break
        else:
            print(f"\n📊 进度: {sum(1 for exp in ['1_baseline', '2_cbam', '3_cbam_wf', '4_iyuav_det', 'comp_yolov5s', 'comp_yolov7'] 
                   if os.path.exists(f'runs/train/{exp}/results.csv') and 
                   sum(1 for _ in open(f'runs/train/{exp}/results.csv')) >= 301)}/6 个实验完成")
            print("😴 继续监控中... (每10分钟检查一次)")
            time.sleep(600)  # 每10分钟检查一次

if __name__ == "__main__":
    main()