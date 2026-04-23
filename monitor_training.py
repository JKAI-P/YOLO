"""
YOLO训练实时监控脚本
定期读取训练结果CSV文件，输出进度和mAP指标对比

用法：
    python monitor_training.py
"""

import os
import time
import pandas as pd
import glob
from datetime import datetime


# ==================== 配置参数 ====================
# 使用glob模式动态查找最新的iyuav_det_200ep相关目录
def find_latest_iyuav_run():
    """查找最新的iyuav_det_200ep训练目录"""
    pattern = "runs/detect/runs/train/iyuav_det_200ep*/results.csv"
    csv_files = glob.glob(pattern)
    if not csv_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_csv = max(csv_files, key=os.path.getmtime)
    return latest_csv

# 使用glob模式动态查找最新的yolov8_baseline_150ep相关目录
def find_latest_yolov8_run():
    """查找最新的yolov8_baseline_150ep训练目录"""
    pattern = "runs/detect/runs/train/yolov8_baseline_150ep*/results.csv"
    csv_files = glob.glob(pattern)
    if not csv_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_csv = max(csv_files, key=os.path.getmtime)
    return latest_csv

# 获取最新的训练结果文件路径
latest_iyuav_csv = find_latest_iyuav_run()
latest_yolov8_csv = find_latest_yolov8_run()

EXPERIMENTS = {
    "IYUAV-Det (200ep)": latest_iyuav_csv if latest_iyuav_csv else "runs/detect/runs/train/iyuav_det_200ep/results.csv",
    "YOLOv8 Baseline (150ep)": latest_yolov8_csv if latest_yolov8_csv else "runs/detect/runs/train/yolov8_baseline_150ep/results.csv",
}

# 不同实验的不同总epochs数
TOTAL_EPOCHS_CONFIG = {
    "IYUAV-Det (200ep)": 200,
    "YOLOv8 Baseline (150ep)": 150,
}

CHECK_INTERVAL = 60  # 检查间隔（秒）


def read_results(csv_path):
    """读取训练结果CSV文件"""
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        # 确保返回最新的数据行
        return df.iloc[-1:].copy()  # 返回最后一行作为最新状态
    except Exception as e:
        print(f"⚠️  读取失败: {e}")
        return None


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.2f}小时"


def get_experiment_status(name, csv_path):
    """获取单个实验的状态信息"""
    df = read_results(csv_path)
    
    if df is None or df.empty:
        return {
            "name": name,
            "status": "未开始",
            "current_epoch": 0,
            "progress": 0,
            "metrics": None
        }
    
    # 获取最新epoch
    latest = df.iloc[-1]
    current_epoch = int(latest['epoch'])
    total_epochs = TOTAL_EPOCHS_CONFIG.get(name, 200)  # 默认200
    progress = (current_epoch / total_epochs) * 100
    
    # 提取关键指标
    metrics = {
        'box_loss': latest.get('train/box_loss', 0),
        'cls_loss': latest.get('train/cls_loss', 0),
        'dfl_loss': latest.get('train/dfl_loss', 0),
        'precision': latest.get('metrics/precision(B)', 0),
        'recall': latest.get('metrics/recall(B)', 0),
        'mAP50': latest.get('metrics/mAP50(B)', 0),
        'mAP50_95': latest.get('metrics/mAP50-95(B)', 0),
    }
    
    # 判断状态
    if current_epoch >= total_epochs:
        status = "✅ 已完成"
    elif current_epoch == 0:
        status = "🔄 初始化中"
    else:
        status = f"🔥 训练中 ({current_epoch}/{total_epochs})"
    
    return {
        "name": name,
        "status": status,
        "current_epoch": current_epoch,
        "progress": progress,
        "metrics": metrics
    }


def main():
    print("🚀 启动YOLO训练监控")
    print(f"📋 监控配置:")
    print(f"   • 检查间隔: {CHECK_INTERVAL}秒")
    print(f"   • 实验数量: {len(EXPERIMENTS)}")
    for name, epochs in TOTAL_EPOCHS_CONFIG.items():
        print(f"   • {name}: {epochs} epochs")
    print(f"\n💡 提示: 按 Ctrl+C 停止监控\n")
    
    # 添加检查点提醒标记
    fifty_epoch_checked = {name: False for name in EXPERIMENTS}
    
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*100}")
            print(f"  📊 YOLO训练实时监控 - {current_time}")
            print(f"{'='*100}")
            
            all_complete = True
            for name, csv_path in EXPERIMENTS.items():
                # 动态更新路径（每次循环都检查最新路径）
                if "iyuav_det_200ep" in str(csv_path):
                    latest_csv = find_latest_iyuav_run()
                    if latest_csv:
                        csv_path = latest_csv
                elif "yolov8_baseline_150ep" in str(csv_path):
                    latest_csv = find_latest_yolov8_run()
                    if latest_csv:
                        csv_path = latest_csv
                
                df = read_results(csv_path)
                if df is None:
                    print(f"\n🔬 {name}:")
                    print(f"   状态: 未开始")
                    print(f"   等待训练数据...")
                    all_complete = False
                    continue
                
                latest = df.iloc[-1]
                epoch = int(latest['epoch'])
                total_epochs = TOTAL_EPOCHS_CONFIG.get(name, 200)
                progress = epoch / total_epochs * 100
                
                status = "🔥 训练中" if epoch < total_epochs else "✅ 完成"
                
                print(f"\n🔬 {name}:")
                print(f"   状态: {status} ({epoch}/{total_epochs})")
                print(f"   Epoch: {epoch}/{total_epochs} ({progress:.1f}%)")
                print(f"   结果文件: {csv_path}")
                
                # 提取关键指标，防止KeyError
                metrics = {
                    'box_loss': latest.get('train/box_loss', 0),
                    'cls_loss': latest.get('train/cls_loss', 0),
                    'dfl_loss': latest.get('train/dfl_loss', 0),
                    'precision': latest.get('metrics/precision(B)', 0),
                    'recall': latest.get('metrics/recall(B)', 0),
                    'mAP50': latest.get('metrics/mAP50(B)', 0),
                    'mAP50_95': latest.get('metrics/mAP50-95(B)', 0),
                }

                print(f"   ┌─ 训练损失:")
                print(f"   │  • box_loss:  {metrics['box_loss']:.4f}")
                print(f"   │  • cls_loss:  {metrics['cls_loss']:.4f}")
                print(f"   │  • dfl_loss:  {metrics['dfl_loss']:.4f}")
                print(f"   └─ 验证指标:")
                print(f"      • Precision: {metrics['precision']:.4f}")
                print(f"      • Recall:    {metrics['recall']:.4f}")
                print(f"      • mAP@50:    {metrics['mAP50']:.4f}  ⭐")
                print(f"      • mAP@50-95: {metrics['mAP50_95']:.4f}  ⭐")
                
                if epoch < total_epochs:
                    all_complete = False
                
                # 检查点提醒（50 epochs for IYUAV-Det, 50 epochs for YOLOv8）
                check_epoch = 50
                if epoch >= check_epoch and not fifty_epoch_checked[name]:
                    print(f"\n🎯 🎯 🎯 重要检查点：{name} 已达到{check_epoch} epochs！")
                    print(f"   当前mAP@50: {metrics['mAP50']:.4f}")
                    print(f"   请评估是否继续训练到{total_epochs} epochs")
                    print(f"   如果性能提升不明显，可以手动停止训练")
                    fifty_epoch_checked[name] = True
            
            if all_complete:
                print(f"\n🎉 所有实验已完成！")
                break
            
            print(f"\n{'='*100}")
            print(f"\n⏰ 下次更新: {CHECK_INTERVAL}秒后...\n")
            
            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
        print("\n📁 结果保存位置:")
        for name, csv_path in EXPERIMENTS.items():
            # 显示实际使用的路径
            if "iyuav_det_200ep" in str(csv_path):
                actual_path = find_latest_iyuav_run()
            elif "yolov8_baseline_150ep" in str(csv_path):
                actual_path = find_latest_yolov8_run()
            else:
                actual_path = csv_path
            
            abs_path = os.path.abspath(actual_path) if actual_path else csv_path
            dir_path = os.path.dirname(abs_path)
            print(f"   • {name}: {dir_path}/")


if __name__ == "__main__":
    main()