"""
YOLO消融实验训练 + 监控脚本
自动串行执行多组消融实验，支持断点续训，同时记录日志

用法：
    python monitor_training.py           # 启动训练+监控
    python monitor_training.py --monitor # 仅监控（不启动训练）
"""

import os
import sys
import time
import glob
import subprocess
import signal
import pandas as pd
from datetime import datetime

# ==================== 训练配置 ====================
# 消融实验队列：按顺序执行，每组完成后自动启动下一组
ABLATION_QUEUE = [
    {
        "name": "Ablation +CBAM",
        "script": "train_ablation_cbam.py",
        "csv": "runs/detect/runs/detect/ablation-cbam/results.csv",
        "checkpoint": "runs/detect/runs/detect/ablation-cbam/weights/last.pt",
        "total_epochs": 150,
        "iou_type": "CIoU",
    },
    {
        "name": "Ablation +CBAM +WeightedFuse",
        "script": "train_ablation_cbam_wf.py",
        "csv": "runs/detect/runs/detect/ablation-cbam-wf/results.csv",
        "checkpoint": "runs/detect/runs/detect/ablation-cbam-wf/weights/last.pt",
        "total_epochs": 150,
        "iou_type": "CIoU",
    },
]

# 已完成的实验（CSV路径）
COMPLETED_EXPERIMENTS = {
    "Baseline": "runs/detect/runs/train/yolov8_baseline_150ep/results.csv",
    "IYUAV-Det (Full)": "runs/detect/runs/train/iyuav-det-fixed/results.csv",
}

# 监控配置
CHECK_INTERVAL = 120  # 监控检查间隔（秒）
STALL_THRESHOLD = 600  # 无进度超过此秒数则判定中断（秒）
LOG_FILE = "training_monitor.log"

# 当前正在训练的进程
current_process = None


def log(msg):
    """同时输出到终端和日志文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_csv_latest(csv_path):
    """读取CSV最后一行"""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df.iloc[-1]
    except Exception:
        return None


def get_current_epoch(csv_path):
    """获取当前已训练的epoch数"""
    latest = read_csv_latest(csv_path)
    if latest is None:
        return 0
    return int(latest["epoch"])


def is_training_complete(csv_path, total_epochs):
    """判断训练是否完成"""
    return get_current_epoch(csv_path) >= total_epochs


def start_training(exp):
    """启动一组训练（支持断点续训）"""
    checkpoint = exp["checkpoint"]
    csv_path = exp["csv"]

    # 检查是否有断点可续
    if os.path.exists(checkpoint):
        current_ep = get_current_epoch(csv_path)
        log(f"发现断点 {checkpoint} (epoch {current_ep})，从断点续训")
        cmd = [
            sys.executable, "-c",
            f"from ultralytics import YOLO; "
            f"model = YOLO('{checkpoint}'); "
            f"model.train(resume=True)"
        ]
    else:
        log(f"未发现断点，从头训练: {exp['script']}")
        cmd = [sys.executable, exp["script"]]

    log(f"启动命令: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd="/mnt/workspace/YOLO")


def wait_for_training(exp, process):
    """监控训练进程，检测中断并自动续训"""
    csv_path = exp["csv"]
    total_epochs = exp["total_epochs"]
    last_epoch = 0
    last_progress_time = time.time()

    while True:
        # 检查进程是否还在运行
        retcode = process.poll()

        current_ep = get_current_epoch(csv_path)

        # 检查是否有进度
        if current_ep > last_epoch:
            last_epoch = current_ep
            last_progress_time = time.time()

        # 训练完成
        if is_training_complete(csv_path, total_epochs):
            log(f"✅ {exp['name']} 训练完成！最终 epoch={current_ep}")
            return True

        # 进程已退出
        if retcode is not None:
            if retcode == 0:
                # 正常退出，再检查一次是否完成
                if is_training_complete(csv_path, total_epochs):
                    log(f"✅ {exp['name']} 训练完成！")
                    return True
                else:
                    log(f"⚠️ 进程正常退出但未到目标epoch (epoch={current_ep}/{total_epochs})，尝试续训")
            else:
                log(f"❌ {exp['name']} 进程异常退出 (code={retcode})，epoch={current_ep}，尝试续训")

            # 自动续训
            time.sleep(5)
            if os.path.exists(exp["checkpoint"]):
                process = start_training(exp)
                last_progress_time = time.time()
                continue
            else:
                log(f"❌ 无断点文件，无法续训，跳过此实验")
                return False

        # 检查是否卡住（无进度超时）
        stall_time = time.time() - last_progress_time
        if stall_time > STALL_THRESHOLD:
            log(f"⚠️ {exp['name']} 超过 {STALL_THRESHOLD}s 无进度，可能卡死，杀死进程并续训")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            time.sleep(5)
            if os.path.exists(exp["checkpoint"]):
                process = start_training(exp)
                last_progress_time = time.time()
                continue
            else:
                log(f"❌ 无断点文件，无法续训")
                return False

        # 打印状态
        stall_warn = f" (⚠️ 无进度 {int(stall_time)}s)" if stall_time > 120 else ""
        log(f"📊 {exp['name']}: epoch {current_ep}/{total_epochs} ({current_ep/total_epochs*100:.1f}%){stall_warn}")

        time.sleep(CHECK_INTERVAL)


def print_summary():
    """打印所有实验的最终汇总"""
    log("\n" + "=" * 80)
    log("📊 消融实验结果汇总")
    log("=" * 80)
    log(f"{'实验':<30} {'mAP@50':>10} {'mAP@50-95':>12} {'Recall':>10} {'Epoch':>8}")
    log("-" * 80)

    all_exps = {}
    all_exps.update(COMPLETED_EXPERIMENTS)
    for exp in ABLATION_QUEUE:
        all_exps[exp["name"]] = exp["csv"]

    for name, csv_path in all_exps.items():
        latest = read_csv_latest(csv_path)
        if latest is not None:
            mAP50 = latest.get("metrics/mAP50(B)", 0)
            mAP5095 = latest.get("metrics/mAP50-95(B)", 0)
            recall = latest.get("metrics/recall(B)", 0)
            epoch = int(latest.get("epoch", 0))
            log(f"{name:<30} {mAP50:>10.4f} {mAP5095:>12.4f} {recall:>10.4f} {epoch:>8}")
        else:
            log(f"{name:<30} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>8}")

    log("=" * 80)


def monitor_only():
    """仅监控模式，不启动训练"""
    log("📋 仅监控模式（不启动训练）")

    all_exps = {}
    all_exps.update(COMPLETED_EXPERIMENTS)
    for exp in ABLATION_QUEUE:
        all_exps[exp["name"]] = exp["csv"]

    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log(f"\n{'='*80}")
            log(f"📊 训练监控 - {current_time}")
            log(f"{'='*80}")

            for name, csv_path in all_exps.items():
                latest = read_csv_latest(csv_path)
                if latest is None:
                    log(f"  {name}: 等待数据...")
                    continue
                epoch = int(latest.get("epoch", 0))
                mAP50 = latest.get("metrics/mAP50(B)", 0)
                mAP5095 = latest.get("metrics/mAP50-95(B)", 0)
                recall = latest.get("metrics/recall(B)", 0)
                log(f"  {name}: epoch={epoch}, mAP50={mAP50:.4f}, mAP50-95={mAP5095:.4f}, R={recall:.4f}")

            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        log("监控已停止")


def main():
    # 处理命令行参数
    if "--monitor" in sys.argv:
        monitor_only()
        return

    log("=" * 80)
    log("🚀 YOLO消融实验训练启动")
    log(f"📋 实验队列: {len(ABLATION_QUEUE)} 组")
    for i, exp in enumerate(ABLATION_QUEUE, 1):
        log(f"   {i}. {exp['name']} (epochs={exp['total_epochs']}, iou={exp['iou_type']})")
    log(f"📋 已完成: {len(COMPLETED_EXPERIMENTS)} 组")
    log(f"📋 日志文件: {LOG_FILE}")
    log("=" * 80)

    # 注册信号处理，优雅退出
    def handle_signal(signum, frame):
        log("收到终止信号，等待当前进程结束...")
        if current_process and current_process.poll() is None:
            current_process.terminate()
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # 逐组执行消融实验
    for i, exp in enumerate(ABLATION_QUEUE, 1):
        csv_path = exp["csv"]
        total_epochs = exp["total_epochs"]

        # 跳过已完成的实验
        if is_training_complete(csv_path, total_epochs):
            log(f"⏭️  [{i}/{len(ABLATION_QUEUE)}] {exp['name']} 已完成，跳过")
            continue

        log(f"\n{'='*80}")
        log(f"🔬 [{i}/{len(ABLATION_QUEUE)}] 开始: {exp['name']}")
        log(f"{'='*80}")

        global current_process
        current_process = start_training(exp)
        success = wait_for_training(exp, current_process)

        if success:
            log(f"✅ {exp['name']} 完成！")
        else:
            log(f"❌ {exp['name']} 失败，继续下一组")

    # 打印最终汇总
    print_summary()
    log("🎉 全部消融实验结束！")


if __name__ == "__main__":
    main()
