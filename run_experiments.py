"""
IYUAV-Det 实验脚本
包含：消融实验(4组) + 对比实验(YOLOv5s, YOLOv7, YOLOv8)
阿里云 A10 GPU, batch=16

用法：
    # 跑全部实验
    python run_experiments.py

    # 只跑消融实验
    python run_experiments.py --ablation

    # 只跑对比实验
    python run_experiments.py --comparison

    # 跑指定实验
    python run_experiments.py --exp baseline
    python run_experiments.py --exp cbam
    python run_experiments.py --exp cbam_wf
    python run_experiments.py --exp iyuav_det
    python run_experiments.py --exp yolov5s
    python run_experiments.py --exp yolov7
"""

import argparse
import time
from ultralytics import YOLO

DATA = "datasets/Det-Fly.v6i.yolov/data.yaml"
DEVICE = "0"
IMGSZ = 640
BATCH = 16
EPOCHS = 300
PATIENCE = 20

# 公共训练参数
COMMON_ARGS = dict(
    data=DATA,
欧    epochs=1,  # 改为1个epoch用于测试
    patience=20,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    workers=0,  # 改为0避免多进程问题，更适合调试
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.0,
    warmup_epochs=0,  # 测试时不需要warmup
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cache=False,  # 测试时不缓存图片
    val=True,
    plots=True,
    exist_ok=False,
    amp=False,  # 禁用AMP检查，避免FileNotFoundError
    verbose=True,  # 显示详细进度
)


def run_exp(name, model_path, extra_args=None):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"  开始实验: {name}")
    print(f"  模型: {model_path}")
    print(f"{'='*60}\n")

    model = YOLO(model_path)
    args = {**COMMON_ARGS, "project": "runs/train", "name": name}
    if extra_args:
        args.update(extra_args)

    start = time.time()
    model.train(**args)
    elapsed = time.time() - start
    print(f"\n实验 {name} 完成，耗时 {elapsed/3600:.1f} 小时\n")


# ==================== 实验定义 ====================

ABLATION_EXPS = {
    # 消融实验1: YOLOv8 Baseline（原始结构）
    "baseline": {
        "name": "1_baseline",
        "model": "ultralytics/cfg/models/v8/yolov8.yaml",
        "extra": {"iou_type": "CIoU"},  # 原始CIoU
    },
    # 消融实验2: +CBAM
    "cbam": {
        "name": "2_cbam",
        "model": "ultralytics/cfg/models/v8/yolov8-ablation-cbam.yaml",
        "extra": {"iou_type": "CIoU"},
    },
    # 消融实验3: +CBAM +WeightedFuse
    "cbam_wf": {
        "name": "3_cbam_wf",
        "model": "ultralytics/cfg/models/v8/yolov8-ablation-cbam-wf.yaml",
        "extra": {"iou_type": "CIoU"},
    },
    # 消融实验4: IYUAV-Det (全部改进)
    "iyuav_det": {
        "name": "4_iyuav_det",
        "model": "ultralytics/cfg/models/v8/yolov8-iyuav.yaml",
        "extra": {"iou_type": "SIoU"},  # SIoU是第4个改进
    },
}

COMPARISON_EXPS = {
    # 对比实验: YOLOv5s
    "yolov5s": {
        "name": "comp_yolov5s",
        "model": "yolov5s.pt",  # 使用预训练权重
    },
    # 对比实验: YOLOv7
    "yolov7": {
        "name": "comp_yolov7",
        "model": "yolov7.pt",  # 使用预训练权重
    },
}


def main():
    parser = argparse.ArgumentParser(description="IYUAV-Det Experiments")
    parser.add_argument("--ablation", action="store_true", help="只跑消融实验")
    parser.add_argument("--comparison", action="store_true", help="只跑对比实验")
    parser.add_argument("--exp", type=str, help="跑指定实验 (baseline/cbam/cbam_wf/iyuav_det/yolov5s/yolov7)")
    args = parser.parse_args()

    if args.exp:
        # 跑单个实验
        all_exps = {**ABLATION_EXPS, **COMPARISON_EXPS}
        if args.exp not in all_exps:
            print(f"未知实验: {args.exp}")
            print(f"可选: {list(all_exps.keys())}")
            return
        exp = all_exps[args.exp]
        run_exp(exp["name"], exp["model"], exp.get("extra"))
        return

    # 确定要跑哪些实验
    exps_to_run = []
    if args.ablation:
        exps_to_run = list(ABLATION_EXPS.values())
    elif args.comparison:
        exps_to_run = list(COMPARISON_EXPS.values())
    else:
        # 默认跑全部
        exps_to_run = list(ABLATION_EXPS.values()) + list(COMPARISON_EXPS.values())

    total = len(exps_to_run)
    print(f"\n共 {total} 个实验待运行")
    print(f"每个实验约 {EPOCHS} epochs, batch={BATCH}, imgsz={IMGSZ}")
    print()

    for i, exp in enumerate(exps_to_run, 1):
        print(f"\n>>> 进度: [{i}/{total}]")
        run_exp(exp["name"], exp["model"], exp.get("extra"))

    print("\n" + "=" * 60)
    print("  全部实验完成！")
    print("  结果保存在 runs/train/ 目录下")
    print("=" * 60)


if __name__ == "__main__":
    main()
