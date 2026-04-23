#!/usr/bin/env python3
"""
YOLOv8 vs IYUAV-Det 对比实验脚本（并行版）
训练200 epochs, 手动优化batch size以充分利用24GB显存

用法：
    python compare_v8_vs_iyuav.py
"""

import time
import multiprocessing as mp
from ultralytics import YOLO

# ==================== 配置参数 ====================
DATA = "datasets/Det-Fly.v6i.yolov/data.yaml"
DEVICE = "0"
IMGSZ = 640
BATCH = 64  # 手动设置大batch size以充分利用24GB显存
EPOCHS = 150  # 改为150 epochs

# 公共训练参数
COMMON_ARGS = dict(
    data=DATA,
    epochs=EPOCHS,
    patience=50,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    workers=12,       # 降低workers减少内存占用
    optimizer="SGD",
    lr0=0.02,         # 适当增加学习率以匹配大batch size
    lrf=0.05,         # 最终学习率也相应调整
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
    warmup_epochs=5,  # 增加warmup以稳定大batch训练
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cache=True,      # 缓存图片以节省内存
    val=True,
    plots=True,
    exist_ok=False,
    amp=True,         # AMP检查已通过，可以保持启用
    verbose=True,
)


def run_single_experiment(name, model_path, extra_args=None):
    """运行单个实验（用于多进程调用）"""
    print(f"\n{'='*80}")
    print(f"  🚀 开始实验: {name}")
    print(f"  📦 模型配置: {model_path}")
    print(f"  ⚙️  训练参数: epochs={EPOCHS}, batch={BATCH}, imgsz={IMGSZ}")
    print(f"{'='*80}\n")

    # 加载模型
    model = YOLO(model_path)
    
    # 准备训练参数
    args = {**COMMON_ARGS, "project": "runs/train", "name": name}
    if extra_args:
        args.update(extra_args)

    # 开始训练
    start_time = time.time()
    try:
        results = model.train(**args)
        elapsed_time = time.time() - start_time
        
        # 输出结果摘要
        print(f"\n{'='*80}")
        print(f"  ✅ 实验完成: {name}")
        print(f"  ⏱️  总耗时: {elapsed_time/60:.1f} 分钟 ({elapsed_time/3600:.2f} 小时)")
        print(f"  📊 平均每个epoch: {elapsed_time/EPOCHS:.1f} 秒")
        print(f"{'='*80}\n")
        
        return {"name": name, "success": True, "elapsed": elapsed_time}
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"  ❌ 实验失败: {name}")
        print(f"  ⏱️  已运行: {elapsed_time/60:.1f} 分钟")
        print(f"  💥 错误信息: {str(e)}")
        print(f"{'='*80}\n")
        
        return {"name": name, "success": False, "elapsed": elapsed_time, "error": str(e)}


def main():
    print("\n" + "="*80)
    print("  🎯 YOLOv8 基准版对比实验（150 epochs）")
    print("="*80)
    print(f"\n📋 实验配置:")
    print(f"  • 数据集: {DATA}")
    print(f"  • Epochs: {EPOCHS}")
    print(f"  • Batch Size: {BATCH} (手动优化)")
    print(f"  • Image Size: {IMGSZ}")
    print(f"  • Device: GPU {DEVICE}")
    print(f"  • Workers: 12 (充分利用GPU)")
    print(f"  • AMP: 启用（混合精度训练加速）")
    print(f"  • Cache: 禁用（避免内存溢出）")

    # 只训练YOLOv8基准版
    experiments = [
        ("yolov8_baseline_150ep", "ultralytics/cfg/models/v8/yolov8.yaml"),
    ]
    
    # 注释掉魔改版（之前已经跑完了）
    # experiments = [
    #     ("iyuav_det_200ep", "ultralytics/cfg/models/v8/yolov8-iyuav.yaml"),
    # ]

    print(f"\n📝 实验列表:")
    for i, (name, cfg) in enumerate(experiments, 1):
        print(f"  {i}. {name.replace('_', ' ').title()} (基准版)")

    print("\n" + "="*80)
    print("  ⚡ 启动训练...")
    print("  💡 提示: 自动batch size将根据GPU显存选择最佳值")
    print("="*80 + "\n")

    # 串行执行（只有一个实验）
    results = []
    for name, model_path in experiments:
        result = run_single_experiment(name, model_path)
        results.append(result)
        
        # 检查是否需要提前停止（50 epochs后性能变化不明显）
        if "yolov8" in name:
            print(f"\n🔍 检查50 epochs性能...")
            # 这里可以添加自动检查逻辑，但现在先手动监控

    # 输出最终总结
    print("\n" + "="*80)
    print("  📊 最终实验总结")
    print("="*80)
    for result in results:
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"  • {result['name']}: {status} (耗时: {result['elapsed']/60:.1f} 分钟)")
    print("="*80)


if __name__ == "__main__":
    main()