#!/usr/bin/env python3
"""
快速测试脚本 - 只跑1个epoch验证所有实验配置
"""

import time
from ultralytics import YOLO

DATA = "datasets/Det-Fly.v6i.yolov/data.yaml"
DEVICE = "0"
IMGSZ = 640
BATCH = 16

# 测试参数（1个epoch快速验证）
TEST_ARGS = dict(
    data=DATA,
    epochs=1,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    workers=0,
    cache=False,
    val=True,
    plots=True,
    exist_ok=False,
    amp=False,
    verbose=True,
)

def test_single_exp(name, model_path, extra_args=None):
    """测试单个实验配置"""
    print(f"\n{'='*60}")
    print(f"🧪 快速测试: {name}")
    print(f"📊 模型: {model_path}")
    print(f"⚡ 设备: CUDA:{DEVICE}, Batch: {BATCH}")
    print(f"📈 Epochs: 1 (仅测试)")
    print(f"{'='*60}\n")
    
    try:
        model = YOLO(model_path)
        args = {**TEST_ARGS, "project": "runs/test", "name": name}
        if extra_args:
            args.update(extra_args)
        
        start = time.time()
        model.train(**args)
        elapsed = time.time() - start
        print(f"\n✅ 测试 {name} 成功完成！耗时: {elapsed:.1f} 秒\n")
        return True
    except Exception as e:
        print(f"\n❌ 测试 {name} 失败: {str(e)}\n")
        return False

# 测试所有实验配置
TEST_EXPS = {
    # 消融实验测试
    "baseline_test": {
        "name": "baseline_test",
        "model": "ultralytics/cfg/models/v8/yolov8.yaml",
        "extra": {"iou_type": "CIoU"},
    },
    "cbam_test": {
        "name": "cbam_test", 
        "model": "ultralytics/cfg/models/v8/yolov8-ablation-cbam.yaml",
        "extra": {"iou_type": "CIoU"},
    },
    "iyuav_test": {
        "name": "iyuav_test",
        "model": "ultralytics/cfg/models/v8/yolov8-iyuav.yaml", 
        "extra": {"iou_type": "SIoU"},
    },
    # 对比实验测试
    "yolov5s_test": {
        "name": "yolov5s_test",
        "model": "yolov5s.pt",
    },
}

def main():
    """运行所有快速测试"""
    print("🚀 开始快速测试所有实验配置...")
    print("💡 这将帮助验证你的魔改代码和数据集配置是否正确\n")
    
    results = {}
    for exp_key, exp_config in TEST_EXPS.items():
        success = test_single_exp(exp_config["name"], exp_config["model"], exp_config.get("extra"))
        results[exp_key] = success
    
    # 总结结果
    print(f"\n{'='*60}")
    print("📋 快速测试总结:")
    successful = sum(results.values())
    total = len(results)
    print(f"✅ 成功: {successful}/{total}")
    print(f"❌ 失败: {total - successful}/{total}")
    
    if successful == total:
        print("\n🎉 所有测试通过！可以开始完整实验了！")
        print("📁 测试结果保存在: runs/test/")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复配置")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()