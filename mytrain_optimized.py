from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

    # Train the model with optimized settings for maximum performance
    model.train(
        data="datasets/Det-Fly.v6i.yolov/data.yaml",  # dataset.yaml path
        epochs=1,  # 先测试1轮验证性能
        patience=30,  # 增加早停轮数，避免过早停止
        imgsz=640,  # train image size
        batch=32,  # 显式设置更大的batch size，充分利用GPU显存
        device="0",  # 使用 NVIDIA GPU 加速 (Linux服务器环境)
        project="runs/train",  # save to project/name
        name="yolov26n_optimized_test",  # save to project/name
        workers=8,  # 增加workers数量，充分利用多核CPU

        cache=True,  # 缓存数据集到内存，加速训练
        optimizer="AdamW",  # 使用AdamW优化器，收敛更稳定
        lr0=0.01,  # 提高初始学习率，加速初期收敛
        lrf=0.01,  # 最终学习率衰减系数
        weight_decay=0.0005,  # 权重衰减
        
        # 数据增强策略 - 适度增强，避免过度影响小目标
        hsv_h=0.015,  # 色调增强
        hsv_s=0.7,  # 饱和度增强  
        hsv_v=0.4,  # 亮度增强
        degrees=10.0,  # 减少旋转角度，避免小目标丢失
        translate=0.1,  # 平移增强
        scale=0.5,  # 缩放增强
        mosaic=1.0,  # Mosaic增强
        mixup=0.1,  # 轻微Mixup增强（小目标也可以适当使用）
        copy_paste=0.0,  # Copy-paste增强（对小目标效果有限，暂时关闭）
        
        # 训练监控和保存
        val=True,  # 每个epoch后验证
        plots=True,  # 保存训练图表
        exist_ok=False,  # 如果为True会覆盖已有实验
        amp=False,  # 禁用AMP检查，避免路径错误（但会稍微降低训练速度）
        verbose=True,  # 显示详细训练信息
        
        # 学习率调度
        cos_lr=True,  # 使用余弦退火学习率调度
        warmup_epochs=3,  # 预热轮数
        warmup_momentum=0.8,  # 预热动量
        
        # 其他优化
        close_mosaic=10,  # 最后10轮关闭Mosaic增强，稳定收敛
    )