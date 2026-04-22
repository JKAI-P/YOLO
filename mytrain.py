from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    model.train(
        data="datasets/Det-Fly.v6i.yolov/data.yaml",  # dataset.yaml path
        epochs=1,  # 先跑1轮确认流程跑通
        patience=20,  # 20轮不提升就早停
        imgsz=640,  # train image size
        batch=-1,  # 提高batch size，充分利用GPU显存
        device="0",  # 使用 NVIDIA GPU 加速 (Linux服务器环境)
        project="runs/train",  # save to project/name
        name="yolov26n_test",  # save to project/name
        workers=4,  # Linux GPU服务器可以使用多进程数据加载

        cache=True,
        optimizer="AdamW",  # 使用AdamW优化器，收敛更稳定
        lr0=0.001,  # 初始学习率
        lrf=0.01,  # 最终学习率衰减系数
        weight_decay=0.0005,  # 权重衰减
        hsv_h=0.015,  # 色调增强
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 亮度增强
        degrees=30.0,  # 旋转角度
        translate=0.1,  # 平移增强
        scale=0.5,  # 缩放增强
        mosaic=1.0,  # Mosaic增强
        mixup=0.0,  # Mixup增强（小目标不建议用）
        copy_paste=0.0,  # Copy-paste增强
        val=True,  # 每个epoch后验证
        plots=True,  # 保存训练图表
        exist_ok=False,  # 如果为True会覆盖已有实验
        amp=False,  # 禁用AMP检查，避免路径错误
        verbose=True,  # 显示详细训练信息
    )