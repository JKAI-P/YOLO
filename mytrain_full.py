from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

    # Train the model with optimized settings for maximum performance - FULL TRAINING
    model.train(
        data="datasets/Det-Fly.v6i.yolov/data.yaml",  # dataset.yaml path
        epochs=100,  # 完整训练100轮
        patience=30,  # 早停轮数，避免过拟合
        imgsz=640,  # train image size
        batch=64,  # 优化后的batch size
        device="0",  # 使用 NVIDIA GPU 加速
        project="runs/train",  # save to project/name
        name="yolov26n_full_optimized",  # 实验名称
        workers=8,  # 优化后的workers数量

        cache=True,  # 缓存数据集到内存
        optimizer="AdamW",  # AdamW优化器
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率衰减系数
        weight_decay=0.0005,  # 权重衰减
        
        # 数据增强 - 平衡增强强度和小目标检测
        hsv_h=0.015,  
        hsv_s=0.7,    
        hsv_v=0.4,    
        degrees=10.0,  # 适度旋转
        translate=0.1, 
        scale=0.5,    
        mosaic=1.0,   
        mixup=0.1,    
        copy_paste=0.0,
        
        # 训练监控
        val=True,     
        plots=True,   
        exist_ok=False,
        amp=False,    # 禁用AMP检查
        verbose=True, 
        
        # 学习率调度
        cos_lr=True,  # 余弦退火
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        # 其他优化
        close_mosaic=10,  # 最后10轮关闭Mosaic
    )