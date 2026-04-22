from ultralytics import YOLO

if __name__ == "__main__":
    # Load IYUAV-Det model from custom YAML
    model = YOLO("ultralytics/cfg/models/v8/yolov8-iyuav.yaml")

    # Train the model
    model.train(
        data="datasets/Det-Fly.v6i.yolov/data.yaml",  # dataset.yaml path
        epochs=300,  # total training epochs
        patience=20,  # early stopping patience
        imgsz=640,  # input image size
        batch=16,  # batch size
        device="0",  # CUDA GPU
        project="runs/train",  # save directory
        name="iyuav-det",  # experiment name
        workers=8,

        # Optimizer
        optimizer="SGD",
        lr0=0.01,  # initial learning rate
        lrf=0.01,  # final learning rate factor (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,

        # Loss - use SIoU for box loss
        iou_type="SIoU",
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # no rotation for UAV detection
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,  # light mixup
        copy_paste=0.0,

        # Warmup
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Other
        cache="disk",
        val=True,
        plots=True,
        exist_ok=False,
        amp=False,  # 禁用AMP检查，避免路径错误
        verbose=True,  # 显示详细训练信息
    )
