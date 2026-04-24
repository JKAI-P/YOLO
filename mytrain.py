from ultralytics import YOLO

if __name__ == "__main__":
    # Load IYUAV-Det model from custom YAML (with fixed CBAM + SIoU)
    model = YOLO("ultralytics/cfg/models/v8/yolov8-iyuav.yaml")

    # Train the model — params aligned with baseline for fair comparison
    model.train(
        data="datasets/Det-Fly.v6i.yolov/data.yaml",
        epochs=150,
        patience=50,
        imgsz=640,
        batch=64,
        device="0",
        project="runs/train",
        name="iyuav-det-fixed",
        workers=12,

        # Optimizer — same as baseline
        optimizer="SGD",
        lr0=0.02,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,

        # Loss — SIoU (paper design)
        iou_type="SIoU",
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Data augmentation — same as baseline
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,

        # Warmup — same as baseline
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Other — same as baseline
        cache=True,
        val=True,
        plots=True,
        exist_ok=False,
        amp=True,
        verbose=True,
        deterministic=True,
        seed=0,
    )
