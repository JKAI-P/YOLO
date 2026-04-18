from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    model.train (
        data="datasets/Det-Fly.v6i.yolov/data.yaml",  # dataset.yaml path
        epochs=50,  # number of epochs to train for
        patience=20,       # 20轮不提升就早停
        imgsz=640,  # train image sizeuu
        batch=8,  # total batch size for all GPUs
        project="runs/train",  # save to project/nam/e
        name="yolov26n",  # save to project/name
        workers=1,  # number of dataloader workers
        cache="ram",  # cache images for faster training
    )5