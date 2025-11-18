import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

model.train(
    data="D:/goldfish/dataset_pose/data.yaml",
    epochs=120,
    imgsz=640,
    batch=8,
    project="D:/goldfish",
    name="pose_training",
    exist_ok=True
)
