"""
YOLOv8 baseline for RDD2022 India road-damage detection.

Model:
    YOLOv8n (pretrained)

Classes:
    D00 - Longitudinal cracks
    D10 - Transverse cracks
    D20 - Alligator cracks
    D40 - Potholes

Experimental configuration used in the preliminary study:
    Image size : 640
    Batch size : 16
    Epochs     : 5
    Workers    : 2
    Device     : CUDA GPU when available

The dataset is expected in YOLO format with a data.yaml file.
"""

from pathlib import Path
import torch
from ultralytics import YOLO


# Update this path for your environment.
DATA_YAML = Path(
    "/content/drive/MyDrive/"
    "Road_Damage_Detection_Project/"
    "RDD2022_India_YOLO/data.yaml"
)

MODEL_NAME = "yolov8n.pt"

EPOCHS = 5
IMAGE_SIZE = 640
BATCH_SIZE = 16
WORKERS = 2


def train():
    """Train YOLOv8n on the RDD2022 India four-class dataset."""

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=device,
        pretrained=True,
    )

    return results


def evaluate(model_path: str):
    """Evaluate a trained YOLOv8 model on the held-out test split."""

    model = YOLO(model_path)

    return model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        device=0 if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    train()
