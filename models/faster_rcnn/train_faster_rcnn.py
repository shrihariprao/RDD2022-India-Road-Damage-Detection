"""
Faster R-CNN baseline for RDD2022 India road-damage detection.

Architecture:
    Faster R-CNN + ResNet-50 FPN

Classes:
    D00 - Longitudinal cracks
    D10 - Transverse cracks
    D20 - Alligator cracks
    D40 - Potholes

Preliminary experiment:
    Train images : 3266
    Validation   : 393
    Test         : 194
    Batch size   : 2
    Learning rate: 0.005
    Momentum     : 0.9
    Weight decay : 0.0005
    Epochs       : 3
    Optimizer    : SGD
    Scheduler    : StepLR(step_size=2, gamma=0.1)

The implementation uses torchvision's pretrained
Faster R-CNN ResNet-50 FPN model with a five-class
predictor: four damage classes + background.

This script is intended as a reproducibility/reference
implementation. Dataset paths should be adapted to the
local environment.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from PIL import Image


CLASS_NAMES = ["D00", "D10", "D20", "D40"]
NUM_CLASSES = len(CLASS_NAMES) + 1

BATCH_SIZE = 2
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
EPOCHS = 3

# Adjust these paths for the local environment.
DATA_ROOT = Path(
    "/content/drive/MyDrive/"
    "Road_Damage_Detection_Project/"
    "RDD2022_India_4Class_Master"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/"
    "Road_Damage_Detection_Project/"
    "results/FasterRCNN_preliminary"
)


class RoadDamageCocoDataset(Dataset):
    """
    COCO-format dataset for Faster R-CNN.

    Project category IDs:
        0 -> D00
        1 -> D10
        2 -> D20
        3 -> D40

    Torchvision detection models reserve label 0 for background,
    therefore labels are shifted to:
        1 -> D00
        2 -> D10
        3 -> D20
        4 -> D40
    """

    def __init__(self, image_dir, annotation_file):
        self.image_dir = Path(image_dir)
        self.coco = COCO(str(annotation_file))

        # Include every image represented in the COCO file,
        # including background-only images.
        self.image_ids = sorted(self.coco.imgToImgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")

        annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(annotation_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]

            if w <= 0 or h <= 0:
                continue

            category_id = int(ann["category_id"])

            if category_id not in range(len(CLASS_NAMES)):
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(category_id + 1)
            areas.append(float(ann["area"]))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64
        )

        areas = torch.as_tensor(
            areas,
            dtype=torch.float32
        )

        iscrowd = torch.as_tensor(
            iscrowd,
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(
                [image_id],
                dtype=torch.int64
            ),
            "area": areas,
            "iscrowd": iscrowd,
        }

        image = F.to_tensor(image)

        return image, target


def collate_fn(batch):
    """
    Detection models receive a list of images and a list of targets.
    """
    return tuple(zip(*batch))


def make_dataset(split):
    """
    Build a dataset from the canonical COCO split.
    """

    split_dir = DATA_ROOT / split
    annotation_file = split_dir / "_annotations.coco.json"

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {split_dir}"
        )

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found: {annotation_file}"
        )

    return RoadDamageCocoDataset(
        image_dir=split_dir,
        annotation_file=annotation_file,
    )


def build_model():
    """
    Create pretrained Faster R-CNN ResNet-50 FPN
    and replace the classifier for four damage classes.
    """

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )

    return model


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
):
    """
    Train Faster R-CNN for one epoch.
    """

    model.train()

    total_loss = 0.0

    for images, targets in data_loader:

        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
            }
            for target in targets
        ]

        loss_dict = model(
            images,
            targets
        )

        loss = sum(
            loss_dict.values()
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(
        len(data_loader),
        1
    )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    output_path,
):
    """
    Save a model checkpoint.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
        },
        output_path,
    )


def build_dataloaders():
    """
    Construct train, validation and test loaders.
    """

    train_dataset = make_dataset("train")
    valid_dataset = make_dataset("valid")
    test_dataset = make_dataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
    )


def main():
    """
    Reproducibility training entry point.

    This does not reproduce the already completed experiment
    automatically unless the dataset paths are available and
    training is explicitly executed.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader, valid_loader, test_loader = (
        build_dataloaders()
    )

    model = build_model()
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.1,
    )

    print(
        "Model: Faster R-CNN ResNet-50 FPN"
    )
    print(
        "Device:",
        device
    )
    print(
        "Classes:",
        CLASS_NAMES
    )
    print(
        "Train images:",
        len(train_loader.dataset)
    )
    print(
        "Validation images:",
        len(valid_loader.dataset)
    )
    print(
        "Test images:",
        len(test_loader.dataset)
    )

    for epoch in range(EPOCHS):

        loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- loss: {loss:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            output_path=(
                OUTPUT_DIR
                / f"fasterrcnn_epoch_{epoch + 1}.pth"
            ),
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
