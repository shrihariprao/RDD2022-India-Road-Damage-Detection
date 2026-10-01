"""
DETR baseline for RDD2022 India road-damage detection.

Architecture:
    DETR with ResNet-50 backbone
    facebook/detr-resnet-50 pretrained initialization

Classes:
    D00 - Longitudinal cracks
    D10 - Transverse cracks
    D20 - Alligator cracks
    D40 - Potholes

Preliminary experiment:
    Batch size       : 2
    Learning rate    : 5e-6
    Weight decay     : 1e-4
    Optimizer        : AdamW
    Epochs           : 3
    Gradient clipping: 0.1

The implementation uses the corrected COCO-to-DETR target
preparation pipeline used in the preliminary experiment.

This script is a reproducibility/reference implementation.
It does not automatically reproduce the previously completed
training run unless training is explicitly executed.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
)


CLASS_NAMES = [
    "D00",
    "D10",
    "D20",
    "D40",
]

MODEL_NAME = "facebook/detr-resnet-50"

NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 2
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-4
EPOCHS = 3
GRADIENT_CLIP = 0.1

DATA_ROOT = Path(
    "/content/drive/MyDrive/"
    "Road_Damage_Detection_Project/"
    "RDD2022_India_4Class_Master"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/"
    "Road_Damage_Detection_Project/"
    "results/DETR_preliminary"
)


class DETRFixedRoadDamageDataset(Dataset):
    """
    COCO-format dataset prepared for Hugging Face DETR.

    The processor expects target annotations in COCO-style
    format with:
        image_id
        annotations

    Each annotation contains:
        category_id
        bbox
        area
        iscrowd
    """

    def __init__(
        self,
        image_dir,
        annotation_file,
        processor,
    ):
        self.image_dir = Path(image_dir)
        self.coco = COCO(str(annotation_file))
        self.processor = processor

        self.image_ids = sorted(
            self.coco.imgToImgs.keys()
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image_info = self.coco.loadImgs(
            image_id
        )[0]

        image_path = (
            self.image_dir
            / image_info["file_name"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        annotation_ids = self.coco.getAnnIds(
            imgIds=[image_id]
        )

        annotations = self.coco.loadAnns(
            annotation_ids
        )

        coco_annotations = []

        for ann in annotations:

            category_id = int(
                ann["category_id"]
            )

            if category_id not in range(
                NUM_CLASSES
            ):
                continue

            x, y, w, h = ann["bbox"]

            if w <= 0 or h <= 0:
                continue

            coco_annotations.append(
                {
                    "id": int(ann["id"]),
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "bbox": [
                        float(x),
                        float(y),
                        float(w),
                        float(h),
                    ],
                    "area": float(
                        ann["area"]
                    ),
                    "iscrowd": int(
                        ann.get(
                            "iscrowd",
                            0,
                        )
                    ),
                }
            )

        target = {
            "image_id": int(image_id),
            "annotations": coco_annotations,
        }

        encoded = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        # Remove batch dimension because the DataLoader
        # creates the batch later.
        pixel_values = encoded[
            "pixel_values"
        ].squeeze(0)

        pixel_mask = encoded[
            "pixel_mask"
        ].squeeze(0)

        labels = encoded[
            "labels"
        ][0]

        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """
    DETR requires padded pixel tensors and a list of
    variable-length target dictionaries.
    """

    pixel_values = torch.stack(
        [
            item["pixel_values"]
            for item in batch
        ]
    )

    pixel_masks = torch.stack(
        [
            item["pixel_mask"]
            for item in batch
        ]
    )

    labels = [
        item["labels"]
        for item in batch
    ]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_masks,
        "labels": labels,
    }


def build_processor():
    """
    Create the DETR image processor.
    """

    return DetrImageProcessor.from_pretrained(
        MODEL_NAME
    )


def build_model():
    """
    Create pretrained DETR with four road-damage classes.
    """

    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    return model


def make_dataset(
    split,
    processor,
):
    """
    Build one dataset split.
    """

    split_dir = DATA_ROOT / split

    annotation_file = (
        split_dir
        / "_annotations.coco.json"
    )

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{split_dir}"
        )

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: "
            f"{annotation_file}"
        )

    return DETRFixedRoadDamageDataset(
        image_dir=split_dir,
        annotation_file=annotation_file,
        processor=processor,
    )


def build_dataloaders(processor):
    """
    Construct train, validation and test loaders.
    """

    train_dataset = make_dataset(
        "train",
        processor,
    )

    valid_dataset = make_dataset(
        "valid",
        processor,
    )

    test_dataset = make_dataset(
        "test",
        processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
    )


def move_batch_to_device(
    batch,
    device,
):
    """
    Move image tensors and target dictionaries to device.
    """

    pixel_values = batch[
        "pixel_values"
    ].to(device)

    pixel_mask = batch[
        "pixel_mask"
    ].to(device)

    labels = [
        {
            key: value.to(device)
            for key, value in target.items()
        }
        for target in batch["labels"]
    ]

    return (
        pixel_values,
        pixel_mask,
        labels,
    )


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
):
    """
    Train DETR for one epoch.
    """

    model.train()

    total_loss = 0.0

    for batch in data_loader:

        (
            pixel_values,
            pixel_mask,
            labels,
        ) = move_batch_to_device(
            batch,
            device,
        )

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )

        loss = outputs.loss

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(
        len(data_loader),
        1,
    )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    output_dir,
):
    """
    Save a DETR checkpoint.
    """

    output_dir = Path(output_dir)

    checkpoint_dir = (
        output_dir
        / f"detr_resnet50_epoch{epoch}"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        checkpoint_dir
    )

    torch.save(
        optimizer.state_dict(),
        checkpoint_dir
        / "optimizer.pt",
    )


def main():
    """
    Reproducibility training entry point.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    processor = build_processor()

    (
        train_loader,
        valid_loader,
        test_loader,
    ) = build_dataloaders(
        processor
    )

    model = build_model()

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Model: DETR ResNet-50"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Classes:",
        CLASS_NAMES,
    )

    print(
        "Train images:",
        len(train_loader.dataset),
    )

    print(
        "Validation images:",
        len(valid_loader.dataset),
    )

    print(
        "Test images:",
        len(test_loader.dataset),
    )

    for epoch in range(EPOCHS):

        loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- loss: {loss:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            output_dir=OUTPUT_DIR,
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
