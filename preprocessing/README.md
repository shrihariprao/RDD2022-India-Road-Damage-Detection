# Dataset Preprocessing

## Overview

The project uses the India subset of the RDD2022 road-damage dataset.

The preprocessing pipeline prepares a common four-class dataset for comparison of four object-detection architectures:

- YOLOv8
- Faster R-CNN
- DETR
- SSD-style MobileNetV2

## Target Classes

The original road-damage labels used in this project are:

| ID | Class | Description |
|---:|---|---|
| 0 | D00 | Longitudinal cracks |
| 1 | D10 | Transverse cracks |
| 2 | D20 | Alligator cracks |
| 3 | D40 | Potholes |

The class IDs are kept consistent in the canonical COCO annotations.

## Dataset Organization

The canonical four-class dataset is organized as:

```text
RDD2022_India_4Class_Master/
├── train/
│   ├── images
│   └── _annotations.coco.json
├── valid/
│   ├── images
│   └── _annotations.coco.json
└── test/
    ├── images
    └── _annotations.coco.json
