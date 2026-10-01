# Road Damage Detection from Images of Indian Roads Using Deep Learning Models

A comparative deep-learning study for detecting and localizing road damage in images from the Indian subset of RDD2022.

## Models

- YOLOv8
- Faster R-CNN
- DETR
- SSD-style MobileNetV2

## Road-Damage Classes

| Class | Description |
|---|---|
| D00 | Longitudinal cracks |
| D10 | Transverse cracks |
| D20 | Alligator cracks |
| D40 | Potholes |

## Dataset

This project uses the India subset of RDD2022 and a four-class detection protocol.

### Dataset split

| Split | Images | Images with targets |
|---|---:|---:|
| Train | 6,532 | 2,738 |
| Validation | 786 | 338 |
| Held-out Test | 388 | 147 |
| Total | 7,706 | 3,223 |

The full image dataset is not stored in this repository. Dataset metadata, configuration, and the canonical COCO annotation files are included.

## Preliminary Results

| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 0.1752 | 0.0715 | 0.2182 | 0.2047 | ~0.2113 | — |
| Faster R-CNN | 0.2705 | 0.1050 | 0.3556 | 0.5192 | 0.4221 | 11.70 |
| DETR | 0.0100 | 0.0030 | — | — | — | 13.57 |
| SSD-style MobileNetV2 | 0.0029 | 0.0006 | — | — | — | 4.53 |

These are preliminary experimental results, not optimized or state-of-the-art claims.

Precision, recall, F1, and inference-speed measurements were not obtained using completely identical procedures across all four implementations. Therefore, the values should be interpreted in the context of the detailed evaluation documentation rather than as a definitive ranking.

## Model Implementations

### YOLOv8

Implementation:

`models/yolov8/train_yolov8.py`

Preliminary configuration:

- YOLOv8n
- Pretrained initialization
- Image size: 640
- Batch size: 16
- Epochs: 5
- Workers: 2

### Faster R-CNN

Implementation:

`models/faster_rcnn/train_faster_rcnn.py`

Architecture:

- Faster R-CNN
- ResNet-50 FPN
- Pretrained weights

Preliminary configuration:

- Batch size: 2
- Learning rate: 0.005
- Momentum: 0.9
- Weight decay: 0.0005
- Epochs: 3
- SGD optimizer
- StepLR scheduler

### DETR

Architecture:

- DETR
- ResNet-50 backbone
- Pretrained `facebook/detr-resnet-50`

Preliminary configuration:

- Batch size: 2
- Learning rate: 5e-6
- Weight decay: 1e-4
- AdamW
- Epochs: 3
- Gradient clipping: 0.1

The DETR experiment uses a corrected COCO-to-DETR target preparation pipeline.

### SSD-style MobileNetV2

This project uses a custom SSD-style implementation with MobileNetV2 as the feature extractor. It is not claimed to be the official TensorFlow Model Garden SSD implementation.

Preliminary configuration:

- Input size: 320 × 320
- Batch size: 8
- Learning rate: 1e-4
- Adam optimizer
- Frozen backbone
- Epochs: 2
- Six anchors per spatial location
- Focal classification loss
- Smooth-L1/Huber box regression

## Repository Structure

```text
RDD2022-India-Road-Damage-Detection/
├── dataset/
├── docs/
├── evaluation/
├── models/
│   ├── yolov8/
│   ├── faster_rcnn/
│   ├── detr/
│   └── ssd_mobilenetv2/
├── notebooks/
├── preprocessing/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
