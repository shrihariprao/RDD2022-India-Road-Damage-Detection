# Evaluation and Preliminary Results

## Evaluation Protocol

The project evaluates four object-detection architectures on the held-out India subset of RDD2022:

- YOLOv8n
- Faster R-CNN ResNet-50 FPN
- DETR ResNet-50
- Custom SSD-style MobileNetV2

The primary object-detection metrics are:

- mAP@50
- mAP@50:95

Additional metrics reported where available include:

- Precision
- Recall
- F1-score
- Inference time
- FPS

## Preliminary Results

| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 0.1752 | 0.0715 | 0.2182 | 0.2047 | ~0.2113 | — |
| Faster R-CNN | 0.2705 | 0.1050 | 0.3556 | 0.5192 | 0.4221 | 11.70 |
| DETR | 0.0100 | 0.0030 | — | — | — | 13.57 |
| SSD-style MobileNetV2 | 0.0029 | 0.0006 | — | — | — | 4.53 |

The complete machine-readable table is available in:

```text
preliminary_results.csv
