# RDD2022 India Road Damage Detection

Deep learning based road damage detection using the RDD2022 India dataset.

## Models

- YOLOv8
- Faster R-CNN
- DETR
- SSD-style MobileNetV2

## Road Damage Classes

- D00 — Longitudinal cracks
- D10 — Transverse cracks
- D20 — Alligator cracks
- D40 — Potholes

## Dataset

The project uses the India subset of RDD2022.

The dataset is maintained separately from this repository and is not uploaded to GitHub.

## Project Structure

- dataset/ — Dataset documentation
- preprocessing/ — Dataset preprocessing and validation
- models/ — Model implementations
- evaluation/ — Evaluation scripts and metrics
- results/ — Experimental results
- notebooks/ — Colab notebooks
- docs/ — Methodology and experimental documentation

## Experimental Status

All four proposed detector families have been implemented with preliminary experiments:

1. YOLOv8
2. Faster R-CNN
3. DETR
4. SSD-style MobileNetV2

The reported experiments are preliminary and unoptimized.
