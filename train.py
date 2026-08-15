"""
Train a custom YOLO11n crop-detection model.

Before running:
1. Install the requirements from requirements.txt.
2. Set the DATA_YAML environment variable to the location of your data.yaml.
3. Make sure yolo11n.pt can be downloaded by Ultralytics, or provide/use
   the appropriate pretrained YOLO11n weight file available to your setup.

Example (PowerShell):
    $env:DATA_YAML="YOUR_DATASET_LOCATION/data.yaml"
    python train.py

The dataset structure expected by data.yaml is documented in README.md.
"""

import os
from ultralytics import YOLO

# Set DATA_YAML in your environment to the location of your dataset YAML.
# Do not hard-code your personal computer path in this file.
DATA_YAML = os.getenv("DATA_YAML")

if not DATA_YAML:
    raise ValueError(
        "DATA_YAML is not set. Set it to the location of your data.yaml "
        "before running this program."
    )

# YOLO11n is the lightweight pretrained model used as the starting point.
model = YOLO("yolo11n.pt")

# Train the custom crop detector.
model.train(
    data=DATA_YAML,
    epochs=70,
    imgsz=640,
    batch=16,
    name="custom_yolo11",
)
