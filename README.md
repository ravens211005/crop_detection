# Crop Detection, Tracking and Depth Estimation

A computer-vision project for **crop detection, object tracking, and relative depth estimation** using YOLO11n, ByteTrack/Norfair, MiDaS, OpenCV, and Gradio.

## Project Components

- `train.py` — trains a custom YOLO11n crop detector.
- `app_bytetrack.py` — YOLO11n + ByteTrack + MiDaS + Gradio.
- `app_norfair.py` — YOLO11n + Norfair + MiDaS + Gradio.

## Pipeline

```text
Dataset → YOLO11n Training → best.pt
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
         ByteTrack               Norfair
              └──────────┬──────────┘
                         ↓
                       MiDaS
                         ↓
                  Relative Depth
                         ↓
                  OpenCV + Gradio
```

## Repository Structure

```text
crop-detection/
├── train.py
├── app_bytetrack.py
├── app_norfair.py
├── data.yaml
├── requirements.txt
├── README.md
├── .gitignore
├── ByteTrack/
├── midas/
└── models/
```

Do not commit personal paths, credentials, virtual environments, or large model files unless required.

## Dataset

The dataset uses standard YOLO format:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each image should have a matching `.txt` label file.

Example:

```text
class_id center_x center_y width height
```

The project uses three crop classes:

```text
0 → beetroot
1 → radish
2 → potato
```

Example `data.yaml`:

```yaml
path: .
train: images/train
val: images/val

nc: 3
names: ["beetroot", "radish", "potato"]
```

## Training

`train.py` fine-tunes YOLO11n on the custom dataset.

Set the dataset configuration path before training.

### Windows PowerShell

```powershell
$env:DATA_YAML="YOUR_DATASET_LOCATION/data.yaml"
python train.py
```

The resulting `best.pt` is used by the detection applications.

## ByteTrack Application

`app_bytetrack.py` performs:

```text
YOLO11n → Detection → ByteTrack → Track ID → MiDaS → X,Y,Z → Gradio
```

YOLO performs detection; ByteTrack maintains object identities across frames/images.

The ByteTrack implementation is based on the official FoundationVision repository:

**https://github.com/FoundationVision/ByteTrack**

## Norfair Application

`app_norfair.py` performs:

```text
YOLO11n → Bounding Box Center → Norfair → Track ID → MiDaS → X,Y,Z → Gradio
```

Norfair performs tracking; it does not perform crop detection.

## X, Y and Z

- **X** — horizontal center coordinate in image pixels.
- **Y** — vertical center coordinate in image pixels.
- **Z** — MiDaS relative depth value.

Example:

```text
X:320 Y:240 Z:12.34
```

**Important:** MiDaS depth is relative/model-dependent in this implementation. It should not be interpreted directly as centimetres or metres without calibration.

## Model Paths

Personal Windows paths are not included in the repository.

Set model locations using environment variables:

```text
YOLO_WEIGHTS
MIDAS_WEIGHTS
```

Example:

```powershell
$env:YOLO_WEIGHTS="YOUR_YOLO_WEIGHTS_LOCATION/best.pt"
$env:MIDAS_WEIGHTS="YOUR_MIDAS_WEIGHTS_LOCATION/dpt_large-midas-2f21e586.pt"
```

Then run:

```powershell
python app_bytetrack.py
```

or:

```powershell
python app_norfair.py
```

The MiDaS source code and ByteTrack/YOLOX source code must also be available at the locations expected by the applications.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

## Model Files

Required model files:

```text
best.pt
dpt_large-midas-2f21e586.pt
```

Large model weights should normally be stored outside GitHub or managed using Git LFS.

## GitHub Hygiene

Do not commit:

```text
__pycache__/
.venv/
venv/
runs/
*.pt
.env
```

Also remove personal Windows/OneDrive paths before publishing.

## Acknowledgements

This project uses the ByteTrack multi-object tracking implementation based on the work of Yifu Zhang and the FoundationVision research team.

Official repository:

**https://github.com/FoundationVision/ByteTrack**

Please refer to the original repository for source-code attribution, licensing, and documentation.
