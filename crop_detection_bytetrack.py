import gradio as gr
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys
from pathlib import Path
from types import SimpleNamespace

# ============================================================
# USER CONFIGURATION
# ============================================================
# Fill in these paths before running the program.
#
# YOLO_MODEL:
#   Path to your trained YOLO .pt checkpoint.
#
# MIDAS_MODEL:
#   Path to the MiDaS DPT-Large model weights.
#
# MIDAS_CODE_PATH:
#   Folder containing the MiDaS Python package/files, including
#   dpt_depth.py and transforms.py.
#
# BYTETRACK_CODE_PATH:
#   Root folder of the local ByteTrack repository. It should
#   contain the "yolox" package.
#
# Example repository structure:
#   project/
#   ├── crop_detection_location.py
#   ├── ByteTrack/
#   │   └── yolox/
#   ├── midas/
#   │   ├── dpt_depth.py
#   │   └── transforms.py
#   └── models/
#       ├── best.pt
#       └── dpt_large-midas-2f21e586.pt
#
# You may replace these with absolute paths if the folders are
# stored elsewhere on your computer.

YOLO_MODEL = r"PATH_TO_YOLO_MODEL/best.pt"
MIDAS_MODEL = r"PATH_TO_MIDAS_MODEL/dpt_large-midas-2f21e586.pt"
MIDAS_CODE_PATH = r"./midas"
BYTETRACK_CODE_PATH = r"./ByteTrack"


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

def resolve_project_path(path_value):
    """Resolve relative paths from the directory containing this script."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


YOLO_MODEL = resolve_project_path(YOLO_MODEL)
MIDAS_MODEL = resolve_project_path(MIDAS_MODEL)
MIDAS_CODE_PATH = resolve_project_path(MIDAS_CODE_PATH)
BYTETRACK_CODE_PATH = resolve_project_path(BYTETRACK_CODE_PATH)

# Add local source trees to Python's import path.
sys.path.insert(0, str(MIDAS_CODE_PATH))
sys.path.insert(0, str(BYTETRACK_CODE_PATH))


from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker


# ============================================================
# VALIDATE USER PATHS
# ============================================================

for path_value, description in [
    (YOLO_MODEL, "YOLO model"),
    (MIDAS_MODEL, "MiDaS model"),
    (MIDAS_CODE_PATH, "MiDaS source directory"),
    (BYTETRACK_CODE_PATH, "ByteTrack source directory"),
]:
    if not path_value.exists():
        raise FileNotFoundError(
            f"{description} not found:\\n{path_value}\\n\\n"
            "Please update the corresponding path in the "
            "USER CONFIGURATION section."
        )


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO(str(YOLO_MODEL))
model.model.names = {0: "beetroot", 1: "radish", 2: "potato"}


# ============================================================
# LOAD MiDaS
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midas = DPTDepthModel(
    str(MIDAS_MODEL),
    backbone="vitl16_384",
    non_negative=True
).to(device).eval()

transform = Compose([
    Resize(384, 384),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet()
])

# === Initialize ByteTrack ===
args = SimpleNamespace(
    track_thresh=0.3,
    match_thresh=0.8,
    buffer_size=30,
    track_buffer=30,
    min_box_area=5,
    mot20=False
)
tracker = BYTETracker(args, frame_rate=30)

# === Main Function ===
def detect_and_track(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === Resize if too small ===
    min_height = 400
    min_width = 600
    h, w = image.shape[:2]
    scale_factor = max(min_width / w, min_height / h, 1.0)
    if scale_factor > 1.0:
        image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.resize(image_rgb, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_LINEAR)

    print("Image received", image.shape)
    results = model(image_rgb)[0]
    print("Number of YOLO boxes:", len(results.boxes))
    detections = []
    class_ids = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        w = x2 - x1
        h = y2 - y1
        detections.append([x1, y1, w, h, conf, cls])
        print(f"Detected class {cls} with confidence {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        class_ids.append(cls)

    dets_np = np.array(detections, dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)
    online_targets = tracker.update(dets_np, image.shape[:2], image.shape[:2])
    print("Number of tracks:", len(online_targets))

    # === MiDaS Depth Estimation ===
    sample = transform({"image": image_rgb, "mask": np.ones(image_rgb.shape[:2], dtype=np.uint8)})
    input_batch = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=image.shape[:2], mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

    for i, target in enumerate(online_targets):
        tlwh = target.tlwh
        tid = target.track_id
        x1, y1, w, h = map(int, tlwh)
        x2, y2 = x1 + w, y1 + h
        cx, cy = x1 + w // 2, y1 + h // 2

        z = float(depth[cy, cx]) if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1] else 0.0

        if not hasattr(target, "class_id"):
            if i < len(detections):
                target.class_id = int(detections[i][5])

        class_id = getattr(target, "class_id", 0)
        class_name = model.model.names[class_id] if class_id in model.model.names else "unknown"

        label = f"{class_name} ID:{tid}"
        coords = f"X:{cx}px Y:{cy}px Z:{z:.2f} units"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)  # Red dot at center
        cv2.putText(image, coords, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        print(f"Drawing: ID {tid} - {class_name} at ({cx}, {cy}, Z={z:.2f})")

    if len(online_targets) == 0:
        for i, det in enumerate(detections):
            x1, y1, w, h, conf, cls = det
            x1, y1, w, h = map(int, [x1, y1, w, h])
            cx, cy = x1 + w // 2, y1 + h // 2
            z = float(depth[cy, cx]) if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1] else 0.0
            label = f"{model.model.names[int(cls)]} Temp{i+1}"
            coords = f"X:{cx}px Y:{cy}px Z:{z:.2f}"

            cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)  # Red dot at center
            cv2.putText(image, coords, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    output_path = PROJECT_DIR / "debug_output.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"Saved output image to {output_path}")
    return image

# === Gradio Interface ===
gr.Interface(
    fn=detect_and_track,
    inputs=gr.Image(type="numpy", label="Upload Crop Image"),
    outputs=gr.Image(type="numpy", label="Crops with Tracking + Z Coordinate"),
    title="Crop Detection, Tracking and Depth",
    description="Detects and tracks crops using YOLO, ByteTrack, and MiDaS. Displays (X, Y, Z) coordinates."
).launch(share=True)
