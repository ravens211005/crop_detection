import gradio as gr
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from norfair import Detection, Tracker, draw_points
import sys
from pathlib import Path


# ============================================================
# CONSTANTS
# ============================================================

REAL_PLANT_HEIGHT_CM = 40
FOCAL_LENGTH_PIX = 1220
EPS = 1e-6


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[INFO] Using device: {device}")


# ============================================================
# LOAD YOLO
# ============================================================

# ============================================================
# USER CONFIGURATION
# ============================================================
# Fill in these paths before running the application.
YOLO_MODEL = r"./best.pt"
MIDAS_MODEL = r"./dpt_large-midas-2f21e586.pt"
MIDAS_CODE_PATH = r"./midas"

sys.path.append(MIDAS_CODE_PATH)



if not Path(YOLO_MODEL).exists():
    raise FileNotFoundError(
        f"YOLO model not found:\\n{YOLO_MODEL}"
    )

print("[INFO] Loading YOLO...")

yolo_model = YOLO(YOLO_MODEL)

print("[INFO] YOLO loaded successfully")


# ============================================================
# LOAD MiDaS
# ============================================================


if not Path(MIDAS_MODEL).exists():
    raise FileNotFoundError(
        f"MiDaS model not found:\n{MIDAS_MODEL}"
    )

print("[INFO] Loading MiDaS...")

midas = DPTDepthModel(
    MIDAS_MODEL,
    backbone="vitl16_384",
    non_negative=True
).to(device).eval()

print("[INFO] MiDaS loaded successfully")


# ============================================================
# MiDaS TRANSFORM
# ============================================================

transform = Compose([
    Resize(384, 384),
    NormalizeImage(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    PrepareForNet()
])


# ============================================================
# NORFAIR TRACKER
# ============================================================

tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=30,
    initialization_delay=0,
    hit_counter_max=1,
    detection_threshold=0.0
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def process_image(image):

    if image is None:
        return None

    # --------------------------------------------------------
    # Convert PIL -> NumPy -> BGR
    # --------------------------------------------------------

    frame = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    height, width = frame.shape[:2]

    print("\n====================================")
    print("Processing image")
    print(f"Image size: {width} x {height}")
    print("====================================")


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = yolo_model(frame, verbose=False)

    boxes = results[0].boxes

    norfair_detections = []
    bbox_info = []

    for box in boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        conf = float(box.conf[0])

        if conf < 0.25:
            continue

        xc = int((x1 + x2) / 2)
        yc = int((y1 + y2) / 2)

        cls_id = int(box.cls[0])

        class_name = yolo_model.model.names.get(
            cls_id,
            "unknown"
        )

        # Norfair detection
        norfair_detections.append(
            Detection(
                points=np.array(
                    [[xc, yc]],
                    dtype=np.float32
                ),
                scores=np.array([conf])
            )
        )

        bbox_info.append({
            "box": (x1, y1, x2, y2),
            "center": (xc, yc),
            "class_name": class_name,
            "confidence": conf
        })

        print(
            f"[YOLO] {class_name} "
            f"confidence={conf:.3f} "
            f"box=({x1},{y1},{x2},{y2})"
        )


    print(
        f"[YOLO] Total detections: "
        f"{len(bbox_info)}"
    )


    # ========================================================
    # NORFAIR TRACKING
    # ========================================================

    tracked_objects = tracker.update(
        norfair_detections
    )

    print(
        f"[NORFAIR] Tracked objects: "
        f"{len(tracked_objects)}"
    )


    # ========================================================
    # MiDaS DEPTH
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    sample = transform({
        "image": rgb,
        "mask": np.ones(
            rgb.shape[:2],
            dtype=np.uint8
        )
    })

    input_batch = torch.from_numpy(
        sample["image"]
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        depth = midas(input_batch)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()


    print(
        f"[MiDaS] Depth range: "
        f"{depth.min():.4f} -> {depth.max():.4f}"
    )


    # ========================================================
    # DISTANCE CALCULATION
    # ========================================================

    if len(tracked_objects) == 0:

        print("[INFO] No tracked objects")

    elif len(bbox_info) == 0:

        print(
            "[WARNING] Tracker returned objects "
            "but there are no YOLO bounding boxes."
        )

    else:

        for obj in tracked_objects:

            # ----------------------------------------------
            # Norfair center
            # ----------------------------------------------

            x, y = obj.estimate[0].astype(int)

            # Keep coordinates inside image
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)


            # ----------------------------------------------
            # MiDaS value
            # ----------------------------------------------

            midas_value = float(
                depth[y, x]
            )

            print(
                f"[MiDaS] ID={obj.id} "
                f"x={x}, y={y}, "
                f"raw_depth={midas_value:.4f}"
            )


            # ----------------------------------------------
            # Find closest YOLO bounding box
            # ----------------------------------------------

            distances = []

            for info in bbox_info:

                center = np.array(
                    info["center"]
                )

                point = np.array(
                    [x, y]
                )

                distances.append(
                    np.linalg.norm(
                        point - center
                    )
                )

            best_idx = int(
                np.argmin(distances)
            )

            x1, y1, x2, y2 = bbox_info[
                best_idx
            ]["box"]

            class_name = bbox_info[
                best_idx
            ]["class_name"]


            # ----------------------------------------------
            # YOLO pixel height
            # ----------------------------------------------

            h_pixels = max(
                1,
                y2 - y1
            )


            # ----------------------------------------------
            # PINHOLE DISTANCE
            #
            # Z = f * H / h
            # ----------------------------------------------

            Z_pinhole = (
                FOCAL_LENGTH_PIX
                * REAL_PLANT_HEIGHT_CM
            ) / (
                h_pixels + EPS
            )


            # ----------------------------------------------
            # IMPORTANT:
            #
            # This is the actual distance produced
            # by the pinhole model.
            #
            # MiDaS is NOT used to calculate the
            # metric distance here.
            # ----------------------------------------------

            Z_real = Z_pinhole


            # ----------------------------------------------
            # DEBUG OUTPUT
            # ----------------------------------------------

            print(
                f"[RESULT] "
                f"ID={obj.id} | "
                f"{class_name} | "
                f"X={x} | "
                f"Y={y} | "
                f"Height={h_pixels}px | "
                f"MiDaS={midas_value:.4f} | "
                f"Distance={Z_real:.2f} cm"
            )


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # LABEL 1
            # =================================================

            label1 = (
                f"{class_name} "
                f"ID:{obj.id} "
                f"MiDaS:{midas_value:.2f}"
            )

            cv2.putText(
                frame,
                label1,
                (x1, max(20, y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


            # =================================================
            # LABEL 2
            # =================================================

            label2 = (
                f"X:{x} "
                f"Y:{y} "
                f"Z:{Z_real:.1f}cm"
            )

            cv2.putText(
                frame,
                label2,
                (x1, max(40, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )


    # ========================================================
    # DRAW NORFAIR POINTS
    # ========================================================

    if len(tracked_objects) > 0:

        draw_points(
            frame,
            tracked_objects,
            color=(0, 0, 255),
            radius=5
        )


    # ========================================================
    # BGR -> RGB
    # ========================================================

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# GRADIO UI
# ============================================================

demo = gr.Interface(
    fn=process_image,

    inputs=gr.Image(
        type="pil",
        label="Upload Crop Image"
    ),

    outputs=gr.Image(
        type="pil",
        label="Tracking + Real Distance"
    ),

    title="YOLO + MiDaS Real Distance",

    description=(
        "YOLO detects crops, Norfair tracks them, "
        "and MiDaS provides relative depth. "
        "Metric distance is calculated using the "
        "pinhole camera model."
    )
)


# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    share=True
)