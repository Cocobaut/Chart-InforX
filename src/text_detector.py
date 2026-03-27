import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import config
import torch

# =============================
# Configuration
# =============================
try:
    TASK1_DETECTION_CONFIG = config.return_task1_detection_config()
except AttributeError:
    TASK1_DETECTION_CONFIG = config.returnTestTask2_1_Config()

Task2_1Config = TASK1_DETECTION_CONFIG

OUTPUT_DIR = TASK1_DETECTION_CONFIG.get("output", TASK1_DETECTION_CONFIG.get("ouput"))

# Enable/disable visualization output
ENABLE_VISUALIZE = True

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


# =============================
# Image IO helpers (support unicode paths on Windows)
# =============================
def read_image_windows(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def save_image_windows(path, img):
    try:
        ext = os.path.splitext(path)[1]
        result, encoded_img = cv2.imencode(ext, img)
        if result:
            encoded_img.tofile(path)
            return True
    except Exception:
        pass
    return False


# =============================
# Convert YOLO OBB to JSON format
# =============================
def convert_obb_to_json_structure(filename, obb_results):
    text_blocks = []

    # Expand bounding box slightly
    expansion = 2

    if obb_results is not None:
        boxes = obb_results.xyxyxyxy.cpu().numpy()

        for idx, box in enumerate(boxes):
            # Compute box center
            center_x = np.mean(box[:, 0])
            center_y = np.mean(box[:, 1])

            expanded_points = []

            for point in box:
                x_old, y_old = point[0], point[1]

                dx = x_old - center_x
                dy = y_old - center_y

                distance = np.sqrt(dx**2 + dy**2)

                if distance > 1e-6:
                    scale_factor = (distance + expansion) / distance
                    x_new = center_x + dx * scale_factor
                    y_new = center_y + dy * scale_factor
                else:
                    x_new, y_new = x_old, y_old

                expanded_points.append([x_new, y_new])

            p = expanded_points
            polygon = {
                "x0": float(p[0][0]), "y0": float(p[0][1]),
                "x1": float(p[1][0]), "y1": float(p[1][1]),
                "x2": float(p[2][0]), "y2": float(p[2][1]),
                "x3": float(p[3][0]), "y3": float(p[3][1])
            }

            block = {"id": idx, "polygon": polygon}
            text_blocks.append(block)

    final_json = {
        "task1": {
            "input": {},
            "name": "Chart Classification",
            "output": {"chart_type": "vertical bar"}
        },
        "task2": {
            "input": {"task1_output": {"chart_type": "vertical bar"}},
            "name": "Text Detection and Recognition",
            "output": {"text_blocks": text_blocks}
        }
    }

    return final_json


# =============================
# Draw OBB polygons on image
# =============================
def visualize_obb(img_path, json_data, output_dir):
    img = read_image_windows(img_path)
    if img is None:
        return

    blocks = json_data["task2"]["output"]["text_blocks"]

    for block in blocks:
        poly = block["polygon"]

        pts = np.array([
            [poly["x0"], poly["y0"]],
            [poly["x1"], poly["y1"]],
            [poly["x2"], poly["y2"]],
            [poly["x3"], poly["y3"]]
        ], np.int32)

        pts = pts.reshape((-1, 1, 2))

        # Draw polygon
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)

    save_image_windows(save_path, img)


# =============================
# Main inference pipeline
# =============================
def main():
    cfg = Task2_1Config if isinstance(Task2_1Config, dict) else TASK1_DETECTION_CONFIG

    MODEL_PATH = cfg["weight"]
    INPUT_IMG_DIR = cfg["input"]
    OUTPUT_DIR = cfg.get("output", cfg.get("ouput"))

    OUTPUT_JSON_DIR = OUTPUT_DIR
    OUTPUT_VIS_DIR = os.path.join(OUTPUT_DIR, "visualized_images")

    if not os.path.exists(OUTPUT_JSON_DIR):
        os.makedirs(OUTPUT_JSON_DIR)

    if ENABLE_VISUALIZE and not os.path.exists(OUTPUT_VIS_DIR):
        os.makedirs(OUTPUT_VIS_DIR)

    print(f"Loading YOLO model from: {MODEL_PATH}")

    # Select GPU if available
    device = 0 if torch.cuda.is_available() else 'cpu'

    if device == 0:
        print(f"Inference device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("Inference device: CPU")

    model = YOLO(MODEL_PATH)

    files = [f for f in os.listdir(INPUT_IMG_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
    total = len(files)

    print(f"Found {total} images. Starting inference...")

    for idx, filename in enumerate(files):
        img_path = os.path.join(INPUT_IMG_DIR, filename)
        json_name = os.path.splitext(filename)[0] + ".json"
        save_json_path = os.path.join(OUTPUT_JSON_DIR, json_name)

        print(f"[{idx+1}/{total}] Processing: {filename}", end="\r")

        try:
            results = model.predict(
                img_path,
                save=False,
                conf=0.25,
                verbose=False,
                device=device,
                imgsz=1024
            )

            result = results[0]

            if result.obb is not None:
                json_data = convert_obb_to_json_structure(filename, result.obb)
            else:
                json_data = convert_obb_to_json_structure(filename, None)

            # Save JSON output
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

            # Optional visualization
            if ENABLE_VISUALIZE:
                visualize_obb(img_path, json_data, OUTPUT_VIS_DIR)

        except Exception as e:
            print(f"\n[ERROR] Failed processing {filename}: {e}")

    print("\nDone.")
    print(f"JSON results saved to: {OUTPUT_JSON_DIR}")

    if ENABLE_VISUALIZE:
        print(f"Visualization images saved to: {OUTPUT_VIS_DIR}")


if __name__ == "__main__":
    main()