"""
Task1_recognize (Recognition-only on detection JSON)

Input:
- Images directory
- Detection JSONs from Task1_detection (task2.output.text_blocks with polygons)

Output:
- JSONs that keep polygons and add recognized text + scores
"""

import json
import logging
import os

import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR

import config


try:
    TASK1_RECOGNIZE_CONFIG = config.return_task1_recognize_config()
except AttributeError:
    TASK1_RECOGNIZE_CONFIG = config.returnTestTask2_2_Config()

# Legacy alias used by app.py.
Task2_Config = TASK1_RECOGNIZE_CONFIG

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
PAD_EXPAND_PX = 4

logging.getLogger("ppocr").setLevel(logging.WARNING)


def init_model():
    """Load PaddleOCR recognizer only (no YOLO detection here)."""
    print("--- Initialize PaddleOCR recognizer-only ---")
    try:
        if paddle.is_compiled_with_cuda():
            paddle.device.set_device("gpu")
            print(" -> [OK] Paddle GPU.")
        else:
            paddle.device.set_device("cpu")
            print(" -> [WARN] Paddle CPU.")
    except Exception:
        pass

    recognizer = PaddleOCR(
        lang="en",
        use_angle_cls=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    print(" -> [OK] Recognizer initialized.")
    return {"recognizer": recognizer}


def read_image_windows(path):
    if not os.path.exists(path):
        print(f"  [ERR] File not found: {path}")
        return None

    if os.path.getsize(path) == 0:
        print(f"  [ERR] Empty file: {path}")
        return None

    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img_bgr = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        print(f"  [ERR] Failed to read image: {exc}")
        return None


def prep_crop_for_rec(crop_rgb: np.ndarray) -> np.ndarray:
    """Light enhancement for OCR on small text crops."""
    try:
        h, w = crop_rgb.shape[:2]
        if h == 0 or w == 0:
            return crop_rgb

        long_side = max(h, w)
        target = 512
        scale = min(2.5, max(1.5, target / long_side))

        if scale > 1.0:
            crop_rgb = cv2.resize(
                crop_rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        blur = cv2.GaussianBlur(crop_rgb, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(crop_rgb, 1.3, blur, -0.3, 0)
        return sharp
    except Exception:
        return crop_rgb


def expand_polygon(pts: np.ndarray, pad: float, max_w: int, max_h: int) -> np.ndarray:
    """Expand polygon radially from centroid to include OCR context."""
    try:
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] < 4:
            return pts

        cx, cy = pts.mean(axis=0)
        expanded = []
        for x, y in pts:
            dx, dy = x - cx, y - cy
            dist = np.hypot(dx, dy)

            if dist == 0:
                nx, ny = x, y
            else:
                scale = (dist + pad) / dist
                nx = cx + dx * scale
                ny = cy + dy * scale

            nx = min(max(nx, 0), max_w - 1)
            ny = min(max(ny, 0), max_h - 1)
            expanded.append([nx, ny])

        return np.asarray(expanded, dtype=np.int32)
    except Exception:
        return np.asarray(pts, dtype=np.int32)


def mask_crop_from_polygon(img: np.ndarray, poly: np.ndarray) -> np.ndarray | None:
    """Crop polygon area and fill outside area with white."""
    try:
        poly = np.asarray(poly, dtype=np.int32)
        if poly.shape[0] < 4:
            return None

        x_min = max(int(np.min(poly[:, 0])), 0)
        x_max = min(int(np.max(poly[:, 0])), img.shape[1] - 1)
        y_min = max(int(np.min(poly[:, 1])), 0)
        y_max = min(int(np.max(poly[:, 1])), img.shape[0] - 1)

        if x_max <= x_min or y_max <= y_min:
            return None

        crop = img[y_min:y_max, x_min:x_max]
        shifted = poly - np.array([x_min, y_min])

        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shifted], 255)

        bg = np.full_like(crop, 255)
        crop_masked = bg.copy()
        crop_masked[mask == 255] = crop[mask == 255]
        return crop_masked
    except Exception:
        return None


def _polygon_to_points(polygon):
    if isinstance(polygon, dict):
        return np.array(
            [
                [polygon["x0"], polygon["y0"]],
                [polygon["x1"], polygon["y1"]],
                [polygon["x2"], polygon["y2"]],
                [polygon["x3"], polygon["y3"]],
            ],
            dtype=np.float32,
        )

    if isinstance(polygon, list):
        if len(polygon) == 8:
            return np.array(
                [
                    [polygon[0], polygon[1]],
                    [polygon[2], polygon[3]],
                    [polygon[4], polygon[5]],
                    [polygon[6], polygon[7]],
                ],
                dtype=np.float32,
            )

        if len(polygon) == 4 and isinstance(polygon[0], (list, tuple)):
            return np.array(polygon, dtype=np.float32)

    return None


def _points_to_polygon_dict(points: np.ndarray):
    points = np.asarray(points, dtype=np.float32)
    return {
        "x0": int(round(points[0][0])),
        "y0": int(round(points[0][1])),
        "x1": int(round(points[1][0])),
        "y1": int(round(points[1][1])),
        "x2": int(round(points[2][0])),
        "y2": int(round(points[2][1])),
        "x3": int(round(points[3][0])),
        "y3": int(round(points[3][1])),
    }


def _extract_text_from_ocr_result(recognizer, crop_rgb: np.ndarray):
    rec_text = ""
    rec_score = 0.0

    try:
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        rec_res = recognizer.ocr(crop_bgr)
        if not rec_res:
            return rec_text, rec_score

        if isinstance(rec_res[0], dict):
            texts_all = []
            scores_all = []
            for entry in rec_res:
                texts_all.extend(entry.get("rec_texts") or [])
                scores_all.extend(entry.get("rec_scores") or [])
            if texts_all:
                rec_text = " ".join(texts_all).strip()
            if scores_all:
                rec_score = float(max(scores_all))
            return rec_text, rec_score

        if isinstance(rec_res[0], list):
            texts_all = []
            scores_all = []
            for cand in rec_res:
                if isinstance(cand, list) and len(cand) >= 2:
                    txt_part = cand[1]
                    if isinstance(txt_part, tuple) and len(txt_part) >= 2:
                        texts_all.append(txt_part[0])
                        scores_all.append(float(txt_part[1]))
                    elif isinstance(txt_part, list) and len(txt_part) >= 2:
                        texts_all.append(txt_part[0])
                        scores_all.append(float(txt_part[1]))
            if texts_all:
                rec_text = " ".join(texts_all).strip()
            if scores_all:
                rec_score = float(max(scores_all))
    except Exception:
        pass

    return rec_text, rec_score


def _load_detection_blocks(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    chart_type = "vertical bar"
    blocks = []

    if isinstance(data, dict):
        task2 = data.get("task2", {})
        chart_type = (
            task2.get("input", {})
            .get("task1_output", {})
            .get("chart_type", "vertical bar")
        )
        blocks = task2.get("output", {}).get("text_blocks", [])
    elif isinstance(data, list):
        blocks = data

    return chart_type, blocks


def _find_json_for_image(input_json_dir: str, image_filename: str):
    stem = os.path.splitext(image_filename)[0]
    candidate_1 = os.path.join(input_json_dir, f"{stem}.json")
    candidate_2 = os.path.join(input_json_dir, f"{image_filename}.json")

    if os.path.exists(candidate_1):
        return candidate_1
    if os.path.exists(candidate_2):
        return candidate_2
    return None


def process_single_image(ocr_model, img_path, detection_blocks):
    img = read_image_windows(img_path)
    if img is None:
        print("  -> [SKIP] Skip image due to read error.")
        return []

    recognizer = ocr_model["recognizer"]
    img_h, img_w = img.shape[:2]
    text_blocks = []

    for index, block in enumerate(detection_blocks):
        try:
            polygon = block.get("polygon")
            points = _polygon_to_points(polygon)
            if points is None:
                continue

            expanded_points = expand_polygon(points, PAD_EXPAND_PX, img_w, img_h)
            crop = mask_crop_from_polygon(img, expanded_points)
            if crop is None or crop.size == 0:
                continue

            crop = prep_crop_for_rec(crop)
            rec_text, rec_score = _extract_text_from_ocr_result(recognizer, crop)

            out_block = {
                "id": int(block.get("id", index)),
                "polygon": _points_to_polygon_dict(points),
                "text": rec_text,
                "score": round(float(block.get("score", 1.0)), 4),
                "rec_score": round(float(rec_score), 4),
            }
            text_blocks.append(out_block)

        except Exception as exc:
            print(f"  [WARN] Box {index} failed: {exc}")
            continue

    return text_blocks


def save_json(data, output_path, chart_type="vertical bar"):
    final_output = {
        "task2": {
            "input": {"task1_output": {"chart_type": chart_type}},
            "name": "Text Detection and Recognition",
            "output": {"text_blocks": data},
        }
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(final_output, file, ensure_ascii=False, indent=4)


def main():
    cfg = Task2_Config if isinstance(Task2_Config, dict) else TASK1_RECOGNIZE_CONFIG

    input_img_dir = cfg["input"]
    input_json_dir = cfg.get("input_json", cfg["output"])
    output_dir = cfg["output"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ocr = init_model()

    files = [
        f for f in os.listdir(input_img_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]
    total_files = len(files)

    print(f"\nFound {total_files} images.")
    print("-" * 50)

    for idx, filename in enumerate(files):
        img_full_path = os.path.join(input_img_dir, filename)
        json_in_path = _find_json_for_image(input_json_dir, filename)
        json_out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")

        print(f"[{idx + 1}/{total_files}] Processing: {filename} ... ", end="")

        if not json_in_path:
            print("-> Skip (missing detection JSON)")
            continue

        try:
            chart_type, detection_blocks = _load_detection_blocks(json_in_path)
            recognized_blocks = process_single_image(ocr, img_full_path, detection_blocks)
            save_json(recognized_blocks, json_out_path, chart_type=chart_type)
            print(f"-> Done ({len(recognized_blocks)} texts)")
        except Exception as exc:
            print(f"-> [ERR] {exc}")

    print("-" * 50)
    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
