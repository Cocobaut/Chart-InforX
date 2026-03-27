"""
Task1_recognize (Recognition-only on detection JSON)

Input:
- Images directory
- Detection JSONs from Task1_detection

Output:
- JSONs with recognized text and scores
"""

import json
import logging
import os

import cv2
import numpy as np
import torch
import paddle
from paddleocr import PaddleOCR

import config


# =============================
# Configuration
# =============================
try:
    TASK1_RECOGNIZE_CONFIG = config.return_task1_recognize_config()
except AttributeError:
    TASK1_RECOGNIZE_CONFIG = config.returnTestTask2_2_Config()

Task2_Config = TASK1_RECOGNIZE_CONFIG

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# Pixel padding for OCR context
PAD_EXPAND_PX = 4

logging.getLogger("ppocr").setLevel(logging.WARNING)


# =============================
# Initialize OCR model
# =============================
def init_model():
    print("Initializing PaddleOCR recognizer...")

    try:
        if paddle.is_compiled_with_cuda():
            paddle.device.set_device("gpu")
            print(" -> Using GPU")
        else:
            paddle.device.set_device("cpu")
            print(" -> Using CPU")
    except Exception:
        pass

    recognizer = PaddleOCR(
        lang="en",
        # use_angle_cls=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    print(" -> OCR model ready")
    return {"recognizer": recognizer}


# =============================
# Read image (unicode-safe)
# =============================
def read_image_windows(path):
    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        return None

    if os.path.getsize(path) == 0:
        print(f"  [ERROR] Empty file: {path}")
        return None

    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img_bgr = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    except Exception as exc:
        print(f"  [ERROR] Failed reading image: {exc}")
        return None


# =============================
# Improve crop for OCR
# =============================
def prep_crop_for_rec(crop_rgb: np.ndarray) -> np.ndarray:
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


# =============================
# Expand polygon for OCR context
# =============================
def expand_polygon(pts: np.ndarray, pad: float, max_w: int, max_h: int) -> np.ndarray:
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


# =============================
# Mask polygon crop
# =============================
def mask_crop_from_polygon(img: np.ndarray, poly: np.ndarray) -> np.ndarray | None:
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


# =============================
# Convert polygon formats
# =============================
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


# =============================
# Extract text from OCR output
# =============================
def _append_text_score(texts_all, scores_all, text_value, score_value):
    if text_value is not None:
        text_norm = str(text_value).strip()
        if text_norm:
            texts_all.append(text_norm)

    if score_value is not None:
        try:
            scores_all.append(float(score_value))
        except (TypeError, ValueError):
            pass


def _extract_from_legacy_ocr_format(obj, texts_all, scores_all):
    """Backward compatible parser for old `ocr()` style outputs."""
    if isinstance(obj, (list, tuple)):
        if (
            len(obj) >= 2
            and isinstance(obj[1], (list, tuple))
            and len(obj[1]) >= 2
        ):
            _append_text_score(texts_all, scores_all, obj[1][0], obj[1][1])
        else:
            for item in obj:
                _extract_from_legacy_ocr_format(item, texts_all, scores_all)


def _extract_text_from_ocr_result(recognizer, crop_rgb: np.ndarray):
    rec_text = ""
    rec_score = 0.0

    try:
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        if hasattr(recognizer, "predict"):
            rec_res = recognizer.predict(crop_bgr)
        else:
            rec_res = recognizer.ocr(crop_bgr)

        if not rec_res:
            return rec_text, rec_score

        texts_all = []
        scores_all = []

        items = rec_res if isinstance(rec_res, list) else [rec_res]
        for cand in items:
            parsed = False

            # PaddleOCR v3 predict() -> OCRResult, json: {"res": {"rec_texts": ..., "rec_scores": ...}}
            if hasattr(cand, "json"):
                try:
                    json_data = cand.json
                    payload = json_data.get("res", json_data) if isinstance(json_data, dict) else None
                    if isinstance(payload, dict):
                        rec_texts = payload.get("rec_texts", [])
                        rec_scores = payload.get("rec_scores", [])

                        if not isinstance(rec_texts, (list, tuple)):
                            rec_texts = [rec_texts]
                        if not isinstance(rec_scores, (list, tuple)):
                            rec_scores = [rec_scores]

                        for txt in rec_texts:
                            _append_text_score(texts_all, scores_all, txt, None)
                        for score in rec_scores:
                            _append_text_score(texts_all, scores_all, None, score)
                        parsed = True
                except Exception:
                    parsed = False

            # Sometimes result may already be dict-like.
            if not parsed and isinstance(cand, dict):
                payload = cand.get("res", cand)
                if isinstance(payload, dict):
                    rec_texts = payload.get("rec_texts", [])
                    rec_scores = payload.get("rec_scores", [])

                    if not isinstance(rec_texts, (list, tuple)):
                        rec_texts = [rec_texts]
                    if not isinstance(rec_scores, (list, tuple)):
                        rec_scores = [rec_scores]

                    for txt in rec_texts:
                        _append_text_score(texts_all, scores_all, txt, None)
                    for score in rec_scores:
                        _append_text_score(texts_all, scores_all, None, score)
                    parsed = True

            # Backward compatibility for old OCR list format.
            if not parsed:
                _extract_from_legacy_ocr_format(cand, texts_all, scores_all)

        if texts_all:
            rec_text = " ".join(texts_all).strip()

        if scores_all:
            rec_score = float(max(scores_all))

    except Exception:
        pass

    return rec_text, rec_score


# =============================
# Process single image
# =============================
def process_single_image(ocr_model, img_path, detection_blocks):
    img = read_image_windows(img_path)
    if img is None:
        print("  -> Skipped (image read error)")
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

            rec_text, rec_score = _extract_text_from_ocr_result(
                recognizer, crop
            )

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

    return text_blocks


# =============================
# Save JSON output
# =============================
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


# =============================
# Main pipeline
# =============================
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

    print(f"\nFound {total_files} images")
    print("-" * 50)

    for idx, filename in enumerate(files):
        img_full_path = os.path.join(input_img_dir, filename)
        json_out_path = os.path.join(
            output_dir,
            os.path.splitext(filename)[0] + ".json"
        )

        print(f"[{idx + 1}/{total_files}] Processing: {filename} ... ", end="")

        try:
            json_in_path = os.path.join(
                input_json_dir,
                os.path.splitext(filename)[0] + ".json"
            )

            if not os.path.exists(json_in_path):
                print("Skip (missing detection JSON)")
                continue

            with open(json_in_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            detection_blocks = data["task2"]["output"]["text_blocks"]

            recognized_blocks = process_single_image(
                ocr,
                img_full_path,
                detection_blocks
            )

            save_json(recognized_blocks, json_out_path)

            non_empty_count = sum(
                1 for block in recognized_blocks
                if str(block.get("text", "")).strip()
            )
            print(
                f"Done ({len(recognized_blocks)} boxes, "
                f"{non_empty_count} recognized)"
            )

        except Exception as exc:
            print(f"[ERROR] {exc}")

    print("-" * 50)
    print(f"Finished. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
