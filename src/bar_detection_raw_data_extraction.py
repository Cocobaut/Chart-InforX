from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

import axis_analysis
import config
import legend_analysis


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _normalize_debug_options(debug_options: dict[str, Any] | None = None):
    cfg = {
        "enabled": False,
        "print_logs": True,
        "show_patch": False,
        "show_similarity": False,
        "save_overlay": False,
        "debug_dir": None,
    }
    if debug_options:
        cfg.update(debug_options)
    return cfg


def _debug_log(debug_options, message):
    if not debug_options:
        return
    if debug_options.get("enabled") and debug_options.get("print_logs", True):
        print(message)


def _resolve_yolo_device(device):
    if isinstance(device, int):
        return device

    if isinstance(device, str):
        d = device.strip().lower()
        if d == "cuda":
            return 0 if torch.cuda.is_available() else "cpu"
        return d

    return 0 if torch.cuda.is_available() else "cpu"


def run_inference(image_paths, output_dir: Path, detector, yolo_device):
    if not image_paths:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    return detector.predict(
        source=[str(path) for path in image_paths],
        imgsz=832,
        conf=0.25,
        iou=0.5,
        save=False,
        project=str(output_dir),
        name="inference_yolo",
        device=yolo_device,
        verbose=False,
    )


def yolo_detect_objects(result):
    boxes = result.boxes
    names = result.names
    detections_by_class = {name: [] for name in names.values()}

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        cls_name = names[cls_id]

        detections_by_class[cls_name].append(
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "score": conf,
            }
        )

    legend_dets = detections_by_class.get("legend", [])
    bar_dets = detections_by_class.get("bar", [])
    plot_dets = detections_by_class.get("plot", [])
    return legend_dets, bar_dets, plot_dets


def _rect_to_xyxy(rect):
    x, y, w, h = rect
    return int(x), int(y), int(x + w), int(y + h)


def _rect_center(rect):
    x, y, w, h = rect
    return int(x + w / 2.0), int(y + h / 2.0)


def draw_debug_image(
    base_image_rgb,
    xaxis,
    yaxis,
    legend_patches,
    legend_text_boxes,
    bar_rects,
    x_label_rects,
    legend_for_bar,
    x_label_for_bar,
    save_path,
):
    image = cv2.cvtColor(base_image_rgb.copy(), cv2.COLOR_RGB2BGR)

    color_axis = (0, 0, 0)
    color_xlabel = (160, 160, 160)
    thickness = 1

    palette = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (128, 0, 255),
        (0, 128, 255),
        (128, 255, 0),
        (255, 128, 0),
    ]

    def color_for_legend(index):
        if index is None or index < 0:
            return (100, 100, 100)
        return palette[index % len(palette)]

    if xaxis is not None:
        x1, y1, x2, y2 = map(int, xaxis)
        cv2.line(image, (x1, y1), (x2, y2), color_axis, thickness)

    if yaxis is not None:
        x1, y1, x2, y2 = map(int, yaxis)
        cv2.line(image, (x1, y1), (x2, y2), color_axis, thickness)

    for rect in x_label_rects or []:
        x1, y1, x2, y2 = _rect_to_xyxy(rect)
        cv2.rectangle(image, (x1, y1), (x2, y2), color_xlabel, thickness)

    legend_count = min(len(legend_patches or []), len(legend_text_boxes or []))
    for legend_index in range(legend_count):
        color = color_for_legend(legend_index)
        patch_rect = legend_patches[legend_index]
        text_rect = legend_text_boxes[legend_index]

        p_x1, p_y1, p_x2, p_y2 = _rect_to_xyxy(patch_rect)
        t_x1, t_y1, t_x2, t_y2 = _rect_to_xyxy(text_rect)
        cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), color, thickness)
        cv2.rectangle(image, (t_x1, t_y1), (t_x2, t_y2), color, thickness)

        c_patch = _rect_center(patch_rect)
        c_text = _rect_center(text_rect)
        cv2.line(image, c_patch, c_text, color, thickness)

    for index, bar_rect in enumerate(bar_rects or []):
        legend_index = None
        if legend_for_bar is not None and index < len(legend_for_bar):
            legend_index = legend_for_bar[index]

        color = color_for_legend(legend_index)
        c_bar = _rect_center(bar_rect)

        x1, y1, x2, y2 = _rect_to_xyxy(bar_rect)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if legend_index is not None and 0 <= legend_index < legend_count:
            c_patch = _rect_center(legend_patches[legend_index])
            cv2.line(image, c_bar, c_patch, color, thickness)

        if x_label_for_bar is not None and index < len(x_label_for_bar):
            label_rect = x_label_for_bar[index]
            if label_rect is not None:
                c_lbl = _rect_center(label_rect)
                cv2.line(image, c_bar, c_lbl, color, thickness)
                lx1, ly1, lx2, ly2 = _rect_to_xyxy(label_rect)
                cv2.rectangle(image, (lx1, ly1), (lx2, ly2), color, thickness)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)


def _debug_print_bar_similarity(debug_options, image_name, sim_matrix, legend_for_bar, legendtexts, bar_rects):
    if not debug_options:
        return
    if not (debug_options.get("enabled") and debug_options.get("show_similarity")):
        return
    if sim_matrix is None:
        return

    n_legends, n_bars = sim_matrix.shape
    print(f"[debug similarity] {image_name}: L={n_legends}, B={n_bars}")
    for bar_index, (bx, by, bw, bh) in enumerate(bar_rects):
        if bar_index >= len(legend_for_bar):
            continue

        legend_index = legend_for_bar[bar_index]
        if legend_index >= len(legendtexts):
            continue

        best_score = sim_matrix[legend_index, bar_index].item()
        print(
            f"  Bar[{bar_index}] bbox=({bx}, {by}, {bw}, {bh}) -> "
            f"Legend[{legend_index}] '{legendtexts[legend_index]}', score={best_score:.4f}"
        )


def _rect_dist_x(rect_a, rect_b):
    ax, _, aw, _ = rect_a
    bx, _, bw, _ = rect_b
    return abs((ax + aw / 2.0) - (bx + bw / 2.0))


def compute_bar_values(
    legendtexts,
    x_tick_list,
    y_tick_list,
    bar_rects,
    legend_for_bar,
    normalize_ratio,
    min_val,
    debug_options=None,
):
    if not legendtexts:
        legendtexts = ["series_0"]

    text_boxes = []
    labels = []

    for rect_box in bar_rects:
        min_distance = sys.maxsize
        closest_box = None
        label_text = None

        for text, text_box in x_tick_list:
            distance = _rect_dist_x(rect_box, text_box)
            if distance < min_distance:
                min_distance = distance
                closest_box = text_box
                label_text = text

        text_boxes.append(closest_box)
        labels.append(label_text)

    bar_heights = [(rect, float(rect[3])) for rect in bar_rects]
    ndigits = axis_analysis.infer_ndigits_from_ticks(y_tick_list, default=1, cap=3)
    y_values = [(rect, round(min_val + h * normalize_ratio, ndigits + 1)) for rect, h in bar_heights]

    is_flat_mode = len(x_tick_list) == 0
    if is_flat_mode:
        data = {legend_text: 0.0 for legend_text in legendtexts}
    else:
        data = {
            legend_text: {x_label: 0.0 for x_label, _ in x_tick_list}
            for legend_text in legendtexts
        }

    for legend_index, legend_text in enumerate(legendtexts):
        _debug_log(debug_options, f"[value] Assign values for legend '{legend_text}'")

        if is_flat_mode:
            found_val = 0.0
            for bar_index, item in enumerate(y_values):
                if bar_index < len(legend_for_bar) and legend_for_bar[bar_index] == legend_index:
                    found_val = item[1]
                    break
            data[legend_text] = found_val
            continue

        for x_label, box in x_tick_list:
            x, _, w, _ = box
            value = 0.0
            dist = sys.maxsize

            for bar_index, item in enumerate(y_values):
                if bar_index >= len(legend_for_bar):
                    continue
                if legend_for_bar[bar_index] != legend_index:
                    continue
                if labels[bar_index] != x_label:
                    continue

                vx, _, vw, _ = item[0]
                cx_bar = vx + vw / 2.0
                cx_lbl = x + w / 2.0
                distance = abs(cx_lbl - cx_bar)
                if distance < dist:
                    dist = distance
                    value = item[1]

            data[legend_text][x_label] = value

    return data, text_boxes


def _write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _get_sorted_texts(items, axis_type: str):
    normalized = []
    for text, rect in items or []:
        text_norm = str(text).strip()
        if not text_norm:
            continue
        if not isinstance(rect, (list, tuple)) or len(rect) < 4:
            continue
        x, y, w, h = map(float, rect[:4])
        normalized.append((text_norm, (x, y, w, h)))

    if axis_type == "x":
        normalized.sort(key=lambda item: item[1][0] + item[1][2] / 2.0)
    elif axis_type == "y":
        normalized.sort(key=lambda item: item[1][1] + item[1][3] / 2.0)
    elif axis_type == "title_x":
        normalized.sort(key=lambda item: item[1][0])
    elif axis_type == "title_y":
        normalized.sort(key=lambda item: item[1][1])

    texts = []
    for text, _ in normalized:
        if text not in texts:
            texts.append(text)
    return texts


def _get_text_blocks_from_payload(task_payload: dict[str, Any]):
    blocks = (
        task_payload.get("task3", {})
        .get("input", {})
        .get("task2_output", {})
        .get("text_blocks")
    )
    if isinstance(blocks, list):
        return blocks

    blocks = (
        task_payload.get("task2", {})
        .get("output", {})
        .get("text_blocks")
    )
    if isinstance(blocks, list):
        return blocks

    return []


def _extract_doi_candidates(image_name: str, task_payload: dict[str, Any]):
    doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
    candidates = []

    def add_matches(text_value):
        if text_value is None:
            return
        text_str = str(text_value)
        for match in doi_pattern.findall(text_str):
            doi = match.rstrip(".,;:)]}>")
            doi = f"doi:{doi}"
            if doi not in candidates:
                candidates.append(doi)

    add_matches(image_name)
    for block in _get_text_blocks_from_payload(task_payload):
        add_matches(block.get("text", ""))

    return candidates


def _order_data_by_xlabels(data: dict[str, Any], x_labels: list[str]):
    if not isinstance(data, dict):
        return data

    ordered_data = {}
    for legend, legend_values in data.items():
        if not isinstance(legend_values, dict):
            ordered_data[legend] = legend_values
            continue

        ordered_legend_values = {}
        for x_label in x_labels:
            if x_label in legend_values:
                ordered_legend_values[x_label] = legend_values[x_label]
        for key, value in legend_values.items():
            if key not in ordered_legend_values:
                ordered_legend_values[key] = value

        ordered_data[legend] = ordered_legend_values

    return ordered_data


def _build_paper_format_record(image_name: str, task_payload: dict[str, Any], axis_result: dict[str, Any], legendtexts, data):
    x_text = _get_sorted_texts(axis_result.get("x_title", []), axis_type="title_x")
    x_labels = _get_sorted_texts(axis_result.get("x_tick_list", []), axis_type="x")
    y_text = _get_sorted_texts(axis_result.get("y_title", []), axis_type="title_y")
    y_labels = _get_sorted_texts(axis_result.get("y_tick_list", []), axis_type="y")

    return {
        "file name": image_name,
        "doi": _extract_doi_candidates(image_name, task_payload),
        "x-text": x_text,
        "x-labels": x_labels,
        "y-text": y_text,
        "y-labels": y_labels,
        "legends": list(legendtexts),
        "data": _order_data_by_xlabels(data, x_labels),
    }


def _build_paper_format_text(records: list[dict[str, Any]]):
    lines = []
    for idx, item in enumerate(records):
        lines.append(f"file name      : {item.get('file name', '')}")
        lines.append(f"doi            : {repr(item.get('doi', []))}")
        lines.append(f"x-text         : {repr(item.get('x-text', []))}")
        lines.append(f"x-labels       : {repr(item.get('x-labels', []))}")
        lines.append(f"y-text         : {repr(item.get('y-text', []))}")
        lines.append(f"y-labels       : {repr(item.get('y-labels', []))}")
        lines.append(f"legends        : {repr(item.get('legends', []))}")
        lines.append(f"data           : {repr(item.get('data', {}))}")
        if idx < len(records) - 1:
            lines.append("")
    return "\n".join(lines)


def get_y_values(
    img_dir,
    json_dir,
    predict_dir,
    detector,
    backbone,
    yolo_device,
    axis_output_dir=None,
    legend_output_dir=None,
    debug_options=None,
):
    debug_options = _normalize_debug_options(debug_options)

    img_dir = Path(img_dir)
    json_dir = Path(json_dir)
    predict_dir = Path(predict_dir)

    axis_output_dir = Path(axis_output_dir) if axis_output_dir else None
    legend_output_dir = Path(legend_output_dir) if legend_output_dir else None

    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
    image_paths.sort()

    results = run_inference(image_paths, predict_dir, detector, yolo_device)

    y_value_dict = {}
    paper_format_records = []

    for index, path in enumerate(image_paths):
        json_path = json_dir / f"{path.stem}.json"
        if not json_path.exists():
            print(f"[warn] Missing JSON for image {path.name}: {json_path}")
            continue

        try:
            task_payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] Failed to read JSON {json_path}: {exc}")
            continue

        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            print(f"[warn] Cannot read image {path.name}, skip")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image_rgb.shape
        actual_image = Image.open(path).convert("RGB")

        legend_dets, bar_dets, plot_dets = yolo_detect_objects(results[index])

        axis_result = axis_analysis.analyze_axis(image_rgb, task_payload, plot_dets)
        if axis_result is None:
            print(f"[warn] Cannot detect axes for {path.name}, skip")
            continue

        xaxis = axis_result["xaxis"]
        yaxis = axis_result["yaxis"]
        x_tick_list = axis_result["x_tick_list"]
        y_tick_list = axis_result["y_tick_list"]
        legend_text_boxes = axis_result["legend_text_boxes"]
        normalize_ratio = axis_result["normalize_ratio"]
        min_val = axis_result["min_val"]

        if normalize_ratio == 0:
            print(f"[warn] Cannot compute normalize_ratio for {path.name}, skip")
            continue

        _debug_log(debug_options, f"[{index}] path: {path.name}, ratio: {normalize_ratio}")

        if axis_output_dir:
            axis_payload = {
                "task": "Task3_axis_analysis",
                "image": path.name,
                "output": axis_analysis.to_serializable_axis_result(axis_result),
            }
            _write_json(axis_output_dir / f"{path.stem}.json", axis_payload)

        try:
            legendtexts, legendrects, legend_text_rects = legend_analysis.match_legend_patches(
                legend_text_boxes,
                legend_dets,
                path.name,
                logger=lambda msg: _debug_log(debug_options, msg),
            )
        except Exception as exc:
            print(f"[warn] Legend patch matching failed for {path.name}: {exc}")
            continue

        bar_rects = []
        for det in bar_dets:
            x1, y1, x2, y2 = det["bbox"]
            bar_rects.append((x1, y1, x2 - x1, y2 - y1))

        if not bar_rects:
            print(f"[warn] No bars detected for {path.name}, skip")
            continue

        legend_embs_list, bar_embs_list = legend_analysis.extract_embeddings(
            actual_image,
            legendrects,
            bar_rects,
            backbone,
            img_width,
            img_height,
            path.name,
            logger=lambda msg: _debug_log(debug_options, msg),
            debug_options=debug_options,
        )

        legend_for_bar, sim_matrix = legend_analysis.match_legend_to_bars_pipeline(
            legend_embs_list,
            bar_embs_list,
            legendtexts,
            bar_rects,
            path.name,
            logger=lambda msg: _debug_log(debug_options, msg),
        )

        _debug_print_bar_similarity(
            debug_options,
            path.name,
            sim_matrix,
            legend_for_bar,
            legendtexts,
            bar_rects,
        )

        data, text_boxes = compute_bar_values(
            legendtexts,
            x_tick_list,
            y_tick_list,
            bar_rects,
            legend_for_bar,
            normalize_ratio,
            min_val,
            debug_options=debug_options,
        )

        if debug_options.get("enabled") and debug_options.get("save_overlay") and debug_options.get("debug_dir"):
            overlay_path = Path(debug_options["debug_dir"]) / f"{path.stem}_debug.png"
            x_label_rects = [box for _, box in x_tick_list]
            draw_debug_image(
                base_image_rgb=image_rgb,
                xaxis=xaxis,
                yaxis=yaxis,
                legend_patches=legendrects,
                legend_text_boxes=legend_text_rects,
                bar_rects=bar_rects,
                x_label_rects=x_label_rects,
                legend_for_bar=legend_for_bar,
                x_label_for_bar=text_boxes,
                save_path=overlay_path,
            )
            _debug_log(debug_options, f"[debug] saved overlay: {overlay_path}")

        if legend_output_dir:
            legend_payload = {
                "task": "Task4_legend_analysis",
                "image": path.name,
                "output": legend_analysis.to_serializable_legend_result(
                    legendtexts,
                    legendrects,
                    legend_text_rects,
                    legend_for_bar,
                ),
            }
            _write_json(legend_output_dir / f"{path.stem}.json", legend_payload)

        y_value_dict[path.name] = data
        paper_format_records.append(
            _build_paper_format_record(
                image_name=path.name,
                task_payload=task_payload,
                axis_result=axis_result,
                legendtexts=legendtexts,
                data=data,
            )
        )

    return y_value_dict, paper_format_records


def _build_rows(y_value_dict):
    rows = []
    for image_name, legends_dict in y_value_dict.items():
        for legend, legend_data in legends_dict.items():
            if isinstance(legend_data, dict):
                for x_label, value in legend_data.items():
                    rows.append(
                        {
                            "image": image_name,
                            "legend": legend,
                            "x_label": x_label,
                            "value": float(value),
                        }
                    )
            else:
                rows.append(
                    {
                        "image": image_name,
                        "legend": legend,
                        "x_label": "Value",
                        "value": float(legend_data),
                    }
                )
    return rows


def save_results(
    df: pd.DataFrame,
    y_value_dict: dict,
    csv_path: str | Path,
    json_path: str | Path,
    paper_format_records: list[dict[str, Any]] | None = None,
    paper_json_path: str | Path | None = None,
    paper_txt_path: str | Path | None = None,
):
    csv_path = Path(csv_path)
    json_path = Path(json_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "meta": {
            "n_rows": int(len(df)),
            "columns": list(df.columns),
        },
        "yValueDict": y_value_dict,
        "records": df.to_dict(orient="records"),
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("Saved CSV:", csv_path)
    print("Saved JSON:", json_path)

    records = paper_format_records or []

    if paper_json_path:
        paper_json_path = Path(paper_json_path)
        paper_json_path.parent.mkdir(parents=True, exist_ok=True)
        paper_payload = {
            "meta": {"n_images": len(records)},
            "records": records,
        }
        with open(paper_json_path, "w", encoding="utf-8") as file:
            json.dump(paper_payload, file, ensure_ascii=False, indent=2)
        print("Saved JSON (paper format):", paper_json_path)

    if paper_txt_path:
        paper_txt_path = Path(paper_txt_path)
        paper_txt_path.parent.mkdir(parents=True, exist_ok=True)
        paper_text = _build_paper_format_text(records)
        with open(paper_txt_path, "w", encoding="utf-8") as file:
            file.write(paper_text)
        print("Saved TXT (paper format):", paper_txt_path)


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    task5_config = config.return_task5_detect_extraction_config()
    task3_config = config.return_task3_axis_analysis_config()
    task4_config = config.return_task4_legend_analysis_config()

    yolo_device = _resolve_yolo_device(task5_config.get("device", "cuda"))
    backbone_device = "cuda" if yolo_device != "cpu" else "cpu"

    detector = YOLO(task5_config["yolo_weight"])
    backbone = legend_analysis.get_backbone(
        task5_config.get("feature_backbone") or task4_config.get("feature_backbone", "resnet50"),
        device=backbone_device,
    )

    debug_options = _normalize_debug_options(
        task5_config.get(
            "debug_options",
            {
                "enabled": True,
                "print_logs": False,
                "show_patch": False,
                "show_similarity": False,
                "save_overlay": True,
                "debug_dir": str(Path(task5_config["output_dir"]) / "debug_viz"),
            },
        )
    )

    y_value_dict, paper_format_records = get_y_values(
        img_dir=task5_config["input_images"],
        json_dir=task5_config["input_json"],
        predict_dir=task5_config["output_dir"],
        detector=detector,
        backbone=backbone,
        yolo_device=yolo_device,
        axis_output_dir=task5_config.get("axis_output_json", task3_config["output_json"]),
        legend_output_dir=task5_config.get("legend_output_json", task4_config["output_json"]),
        debug_options=debug_options,
    )

    rows = _build_rows(y_value_dict)
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["image", "legend", "x_label", "value"])

    save_results(
        df=df,
        y_value_dict=y_value_dict,
        csv_path=task5_config["output_csv"],
        json_path=task5_config["output_json"],
        paper_format_records=paper_format_records,
        paper_json_path=task5_config.get(
            "output_paper_json",
            str(Path(task5_config["output_dir"]) / "result_paper_format.json"),
        ),
        paper_txt_path=task5_config.get(
            "output_paper_txt",
            str(Path(task5_config["output_dir"]) / "result_paper_format.txt"),
        ),
    )


if __name__ == "__main__":
    main()
