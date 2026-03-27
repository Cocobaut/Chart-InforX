from __future__ import annotations

import re
from typing import Any

import numpy as np


def clean_text(image_text: list[tuple[str, tuple[float, float, float, float]]]):
    return [
        (text, (textx, texty, w, h))
        for text, (textx, texty, w, h) in image_text
        if text.strip() != "I"
    ]


def point_line_distance(px, py, x1, y1, x2, y2):
    return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)


def detect_axes(plot_dets: list[dict[str, Any]]):
    if not plot_dets:
        return None, None

    best_plot = max(
        plot_dets,
        key=lambda det: (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1]),
    )

    x1, y1, x2, y2 = map(int, best_plot["bbox"])
    xaxis = (x1, y2, x2, y2)
    yaxis = (x1, y1, x1, y2)
    return xaxis, yaxis


def get_probable_labels(image, payload: dict[str, Any], xaxis, yaxis):
    try:
        text_blocks = payload["task3"]["input"]["task2_output"]["text_blocks"]
    except KeyError:
        text_blocks = payload.get("task2", {}).get("output", {}).get("text_blocks", [])

    id_to_text = {}
    id_to_rect = {}
    raw_image_text = []

    for block in text_blocks:
        bid = block["id"]
        text = block.get("text", "")
        poly = block["polygon"]

        xs = [poly["x0"], poly["x1"], poly["x2"], poly["x3"]]
        ys = [poly["y0"], poly["y1"], poly["y2"], poly["y3"]]
        x_min, y_min = min(xs), min(ys)
        rect = (x_min, y_min, max(xs) - x_min, max(ys) - y_min)

        id_to_text[bid] = text
        id_to_rect[bid] = rect
        raw_image_text.append((text, rect))

    image_text = clean_text(raw_image_text)

    text_roles = payload.get("task3", {}).get("output", {}).get("text_roles", [])
    id_to_role = {item["id"]: item["role"] for item in text_roles}

    tick_blocks = []
    axis_blocks = []
    legend_blocks = []

    for bid, role in id_to_role.items():
        if bid not in id_to_text:
            continue

        text = id_to_text[bid]
        rect = id_to_rect[bid]

        if role == "tick_label":
            tick_blocks.append((text, rect))
        elif role == "axis_title":
            axis_blocks.append((text, rect))
        elif role == "legend_label":
            legend_blocks.append((text, rect))

    x1, y1, x2, y2 = xaxis
    yx1, yy1, yx2, yy2 = yaxis

    x_tick_list = []
    y_tick_list = []

    for text, (tx, ty, w, h) in tick_blocks:
        cx = tx + w / 2.0
        cy = ty + h / 2.0

        side_xaxis = np.sign((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1))
        side_yaxis = np.sign((yx2 - yx1) * (cy - yy1) - (yy2 - yy1) * (cx - yx1))

        if side_yaxis == 1:
            y_tick_list.append((text, (tx, ty, w, h)))
        elif side_xaxis == 1 and side_yaxis == -1:
            x_tick_list.append((text, (tx, ty, w, h)))

    x_title = []
    y_title = []

    for text, (tx, ty, w, h) in axis_blocks:
        cx = tx + w / 2.0
        cy = ty + h / 2.0

        dist_to_x = point_line_distance(cx, cy, x1, y1, x2, y2)
        dist_to_y = point_line_distance(cx, cy, yx1, yy1, yx2, yy2)

        if dist_to_y < dist_to_x:
            y_title.append((text, (tx, ty, w, h)))
        else:
            x_title.append((text, (tx, ty, w, h)))

    legend_text_boxes = legend_blocks[:]

    return (
        image,
        x_tick_list,
        x_title,
        y_tick_list,
        y_title,
        legend_text_boxes,
        image_text,
    )


def reject_outliers(data, m=1):
    if len(data) == 0:
        return data

    std = float(np.std(data))
    if std == 0:
        return data

    return data[abs(data - np.mean(data)) <= m * std]


def get_ratio_optimized(y_tick_list):
    list_text = []
    list_ticks = []
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    for text, (_, texty, _, h) in y_tick_list:
        numbers = re.findall(pattern, text.strip())
        if not numbers:
            continue
        try:
            best_match = max(numbers, key=len)
            list_text.append(float(best_match))
            list_ticks.append(float(texty + h))
        except ValueError:
            continue

    if len(list_text) < 2:
        return sorted(list_text), 0.0, (0.0, 0.0)

    text_sorted = sorted(list_text)
    ticks_sorted = sorted(list_ticks)

    ticks_diff = np.array(
        [ticks_sorted[i] - ticks_sorted[i - 1] for i in range(1, len(ticks_sorted))],
        dtype=float,
    )
    text_diff = np.array(
        [text_sorted[i] - text_sorted[i - 1] for i in range(1, len(text_sorted))],
        dtype=float,
    )

    ticks_diff = reject_outliers(ticks_diff, m=1)
    text_diff = reject_outliers(text_diff, m=1)

    if len(ticks_diff) == 0:
        return text_sorted, 0.0, (text_sorted[0], ticks_sorted[0])

    mean_ticks_diff = float(np.mean(ticks_diff))
    if mean_ticks_diff == 0:
        return text_sorted, 0.0, (text_sorted[0], ticks_sorted[0])

    normalize_ratio = float(np.mean(text_diff) / mean_ticks_diff)
    return text_sorted, normalize_ratio, (text_sorted[0], ticks_sorted[0])


def compute_ratio(y_tick_list):
    return get_ratio_optimized(y_tick_list)


def infer_ndigits_from_ticks(y_tick_list, default=1, cap=3):
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    decimals = []

    for text, _ in y_tick_list:
        nums = re.findall(pattern, text.strip())
        if not nums:
            continue

        token = max(nums, key=len)
        if "e" in token.lower():
            return min(cap, max(default, 2))

        if "." in token:
            frac = token.split(".", 1)[1]
            frac = frac.split("e", 1)[0].split("E", 1)[0]
            decimals.append(len(frac.rstrip("0")))
        else:
            decimals.append(0)

    if not decimals:
        return default
    return min(cap, max(decimals))


def analyze_axis(image, payload: dict[str, Any], plot_dets: list[dict[str, Any]]):
    xaxis, yaxis = detect_axes(plot_dets)
    if xaxis is None or yaxis is None:
        return None

    (
        image,
        x_tick_list,
        x_title,
        y_tick_list,
        y_title,
        legend_text_boxes,
        image_text,
    ) = get_probable_labels(image, payload, xaxis, yaxis)

    list_text, normalize_ratio, (min_val, min_pixel) = compute_ratio(y_tick_list)

    return {
        "image": image,
        "xaxis": xaxis,
        "yaxis": yaxis,
        "x_tick_list": x_tick_list,
        "x_title": x_title,
        "y_tick_list": y_tick_list,
        "y_title": y_title,
        "legend_text_boxes": legend_text_boxes,
        "image_text": image_text,
        "list_text": list_text,
        "normalize_ratio": normalize_ratio,
        "min_val": min_val,
        "min_pixel": min_pixel,
    }


def to_serializable_axis_result(axis_result: dict[str, Any]):
    return {
        "xaxis": list(axis_result["xaxis"]),
        "yaxis": list(axis_result["yaxis"]),
        "x_tick_list": [
            {"text": text, "rect": list(rect)} for text, rect in axis_result["x_tick_list"]
        ],
        "x_title": [{"text": text, "rect": list(rect)} for text, rect in axis_result["x_title"]],
        "y_tick_list": [
            {"text": text, "rect": list(rect)} for text, rect in axis_result["y_tick_list"]
        ],
        "y_title": [{"text": text, "rect": list(rect)} for text, rect in axis_result["y_title"]],
        "legend_text_boxes": [
            {"text": text, "rect": list(rect)} for text, rect in axis_result["legend_text_boxes"]
        ],
        "normalize_ratio": axis_result["normalize_ratio"],
        "min_val": axis_result["min_val"],
        "min_pixel": axis_result["min_pixel"],
    }


# Backward-compatible aliases from data_extractor_v10.py naming.
getProbableLabels = get_probable_labels
getRatio_optimized = get_ratio_optimized
