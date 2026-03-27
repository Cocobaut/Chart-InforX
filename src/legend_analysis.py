from __future__ import annotations

from typing import Any, Callable

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


BIG_COST = 1e6


def _log(logger: Callable[[str], None] | None, message: str):
    if logger:
        logger(message)


class FeatureMapMeanPool(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)[0]
        return x.mean(dim=(2, 3))


def assign_legend_patches(
    legend_boxes,
    patch_rects,
    y_tol=20,
    prefer_left=True,
    max_cost=None,
):
    n_legends = len(legend_boxes)
    n_patches = len(patch_rects)

    if n_legends == 0 or n_patches == 0:
        return [None] * n_legends

    cost = np.full((n_legends, n_patches), BIG_COST, dtype=np.float32)

    for i, (_, (tx, ty, tw, th)) in enumerate(legend_boxes):
        cx_legend = tx + tw / 2.0
        cy_legend = ty + th / 2.0

        for j, (x, y, w, h) in enumerate(patch_rects):
            cx_patch = x + w / 2.0
            cy_patch = y + h / 2.0

            dy = abs(cy_legend - cy_patch)
            if dy > y_tol:
                continue

            dx = cx_legend - cx_patch

            if prefer_left:
                if dx <= 0:
                    continue
                dist = dx + 0.3 * dy
            else:
                dist = float(np.hypot(dx, dy))

            cost[i, j] = dist

    mapping = [None] * n_legends

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            c = float(cost[i, j])
            if c >= BIG_COST:
                continue
            if max_cost is not None and c > max_cost:
                continue
            mapping[i] = patch_rects[j]
    else:
        used = set()
        for i in range(n_legends):
            row = cost[i]
            j = int(np.argmin(row))
            c = float(row[j])
            if c >= BIG_COST:
                continue
            if max_cost is not None and c > max_cost:
                continue
            if j in used:
                continue
            mapping[i] = patch_rects[j]
            used.add(j)

    return mapping


def shrink_legend_bbox(
    bbox,
    img_size,
    ratio: float = 0.12,
    min_px: int = 2,
    max_px: int = 3,
):
    x1, y1, x2, y2 = bbox
    width, height = img_size
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 2 or bh <= 2:
        return x1, y1, x2, y2

    dx = bw * ratio
    dy = bh * ratio

    max_dx_allowed = max(0, (bw - 2) / 2)
    max_dy_allowed = max(0, (bh - 2) / 2)

    if max_dx_allowed <= 0 or max_dy_allowed <= 0:
        return x1, y1, x2, y2

    dx = min(max(dx, min_px), max_px, max_dx_allowed)
    dy = min(max(dy, min_px), max_px, max_dy_allowed)

    x1_new = max(0, min(x1 + dx, width - 1))
    x2_new = max(0, min(x2 - dx, width))
    y1_new = max(0, min(y1 + dy, height - 1))
    y2_new = max(0, min(y2 - dy, height))

    if x2_new <= x1_new or y2_new <= y1_new:
        return x1, y1, x2, y2

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def shrink_bar_bbox_vertical(
    bbox,
    img_size,
    ratio_x: float = 0.12,
    min_px: int = 2,
    max_px: int = 4,
    shrink_y_px: int = 0,
):
    x1, y1, x2, y2 = bbox
    width, height = img_size
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 2 or bh <= 2:
        return x1, y1, x2, y2

    dx = bw * ratio_x
    max_dx_allowed = max(0, (bw - 2) / 2)
    dx = min(max(dx, min_px), max_px, max_dx_allowed) if max_dx_allowed > 0 else 0

    dy = min(shrink_y_px, max(0, (bh - 2) / 2))

    x1_new = max(0, min(x1 + dx, width - 1))
    x2_new = max(0, min(x2 - dx, width))
    y1_new = max(0, min(y1 + dy, height - 1))
    y2_new = max(0, min(y2 - dy, height))

    if x2_new <= x1_new or y2_new <= y1_new:
        return x1, y1, x2, y2

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def central_crop_to_size(patch_pil: Image.Image, target_size):
    w_target, h_target = target_size
    w, h = patch_pil.size

    if w_target <= 0 or h_target <= 0:
        return patch_pil

    w_target = min(w_target, w)
    h_target = min(h_target, h)

    left = (w - w_target) // 2
    top = (h - h_target) // 2
    right = left + w_target
    bottom = top + h_target

    return patch_pil.crop((left, top, right, bottom))


def get_backbone(model_type: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "resnet50":
        base_model = timm.create_model(
            "resnet50",
            pretrained=True,
            features_only=True,
            out_indices=(1,),
        )
        model = FeatureMapMeanPool(base_model)
        data_config = resolve_model_data_config(base_model)
    elif model_type == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        data_config = resolve_model_data_config(model)
    elif model_type == "efficientnet_b1":
        model = timm.create_model("efficientnet_b1", pretrained=True, num_classes=0)
        data_config = resolve_model_data_config(model)
    elif model_type == "efficientnet_b0_mid":
        base_model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            out_indices=(2,),
        )
        model = FeatureMapMeanPool(base_model)
        data_config = resolve_model_data_config(base_model)
    elif model_type == "efficientnet_b1_mid":
        base_model = timm.create_model(
            "efficientnet_b1",
            pretrained=True,
            features_only=True,
            out_indices=(2,),
        )
        model = FeatureMapMeanPool(base_model)
        data_config = resolve_model_data_config(base_model)
    elif model_type == "clip_vitb32":
        model = timm.create_model(
            "vit_base_patch32_clip_224.openai",
            pretrained=True,
            num_classes=0,
        )
        data_config = resolve_model_data_config(model)
    elif model_type == "swin_tiny":
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0,
        )
        data_config = resolve_model_data_config(model)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()
    model.to(device)
    transform = create_transform(**data_config, is_training=False)

    return {
        "model": model,
        "transform": transform,
        "device": device,
    }


def extract_patch_embedding(
    image,
    bbox,
    backbone,
    kind: str = "generic",
    legend_ref_size: tuple[int, int] | None = None,
    debug_options: dict[str, Any] | None = None,
):
    model = backbone["model"]
    transform = backbone["transform"]
    device = backbone["device"]

    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype("uint8"))
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise TypeError("image must be a numpy array or PIL.Image")

    width, height = image_pil.size
    x1, y1, x2, y2 = bbox

    if kind == "legend":
        x1, y1, x2, y2 = shrink_legend_bbox((x1, y1, x2, y2), (width, height))
    elif kind == "bar":
        x1, y1, x2, y2 = shrink_bar_bbox_vertical((x1, y1, x2, y2), (width, height))

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after preprocessing: {(x1, y1, x2, y2)} for image size {(width, height)}")

    patch = image_pil.crop((x1, y1, x2, y2))

    if kind == "bar" and legend_ref_size is not None:
        patch = central_crop_to_size(patch, legend_ref_size)

    if debug_options and debug_options.get("enabled") and debug_options.get("show_patch"):
        try:
            import matplotlib.pyplot as plt

            print(f"[debug patch][{kind}] bbox={(x1, y1, x2, y2)}")
            plt.figure(figsize=(2, 2))
            plt.imshow(patch)
            plt.axis("off")
            plt.show()
        except Exception:
            pass

    patch_tensor = transform(patch).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(patch_tensor)
        feat = feat.view(feat.size(0), -1)
        feat = F.normalize(feat, p=2, dim=1)

    return feat.squeeze(0).cpu()


def match_legend_to_bars(legend_embs: torch.Tensor, bar_embs: torch.Tensor):
    if legend_embs.ndim != 2 or bar_embs.ndim != 2:
        raise ValueError("legend_embs and bar_embs must be 2D tensors")

    if legend_embs.size(1) != bar_embs.size(1):
        raise ValueError("Embedding dim mismatch between legend and bar")

    legend_embs = legend_embs.float()
    bar_embs = bar_embs.float()

    sim_matrix = torch.matmul(legend_embs, bar_embs.t())
    scores, indices = torch.max(sim_matrix, dim=1)

    return {
        "legend_to_bar": indices.tolist(),
        "scores": scores.tolist(),
        "similarity_matrix": sim_matrix,
    }


def match_legend_patches(legend_text_boxes, legend_dets, image_name, logger=None):
    legendtexts = []
    legendrects = []
    legend_text_rects = []

    if not legend_text_boxes:
        _log(logger, f"[warn] No legend text for {image_name}, fallback to series_0")
        return ["series_0"], legendrects, legend_text_rects

    legend_patch_boxes = []
    for det in legend_dets:
        x1, y1, x2, y2 = det["bbox"]
        legend_patch_boxes.append((x1, y1, x2 - x1, y2 - y1))

    if len(legend_patch_boxes) != len(legend_text_boxes):
        raise ValueError(
            f"Legend count mismatch for {image_name}: {len(legend_patch_boxes)} patches vs {len(legend_text_boxes)} texts"
        )

    assignments = assign_legend_patches(
        legend_boxes=legend_text_boxes,
        patch_rects=legend_patch_boxes,
        y_tol=20,
        prefer_left=True,
        max_cost=None,
    )

    for index, box in enumerate(legend_text_boxes):
        text, (textx, texty, width, height) = box
        patch_box = assignments[index]
        if patch_box is None:
            _log(logger, f"[warn] No patch near legend '{text}' in {image_name}")
            continue

        legendrects.append(patch_box)
        legendtexts.append(text)
        legend_text_rects.append((textx, texty, width, height))

    if not legendtexts:
        _log(logger, f"[warn] Cannot pair legend patch/text for {image_name}, fallback to series_0")
        return ["series_0"], [], []

    return legendtexts, legendrects, legend_text_rects


def extract_embeddings(
    actual_image,
    legendrects,
    bar_rects,
    backbone,
    img_width,
    img_height,
    image_name,
    logger=None,
    debug_options=None,
):
    legend_embs_list = []
    legend_sizes = []

    for (lx, ly, lw, lh) in legendrects:
        bbox_xyxy = (lx, ly, lx + lw, ly + lh)
        try:
            sx1, sy1, sx2, sy2 = shrink_legend_bbox(bbox_xyxy, (img_width, img_height))
            legend_sizes.append((sx2 - sx1, sy2 - sy1))

            emb = extract_patch_embedding(
                actual_image,
                bbox_xyxy,
                backbone,
                kind="legend",
                legend_ref_size=None,
                debug_options=debug_options,
            )
            legend_embs_list.append(emb)
        except Exception as exc:
            _log(logger, f"[warn] Legend embedding failed at {bbox_xyxy} in {image_name}: {exc}")

    legend_ref_size = None
    if legend_sizes:
        avg_w = int(sum(w for w, _ in legend_sizes) / len(legend_sizes))
        avg_h = int(sum(h for _, h in legend_sizes) / len(legend_sizes))
        legend_ref_size = (avg_w, avg_h)

    bar_embs_list = []
    for (bx, by, bw, bh) in bar_rects:
        bbox_xyxy = (bx, by, bx + bw, by + bh)
        try:
            emb = extract_patch_embedding(
                actual_image,
                bbox_xyxy,
                backbone,
                kind="bar",
                legend_ref_size=legend_ref_size,
                debug_options=debug_options,
            )
            bar_embs_list.append(emb)
        except Exception as exc:
            _log(logger, f"[warn] Bar embedding failed at {bbox_xyxy} in {image_name}: {exc}")

    return legend_embs_list, bar_embs_list


def match_legend_to_bars_pipeline(
    legend_embs_list,
    bar_embs_list,
    legendtexts,
    bar_rects,
    image_name,
    logger=None,
):
    legend_for_bar = None
    sim_matrix = None

    if legend_embs_list and bar_embs_list and len(legend_embs_list) == len(legendtexts):
        try:
            legend_embs = torch.stack(legend_embs_list)
            bar_embs = torch.stack(bar_embs_list)
            matches = match_legend_to_bars(legend_embs, bar_embs)
            sim_matrix = matches["similarity_matrix"]
            legend_for_bar = torch.argmax(sim_matrix, dim=0).tolist()
        except Exception as exc:
            _log(logger, f"[warn] match_legend_to_bars failed in {image_name}: {exc}")
            legend_for_bar = None

    if legend_for_bar is None:
        if len(legendtexts) == 1:
            legend_for_bar = [0] * len(bar_rects)
        else:
            _log(logger, f"[warn] fallback: assign all bars to first legend in {image_name}")
            legend_for_bar = [0] * len(bar_rects)

    return legend_for_bar, sim_matrix


def to_serializable_legend_result(legendtexts, legendrects, legend_text_rects, legend_for_bar):
    return {
        "legend_texts": list(legendtexts),
        "legend_patches": [list(rect) for rect in legendrects],
        "legend_text_rects": [list(rect) for rect in legend_text_rects],
        "legend_for_bar": list(legend_for_bar),
    }
