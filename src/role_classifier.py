import torch
import os
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification, LayoutLMv3Model
import torch.nn as nn
from tqdm.auto import tqdm
import config


# =============================
# Configuration
# =============================
try:
    TEST_CONFIG = config.return_task2_role_classifier_config()
except AttributeError:
    TEST_CONFIG = config.returnTestTask3_Config()


def resolve_torch_device(requested_device):
    """Resolve runtime device and fallback to CPU if CUDA is unavailable."""
    try:
        requested = str(requested_device).strip() if requested_device is not None else "cpu"
        resolved = str(torch.device(requested))
    except (TypeError, RuntimeError, ValueError):
        print(f"[WARN] Invalid device '{requested_device}', fallback to CPU.")
        return "cpu"

    if resolved.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Using CPU.")
        return "cpu"

    return resolved


# =============================
# Model Definition
# =============================
class LayoutLMv3ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomLayoutLMv3(LayoutLMv3ForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = LayoutLMv3ClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids=None, bbox=None, attention_mask=None,
                labels=None, pixel_values=None, **kwargs):

        kwargs.pop("num_items_in_batch", None)

        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs
        )

        if input_ids is not None:
            sequence_output = outputs[0][:, :input_ids.size(1), :]
        else:
            sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


# =============================
# Data utilities
# =============================
def normalize_bbox(box, w, h):
    return [
        int(1000 * box[0] / w),
        int(1000 * box[1] / h),
        int(1000 * box[2] / w),
        int(1000 * box[3] / h)
    ]


def load_icpr_bar_charts_flat(data_dir_images, data_dir_json, target_labels):
    dataset_dicts = []
    label2id = {label: i for i, label in enumerate(target_labels)}

    if not os.path.exists(data_dir_json) or not os.path.exists(data_dir_images):
        print("Error: image or annotation directory not found.")
        return []

    files = sorted([f for f in os.listdir(data_dir_json) if f.endswith(".json")])
    print(f"Found {len(files)} annotation files.")

    for file in tqdm(files, desc="Loading data"):
        try:
            json_path = os.path.join(data_dir_json, file)

            with open(json_path, "r", encoding="utf8") as f:
                data = json.load(f)

            task_data = data.get("task2")
            if not task_data:
                continue

            chart_type = "vertical bar"
            try:
                chart_type = task_data["input"]["task1_output"]["chart_type"]
            except:
                pass

            img_name_base = os.path.splitext(file)[0]

            matched_path = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
                candidate = os.path.join(data_dir_images, img_name_base + ext)
                if os.path.exists(candidate):
                    matched_path = candidate
                    break

            if not matched_path:
                continue

            with Image.open(matched_path) as img:
                w, h = img.size

            text_blocks = task_data["output"]["text_blocks"]

            words, bboxes, labels = [], [], []
            original_blocks_cleaned = []

            for block in text_blocks:
                text = block.get("text", "").strip()
                if not text:
                    continue

                role_str = "OTHER"

                poly = block.get("polygon")
                if isinstance(poly, dict):
                    poly_list = [
                        poly["x0"], poly["y0"],
                        poly["x1"], poly["y1"],
                        poly["x2"], poly["y2"],
                        poly["x3"], poly["y3"]
                    ]
                elif isinstance(poly, list):
                    poly_list = poly
                else:
                    continue

                x_c, y_c = poly_list[0::2], poly_list[1::2]
                box = [min(x_c), min(y_c), max(x_c), max(y_c)]

                words.append(text)
                bboxes.append(normalize_bbox(box, w, h))
                labels.append(label2id.get(role_str, 0))

                original_blocks_cleaned.append(block)

            if words:
                dataset_dicts.append({
                    "id": file,
                    "image_path": matched_path,
                    "words": words,
                    "bboxes": bboxes,
                    "labels": labels,
                    "original_blocks": original_blocks_cleaned,
                    "chart_type": chart_type
                })

        except:
            continue

    return dataset_dicts


# =============================
# Visualization
# =============================
def visualize_result(image_path, original_blocks, predicted_roles_map, output_path):
    """Draw predicted roles on image."""
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        colors = {
            "chart_title": (255, 0, 0),
            "axis_title": (0, 0, 255),
            "tick_label": (0, 128, 0),
            "legend_label": (255, 165, 0),
            "legend_title": (255, 20, 147),
            "mark_label": (128, 0, 128),
            "value_label": (0, 255, 255),
            "other": (192, 192, 192)
        }

        for i, block in enumerate(original_blocks):
            role = predicted_roles_map.get(i, "other").lower()
            color = colors.get(role, (0, 0, 0))

            poly = block.get("polygon")
            xy_coords = []

            if isinstance(poly, dict):
                xy_coords = [
                    (poly["x0"], poly["y0"]),
                    (poly["x1"], poly["y1"]),
                    (poly["x2"], poly["y2"]),
                    (poly["x3"], poly["y3"])
                ]
            elif isinstance(poly, list):
                xy_coords = list(zip(poly[0::2], poly[1::2]))

            if xy_coords:
                draw.polygon(xy_coords, outline=color, width=3)

                x_min = min(p[0] for p in xy_coords)
                y_min = min(p[1] for p in xy_coords)

                text_bbox = draw.textbbox((x_min, y_min), role, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x_min, y_min), role, fill="white", font=font)

        image.save(output_path)

    except Exception as e:
        print(f"Visualization error: {image_path} -> {str(e)}")


# =============================
# Main inference
# =============================
def main():
    print(f"Loading model from {TEST_CONFIG['model_path']}")

    try:
        model = CustomLayoutLMv3.from_pretrained(TEST_CONFIG["model_path"])
    except:
        model = LayoutLMv3ForTokenClassification.from_pretrained(TEST_CONFIG["model_path"])

    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False
    )

    runtime_device = resolve_torch_device(TEST_CONFIG.get("device", "cpu"))
    print(f"Using device: {runtime_device}")

    model.to(runtime_device)
    model.eval()

    if not os.path.exists(TEST_CONFIG["output_dir"]):
        os.makedirs(TEST_CONFIG["output_dir"])

    vis_dir = os.path.join(TEST_CONFIG["output_dir"], "visualization")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"Created visualization directory: {vis_dir}")

    test_data = load_icpr_bar_charts_flat(
        TEST_CONFIG["data_dir_images"],
        TEST_CONFIG["data_dir_json"],
        TEST_CONFIG["labels"]
    )

    if len(test_data) == 0:
        return

    id2label = {i: label for i, label in enumerate(TEST_CONFIG["labels"])}

    print("Starting inference...")

    for item in tqdm(test_data, desc="Processing"):
        image = Image.open(item["image_path"]).convert("RGB")

        clamped_bboxes = [
            [max(0, min(1000, b)) for b in box]
            for box in item["bboxes"]
        ]

        encoding = processor(
            image,
            item["words"],
            boxes=clamped_bboxes,
            word_labels=item["labels"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        inputs = {k: v.to(runtime_device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = model(**inputs)

            if logits.shape[1] > inputs["labels"].shape[1]:
                logits = logits[:, :inputs["labels"].shape[1], :]

            preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        word_ids = encoding.word_ids()
        predicted_roles_map = {}

        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in predicted_roles_map:
                pred_id = preds[idx]
                predicted_roles_map[word_idx] = id2label[pred_id]

        text_roles_output = []
        original_blocks = item["original_blocks"]

        for i, block in enumerate(original_blocks):
            role = predicted_roles_map.get(i, "other").lower()
            text_roles_output.append({
                "id": block["id"],
                "role": role
            })

        final_json = {
            "task3": {
                "input": {
                    "task1_output": {"chart_type": item["chart_type"]},
                    "task2_output": {"text_blocks": original_blocks}
                },
                "output": {"text_roles": text_roles_output}
            }
        }

        output_path = os.path.join(TEST_CONFIG["output_dir"], item["id"])

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)

        vis_filename = os.path.splitext(item["id"])[0] + "_vis.png"
        vis_path = os.path.join(vis_dir, vis_filename)

        visualize_result(
            image_path=item["image_path"],
            original_blocks=original_blocks,
            predicted_roles_map=predicted_roles_map,
            output_path=vis_path
        )

    print("Finished. Check output directory and visualization folder.")


if __name__ == "__main__":
    main()