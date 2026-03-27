from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
WEIGHTS_ROOT = PROJECT_ROOT / "weights"
PIPELINE_ROOT = DATA_ROOT / "pipeline_outputs"

DEFAULT_IMAGE_DIR = DATA_ROOT / "sample_images"
TASK1_DETECTION_OUT = PIPELINE_ROOT / "task1_detection"
TASK1_RECOGNIZE_OUT = PIPELINE_ROOT / "task1_recognize"
TASK2_ROLE_OUT = PIPELINE_ROOT / "task2_role_classifier"
TASK3_AXIS_OUT = PIPELINE_ROOT / "task3_axis_analysis"
TASK4_LEGEND_OUT = PIPELINE_ROOT / "task4_legend_analysis"
TASK5_EXTRACTION_OUT = PIPELINE_ROOT / "task5_detect_extraction"

model_path_layoutlmv3 = str(WEIGHTS_ROOT / "checkpoint-10000")
model_path_yolo = str(WEIGHTS_ROOT / "yolo_elements.pt")
model_path_yolo_text = str(WEIGHTS_ROOT / "yolo_text.pt")
model_path_yolo_recognize = str(WEIGHTS_ROOT / "best_det.pt")

# Requested config names.
Task1_detection = {
    "input": str(DEFAULT_IMAGE_DIR),
    "output": str(TASK1_DETECTION_OUT),
    "weight": model_path_yolo_text,
}

Task1_recognize = {
    "input": str(DEFAULT_IMAGE_DIR),
    "input_json": str(TASK1_DETECTION_OUT),
    "output": str(TASK1_RECOGNIZE_OUT),
    "weight": model_path_yolo_recognize,
}

Task2_role_classifier = {
    "model_path": model_path_layoutlmv3,
    "data_dir_images": str(DEFAULT_IMAGE_DIR),
    "data_dir_json": str(TASK1_RECOGNIZE_OUT),
    "labels": [
        "CHART_TITLE",
        "LEGEND_TITLE",
        "LEGEND_LABEL",
        "AXIS_TITLE",
        "TICK_LABEL",
        "TICK_GROUPING",
        "MARK_LABEL",
        "VALUE_LABEL",
        "OTHER",
    ],
    "device": "cuda",
    "output_dir": str(TASK2_ROLE_OUT),
}

Task3_axis_analysis = {
    "input_images": str(DEFAULT_IMAGE_DIR),
    "input_json": str(TASK2_ROLE_OUT),
    "yolo_weight": model_path_yolo,
    "output_json": str(TASK3_AXIS_OUT),
}

Task4_legend_analysis = {
    "input_images": str(DEFAULT_IMAGE_DIR),
    "input_json": str(TASK2_ROLE_OUT),
    "yolo_weight": model_path_yolo,
    "feature_backbone": "resnet50",
    "output_json": str(TASK4_LEGEND_OUT),
}

Task5_detect_extraction = {
    "input_images": str(DEFAULT_IMAGE_DIR),
    "input_json": str(TASK2_ROLE_OUT),
    "yolo_weight": model_path_yolo,
    "feature_backbone": "resnet50",
    "device": "cuda",
    "output_dir": str(TASK5_EXTRACTION_OUT),
    "output_json": str(TASK5_EXTRACTION_OUT / "result.json"),
    "output_csv": str(TASK5_EXTRACTION_OUT / "result.csv"),
    "axis_output_json": str(TASK3_AXIS_OUT),
    "legend_output_json": str(TASK4_LEGEND_OUT),
}


def return_task1_detection_config():
    return dict(Task1_detection)


def return_task1_recognize_config():
    return dict(Task1_recognize)


def return_task2_role_classifier_config():
    return dict(Task2_role_classifier)


def return_task3_axis_analysis_config():
    return dict(Task3_axis_analysis)


def return_task4_legend_analysis_config():
    return dict(Task4_legend_analysis)


def return_task5_detect_extraction_config():
    return dict(Task5_detect_extraction)


# Backward compatibility for existing modules/app.
Dataset_Image = Task1_detection["input"]
Output_Json_Task_2_1 = Task1_detection["output"]
Output_Json_Task_2_2 = Task1_recognize["output"]
Output_Json_Task_2 = Output_Json_Task_2_2
Output_Json_Task_3 = Task2_role_classifier["output_dir"]
Output_Json_Task_4 = Task5_detect_extraction["output_dir"]
Output_Excel_Task_4 = Task5_detect_extraction["output_csv"]

Task2_1Config = Task1_detection
Task2Config = Task1_recognize
Task3Config = Task2_role_classifier
Task4Config = Task5_detect_extraction


def returnTestTask2_1_Config():
    return return_task1_detection_config()


def returnTestTask2_2_Config():
    return return_task1_recognize_config()


def returnTestTask3_Config():
    return return_task2_role_classifier_config()


def returnTestTask4_Config():
    return return_task5_detect_extraction_config()


def returnTestTask2_Config():
    return return_task1_recognize_config()
