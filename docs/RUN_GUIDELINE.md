

# Project Run Guide

## 1. Environment Requirements

* Python `3.10+` (recommended `3.10` or `3.11`)
* Windows or Linux
* For GPU: compatible NVIDIA driver + CUDA

## 2. Quick Setup

From the project root directory `Chart_InforX`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If running Paddle with GPU:

* `requirements.txt` currently uses `paddlepaddle` (CPU).
* Replace it with `paddlepaddle-gpu` according to the comment in the file and reinstall.

## 3. Prepare Model Weights

Ensure the following files/folders exist:

```text
weights/
  yolo_text.pt
  best_det.pt
  yolo_elements.pt
  checkpoint-10000/   # extracted LayoutLMv3 model folder
```

Note:

* If you only have `checkpoint-10000.rar`, extract it to:

```
weights/checkpoint-10000
```

## 4. Path Configuration

All paths are configured in:

* `src/config.py`

Current main configs:

* `Task1_detection`
* `Task1_recognize`
* `Task2_role_classifier`
* `Task3_axis_analysis`
* `Task4_legend_analysis`
* `Task5_detect_extraction`

Defaults:

* Input image: `data/sample_images`
* Output: `data/pipeline_outputs/...`

To change input data, modify:

* `Task1_detection["input"]`
* `Task1_recognize["input"]`
* `Task2_role_classifier["data_dir_images"]`
* `Task5_detect_extraction["input_images"]`

## 5. Run Pipeline via Command Line

### Method 1: Run Full Pipeline

```powershell
python src/pipeline.py
```

Execution order:

1. `text_detector.py` (Task1 detection)
2. `text_recognizer.py` (Task1 recognize)
3. `role_classifier.py` (Task2 role classifier)
4. `bar_detection_extraction.py` (Task5 detect + extraction, also outputs Task3/Task4)

### Method 2: Run Step-by-Step

```powershell
python src/text_detector.py
python src/text_recognizer.py
python src/role_classifier.py
python src/bar_detection_extraction.py
```

## 6. Run Streamlit App

```powershell
streamlit run app.py
```

Then:

1. Upload chart image
2. Click `Run Extraction`
3. View and download results in UI

## 7. Output Locations

Default output directory:

```text
data/pipeline_outputs/
  task1_detection/
  task1_recognize/
  task2_role_classifier/
  task3_axis_analysis/
  task4_legend_analysis/
  task5_detect_extraction/
    result.csv
    result.json
```

## 8. Common Errors and Fixes

### `ModuleNotFoundError: timm`

```powershell
pip install timm==1.0.22
```

### Weights Not Found

Check:

* File names in `weights/`
* Paths in `src/config.py`

### CUDA Error

If no GPU or CUDA mismatch:

* Change `device` in config to `"cpu"`
* Install CPU versions of torch/paddle

## 9. Quick Smoke Test

Place 1–2 images into:

```
data/sample_images
```

Then run:

```powershell
python src/pipeline.py
```

If successful, these files will be created:

* `data/pipeline_outputs/task5_detect_extraction/result.csv`
* `data/pipeline_outputs/task5_detect_extraction/result.json`
