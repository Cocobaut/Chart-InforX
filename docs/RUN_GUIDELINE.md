# Huong Dan Chay Du An


## 1. Yeu cau moi truong

- Python `3.10+` (khuyen nghi `3.10` hoac `3.11`)
- Windows/Linux deu duoc
- Neu chay GPU: NVIDIA driver + CUDA phu hop

## 2. Cai dat nhanh

Tu thu muc goc project `Chart_InforX`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu ban chay GPU voi Paddle:
- Trong `requirements.txt` dang de `paddlepaddle` (CPU).
- Doi sang `paddlepaddle-gpu` theo dong comment trong file va cai lai.

## 3. Chuan bi model weights

Dam bao cac file/thu muc sau ton tai:

```text
weights/
  yolo_text.pt
  best_det.pt
  yolo_elements.pt
  checkpoint-10000/   # thu muc model LayoutLMv3 da extract
```

Luu y:
- Neu ban chi co file nen `checkpoint-10000.rar` thi can giai nen thanh thu muc `weights/checkpoint-10000`.

## 4. Cau hinh duong dan

Tat ca duong dan cau hinh nam trong file:
- `src/config.py`

Cac config chinh hien tai:
- `Task1_detection`
- `Task1_recognize`
- `Task2_role_classifier`
- `Task3_axis_analysis`
- `Task4_legend_analysis`
- `Task5_detect_extraction`

Mac dinh:
- Input image: `data/sample_images`
- Output: `data/pipeline_outputs/...`

Neu ban muon thay data dau vao, sua:
- `Task1_detection["input"]`
- `Task1_recognize["input"]`
- `Task2_role_classifier["data_dir_images"]`
- `Task5_detect_extraction["input_images"]`

## 5. Chay pipeline bang command line

### Cach 1: Chay toan bo pipeline

```powershell
python src/pipeline.py
```

Pipeline se chay theo thu tu:
1. `text_detector.py` (Task1 detection)
2. `text_recognizer.py` (Task1 recognize)
3. `role_classifier.py` (Task2 role classifier)
4. `bar_detection_raw_data_extraction.py` (Task5 detect + extraction, dong thoi xuat ket qua Task3/Task4)

### Cach 2: Chay tung buoc

```powershell
python src/text_detector.py
python src/text_recognizer.py
python src/role_classifier.py
python src/bar_detection_raw_data_extraction.py
```

## 6. Chay app Streamlit

```powershell
streamlit run app.py
```

Sau do:
1. Upload anh chart
2. Bam `Run Extraction`
3. Xem va tai ket qua tren giao dien

## 7. Vi tri ket qua output

Du lieu output mac dinh:

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

## 8. Loi thuong gap va cach xu ly

### Loi `ModuleNotFoundError: timm`

```powershell
pip install timm==1.0.22
```

### Loi khong tim thay weights

Kiem tra lai:
- Ten file trong `weights/`
- Duong dan trong `src/config.py`

### Loi CUDA

Neu may khong co GPU hoac mismatch CUDA:
- Doi `device` trong config ve `"cpu"`
- Cai ban CPU cho torch/paddle

## 9. Smoke test nhanh

De test nhanh, dat 1-2 anh vao `data/sample_images`, sau do chay:

```powershell
python src/pipeline.py
```

Neu thanh cong, file sau se duoc tao:
- `data/pipeline_outputs/task5_detect_extraction/result.csv`
- `data/pipeline_outputs/task5_detect_extraction/result.json`
