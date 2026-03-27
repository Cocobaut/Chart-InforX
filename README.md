<h1 align="center">📊 Automated Bar Chart Information Extraction</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=Streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-yellow.svg" alt="YOLO">
</div>

<br>

This project automates the extraction of structural data (CSV/Excel) from unstructured **Vertical Bar Chart** images. It leverages a Multi-stage Deep Learning pipeline combining Computer Vision (CV) and Natural Language Processing (NLP) techniques, heavily inspired by the ICPR 2022 Chart-Info Challenge.

## ✨ Features
* **Text Detection & Recognition:** Accurate text boundary detection (YOLO OBB) and text-reading (PaddleOCR).
* **Multimodal Role Classification:** Understands whether a text is a title, an axis label, or a legend using **LayoutLMv3**.
* **Precise Math Mapping:** Calculates Pixel-to-Value ratios and matches color legends to bars via embedding distances (**ResNet50**).
* **Interactive UI:** Built-in **Streamlit** user interface for easy uploading and result visualization.

## 📂 Project Structure
```text
Chart_InforX/
├── app.py                   # Streamlit Web UI
├── src/                     # Core Deep Learning Pipeline
│   ├── pipeline.py          # Runner linking all models
│   ├── text_detector.py     # Module for detecting text (YOLO)
│   ├── text_recognizer.py   # Module for reading text (PaddleOCR)
│   ├── role_classifier.py   # Module for text roles (LayoutLMv3)
│   └── data_extractor.py    # Module for mapping axes & calculating values
├── docs/                    # Deep dive documentation
│   ├── METHODOLOGY.md       # Explanation of pipeline & models
│   └── DATASET.md           # Dataset origin and constraints
├── data/                    # Sample images & temporary app data
└── weights/                 # Model checkpoints (Download instructions) 
```

## 🚀 Installation

**1. Clone the repository / Setup Environment:**
```bash
git clone <your-repo-link>
cd Chart_InforX
python -m venv venv
# Activate your venv
pip install -r requirements.txt
```

**2. Download Pre-trained Weights:**
Because models (YOLO, LayoutLMv3, ResNet) exceed GitHub file limits, please check the instructions in `weights/download_weights.md` to get the necessary `.pt` checkpoints.

**3. Run the App:**
```bash
streamlit run app.py
```

## 📖 Learn More
* Want to understand the step-by-step algorithms used? Check out [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
* Want to know about the ICPR 2022 dataset used for training? Check out [docs/DATASET.md](docs/DATASET.md)

## 👥 Contributors
**Instructors:** Mai Xuân Toàn, Trần Tuấn Anh, Huỳnh Văn Thống, Trần Hồng Tài.

**Students:**
* Lê Trần Tấn Phát (2312580)
* Bùi Ngọc Phúc (2312665)
* Nguyễn Hồ Quang Khải (2352538)


