# 🧠 Methodology

This project translates unstructured Vertical Bar Chart images into structured numerical data through a systematic multi-stage Deep Learning pipeline. By focusing exclusively on Vertical Bar Charts, we bypass the need for an initial chart classification stage.

Below is the detailed breakdown of the 4 core modules:

## 1. Text Detection (`src/text_detector.py`)
- **Objective:** Find all text instances in the chart.
- **Model Used:** YOLOv8-OBB (Oriented Bounding Boxes).
- **Process:** The graph image is passed through the YOLO model to detect bounding boxes around any text (including tilted text). It outputs the coordinates for each detection.

## 2. Text Recognition (`src/text_recognizer.py`)
- **Objective:** Read the text from the detected cropped regions.
- **Model Used:** PaddleOCR (v3.x).
- **Process:** We use PaddleOCR strictly for recognition (since detection is handled by YOLO OBB). The image crops are parsed to text strings and associated with their respective bounding box JSON data.

## 3. Text Role Classification (`src/role_classifier.py`)
- **Objective:** Determine the semantic function of the extracted text (e.g., Chart Title, Axis Title, Tick Label).
- **Model Used:** LayoutLMv3.
- **Process:** LayoutLMv3 is a multimodal transformer. We feed the text strings, bounding box coordinates, and visual features into this model. It classifies each piece of text into one of 9 predefined classes (like `AXIS_TITLE`, `TICK_LABEL`, `LEGEND_LABEL`).

## 4. Full Data Extraction (`src/data_extractor.py`)
This final module encompasses three crucial sub-tasks:
1. **Axis Analysis:** Uses geometric mapping to associate `TICK_LABEL` values with the X/Y axes and calculates the **Pixel-to-Value ratio**.
2. **Legend Analysis:** Uses the Hungarian Algorithm (`linear_sum_assignment`) to match Legend labels with their corresponding color patches.
3. **Bar Extraction:**
   - Detects the graphical column bars using **YOLOv8s**.
   - Uses **ResNet50 / timm** as a backbone to extract visual embeddings from both Legends and Bars.
   - Computes Cosine/L2 distances to assign each Bar to an appropriate Legend class.
   - Projects the Bar height using the Pixel-to-Value ratio to get the actual float values and dumps them into a CSV file.