# ICPR 2022 Chart-Info Dataset
## 📌 Dataset Overview
This project uses the publicly available dataset from the **ICPR 2022 Chart-Info Challenge**. The dataset was created to facilitate research in extracting data from scientific charts.

## 📊 Content & Structure
The dataset contains thousands of chart images sourced from PubMed Central (PMC). The chart images are accompanied by rich annotations, making it a benchmark for multiple tasks:
1. Chart Classification (e.g., Vertical Bar, Line, Pie, Scatter).
2. Text Detection and Recognition.
3. Text Role Classification (Titles, Labels, Ticks).
4. Axis Analysis
5. Legend Analysis
6. Data Extraction
    a. Plot Element Detection/Classification
    b. Raw Data Extractio

## ✂️ Preprocessing for This Project
In this specific project, our focus is **exclusively on Vertical Bar Charts**. 
- We extracted filtering logic to bypass the Task 1 (Chart Classification) and assumed the input image is a vertical bar chart.
- The dataset images (`data/sample_images/`) contain test samples used in our pipeline.

For more details about the original data, visit the [official ICPR 2022 Chart-Info website](https://chartinfo.github.io/).
