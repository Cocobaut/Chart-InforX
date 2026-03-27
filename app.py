import os
import sys
import shutil
import time
from pathlib import Path
from typing import List
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import json

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = BASE_DIR / "src"

if str(SUBMISSION_DIR) not in sys.path:
    sys.path.append(str(SUBMISSION_DIR))

os.chdir(BASE_DIR)

import config  # noqa: E402

# --- GLOBAL CONFIGURATION ---
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def get_result_csv_path() -> Path:
    return Path(config.Task5_detect_extraction["output_csv"]).resolve()


def get_result_paper_json_path() -> Path:
    configured = config.Task5_detect_extraction.get("output_paper_json")
    if configured:
        return Path(configured).resolve()
    return (get_result_csv_path().parent / "result_paper_format.json").resolve()


def get_result_paper_txt_path() -> Path:
    configured = config.Task5_detect_extraction.get("output_paper_txt")
    if configured:
        return Path(configured).resolve()
    return (get_result_csv_path().parent / "result_paper_format.txt").resolve()


def _format_paper_record(record: dict) -> str:
    return "\n".join(
        [
            f"file name      : {record.get('file name', '')}",
            f"doi            : {repr(record.get('doi', []))}",
            f"x-text         : {repr(record.get('x-text', []))}",
            f"x-labels       : {repr(record.get('x-labels', []))}",
            f"y-text         : {repr(record.get('y-text', []))}",
            f"y-labels       : {repr(record.get('y-labels', []))}",
            f"legends        : {repr(record.get('legends', []))}",
            f"data           : {repr(record.get('data', {}))}",
        ]
    )


def render_paper_format_output_section():
    paper_json_path = get_result_paper_json_path()
    paper_txt_path = get_result_paper_txt_path()
    if not paper_json_path.exists() and not paper_txt_path.exists():
        return

    st.markdown("---")
    st.markdown(
        """
        <div class="phase-header" style="margin-top:8px">
            <div class="phase-badge done">✓</div>
            <div>
                <div class="phase-title">Paper-format Output</div>
                <div class="phase-desc">Structured summary with file name / axis texts / labels / legends / data.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    payload = {}
    records = []
    if paper_json_path.exists():
        try:
            with open(paper_json_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                raw_records = payload.get("records", [])
                if isinstance(raw_records, list):
                    records = raw_records
        except Exception as exc:
            st.warning(f"Could not parse paper-format JSON: {exc}")

    preview_tab, raw_tab = st.tabs(["Preview", "Raw Files"])
    with preview_tab:
        if records:
            st.caption(f"{len(records)} record(s)")
            for idx, record in enumerate(records):
                file_name = record.get("file name", f"record_{idx + 1}")
                with st.expander(f"🧾 {file_name}", expanded=(idx == 0)):
                    st.code(_format_paper_record(record), language="text")
        elif paper_txt_path.exists():
            try:
                text_content = paper_txt_path.read_text(encoding="utf-8")
                st.code(text_content, language="text")
            except Exception as exc:
                st.warning(f"Could not read paper-format text file: {exc}")
        else:
            st.info("No paper-format content found yet.")

    with raw_tab:
        dl_col1, dl_col2 = st.columns(2)
        if paper_json_path.exists():
            json_bytes = paper_json_path.read_bytes()
            with dl_col1:
                st.download_button(
                    "⬇ Download paper JSON",
                    data=json_bytes,
                    file_name=paper_json_path.name,
                    mime="application/json",
                    key="dl_paper_json",
                    use_container_width=True,
                )
            with st.expander("Show JSON content", expanded=False):
                if payload:
                    st.json(payload)
                else:
                    st.code(json_bytes.decode("utf-8", errors="replace"), language="json")

        if paper_txt_path.exists():
            text_content = paper_txt_path.read_text(encoding="utf-8")
            with dl_col2:
                st.download_button(
                    "⬇ Download paper TXT",
                    data=text_content,
                    file_name=paper_txt_path.name,
                    mime="text/plain",
                    key="dl_paper_txt",
                    use_container_width=True,
                )
            with st.expander("Show TXT content", expanded=False):
                st.code(text_content, language="text")


# --- PAGE SETUP ---
st.set_page_config(
    page_title="ChartIQ · Bar Chart Extraction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- STYLING ---
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* ── Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

        /* ── CSS Variables ── */
        :root {
            --blue-600:   #1A56DB;
            --blue-500:   #3F83F8;
            --blue-50:    #EBF5FF;
            --slate-900:  #0F172A;
            --slate-700:  #334155;
            --slate-500:  #64748B;
            --slate-300:  #CBD5E1;
            --slate-100:  #F1F5F9;
            --slate-50:   #F8FAFC;
            --white:      #FFFFFF;
            --green-500:  #10B981;
            --green-50:   #ECFDF5;
            --amber-500:  #F59E0B;
            --amber-50:   #FFFBEB;
            --red-500:    #EF4444;
            --red-50:     #FEF2F2;
            --radius-sm:  6px;
            --radius-md:  10px;
            --radius-lg:  16px;
            --shadow-sm:  0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04);
            --shadow-md:  0 4px 12px rgba(15,23,42,0.10), 0 2px 4px rgba(15,23,42,0.06);
            --shadow-lg:  0 10px 30px rgba(15,23,42,0.12), 0 4px 8px rgba(15,23,42,0.06);
        }

        /* ── Base Reset ── */
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif !important;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--slate-50);
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: var(--white) !important;
            border-right: 1px solid var(--slate-300) !important;
            padding-top: 0 !important;
        }
        [data-testid="stSidebar"] * {
            color: var(--slate-700) !important;
        }
        [data-testid="stSidebar"] .sidebar-logo {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 24px 20px 20px;
            border-bottom: 1px solid var(--slate-100);
            margin-bottom: 20px;
        }
        [data-testid="stSidebar"] .sidebar-logo .logo-icon {
            width: 36px; height: 36px;
            background: var(--blue-600);
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px;
        }
        [data-testid="stSidebar"] .sidebar-logo .logo-text {
            font-weight: 700;
            font-size: 17px;
            color: var(--slate-900) !important;
            letter-spacing: -0.3px;
        }
        [data-testid="stSidebar"] .sidebar-logo .logo-sub {
            font-size: 11px;
            color: var(--slate-500) !important;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* ── Global Text ── */
        p, div, label, li, span {
            color: var(--slate-700);
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--slate-900) !important;
            font-family: 'DM Sans', sans-serif !important;
        }

        /* ── Buttons ── */
        div.stButton > button {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            border-radius: var(--radius-sm) !important;
            border: none !important;
            transition: all 0.15s ease !important;
            letter-spacing: -0.1px !important;
        }
        div.stButton > button[kind="primary"] {
            background: var(--blue-600) !important;
            color: var(--white) !important;
            box-shadow: var(--shadow-sm) !important;
            padding: 10px 20px !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #1648C8 !important;
            box-shadow: var(--shadow-md) !important;
            transform: translateY(-1px) !important;
        }
        div.stButton > button[kind="secondary"] {
            background: var(--white) !important;
            color: var(--slate-700) !important;
            border: 1px solid var(--slate-300) !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            background: var(--slate-50) !important;
            border-color: var(--slate-400) !important;
        }

        /* ── Download Button ── */
        div.stDownloadButton > button {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            background: var(--white) !important;
            color: var(--blue-600) !important;
            border: 1.5px solid var(--blue-500) !important;
            border-radius: var(--radius-sm) !important;
            transition: all 0.15s ease !important;
        }
        div.stDownloadButton > button:hover {
            background: var(--blue-50) !important;
        }

        /* ── Hero Section ── */
        .hero-section {
            background: var(--white);
            border-bottom: 1px solid var(--slate-200);
            padding: 32px 40px 28px;
            margin: -1rem -1rem 32px -1rem;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--blue-50);
            color: var(--blue-600);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            padding: 4px 10px;
            border-radius: 99px;
            margin-bottom: 12px;
        }
        .hero-title {
            font-size: 28px !important;
            font-weight: 700 !important;
            color: var(--slate-900) !important;
            letter-spacing: -0.6px;
            margin: 0 0 6px 0 !important;
            line-height: 1.25;
        }
        .hero-subtitle {
            font-size: 15px;
            color: var(--slate-500);
            font-weight: 400;
            margin: 0;
        }

        /* ── Phase Headers ── */
        .phase-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }
        .phase-badge {
            width: 28px; height: 28px;
            background: var(--blue-600);
            color: white;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 13px;
            font-weight: 700;
            flex-shrink: 0;
        }
        .phase-badge.done {
            background: var(--green-500);
        }
        .phase-badge.pending {
            background: var(--slate-300);
        }
        .phase-title {
            font-size: 17px !important;
            font-weight: 600 !important;
            color: var(--slate-900) !important;
            margin: 0 !important;
        }
        .phase-desc {
            font-size: 13px;
            color: var(--slate-500);
            margin: 0;
        }

        /* ── Upload Zone ── */
        .upload-zone-wrapper {
            background: var(--white);
            border: 2px dashed var(--slate-300);
            border-radius: var(--radius-lg);
            padding: 40px 32px;
            text-align: center;
            transition: border-color 0.2s ease;
        }
        .upload-zone-wrapper:hover {
            border-color: var(--blue-500);
        }
        .upload-icon {
            font-size: 40px;
            margin-bottom: 12px;
            display: block;
        }
        .upload-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--slate-900);
            margin-bottom: 4px;
        }
        .upload-hint {
            font-size: 13px;
            color: var(--slate-500);
            margin-bottom: 0;
        }
        [data-testid="stFileUploader"] {
            background: transparent !important;
            border: none !important;
        }
        [data-testid="stFileUploader"] > div {
            background: transparent !important;
        }

        /* ── Image Grid Cards ── */
        .img-card {
            background: var(--white);
            border: 1px solid var(--slate-200);
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        .img-card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }
        .img-card-body { padding: 8px 10px; }
        .img-card-name {
            font-size: 12px;
            font-weight: 500;
            color: var(--slate-700);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .img-card-meta {
            font-size: 11px;
            color: var(--slate-400);
            font-family: 'DM Mono', monospace;
        }

        /* ── Step Tracker ── */
        .step-tracker {
            background: var(--white);
            border: 1px solid var(--slate-200);
            border-radius: var(--radius-lg);
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-sm);
        }
        .step-tracker-title {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--slate-500);
            margin-bottom: 20px;
        }
        .steps-row {
            display: flex;
            align-items: flex-start;
            gap: 0;
        }
        .step-item {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
        }
        .step-item:not(:last-child)::after {
            content: '';
            position: absolute;
            top: 14px;
            left: calc(50% + 14px);
            right: calc(-50% + 14px);
            height: 2px;
            background: var(--slate-200);
            z-index: 0;
        }
        .step-item.done:not(:last-child)::after {
            background: var(--green-500);
        }
        .step-item.active:not(:last-child)::after {
            background: linear-gradient(to right, var(--blue-500), var(--slate-200));
        }
        .step-dot {
            width: 28px; height: 28px;
            border-radius: 50%;
            background: var(--slate-200);
            border: 2px solid var(--slate-200);
            display: flex; align-items: center; justify-content: center;
            font-size: 12px; font-weight: 700;
            color: var(--slate-500);
            position: relative; z-index: 1;
            transition: all 0.3s ease;
        }
        .step-dot.done {
            background: var(--green-500);
            border-color: var(--green-500);
            color: white;
        }
        .step-dot.active {
            background: var(--blue-600);
            border-color: var(--blue-600);
            color: white;
            box-shadow: 0 0 0 4px var(--blue-50);
            animation: pulse-blue 1.5s infinite;
        }
        @keyframes pulse-blue {
            0%, 100% { box-shadow: 0 0 0 4px var(--blue-50); }
            50%       { box-shadow: 0 0 0 8px rgba(63,131,248,0.15); }
        }
        .step-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--slate-500);
            margin-top: 8px;
            text-align: center;
            line-height: 1.3;
        }
        .step-label.done  { color: var(--green-500); }
        .step-label.active { color: var(--blue-600); font-weight: 600; }

        /* ── Metrics Cards ── */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 28px;
        }
        .metric-card {
            background: var(--white);
            border: 1px solid var(--slate-200);
            border-radius: var(--radius-md);
            padding: 20px 22px;
            box-shadow: var(--shadow-sm);
        }
        .metric-icon {
            font-size: 22px;
            margin-bottom: 10px;
            display: block;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--slate-900);
            letter-spacing: -1px;
            line-height: 1;
            margin-bottom: 4px;
        }
        .metric-label {
            font-size: 13px;
            color: var(--slate-500);
            font-weight: 400;
        }
        .metric-card.blue  { border-top: 3px solid var(--blue-600); }
        .metric-card.green { border-top: 3px solid var(--green-500); }
        .metric-card.amber { border-top: 3px solid var(--amber-500); }

        /* ── Result Cards ── */
        .result-card {
            background: var(--white);
            border: 1px solid var(--slate-200);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            margin-bottom: 20px;
        }
        .result-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 20px;
            border-bottom: 1px solid var(--slate-100);
            background: var(--slate-50);
        }
        .result-card-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--slate-900);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .result-badge {
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 99px;
        }
        .result-badge.success {
            background: var(--green-50);
            color: var(--green-500);
        }
        .result-badge.empty {
            background: var(--slate-100);
            color: var(--slate-500);
        }
        .result-card-body { padding: 20px; }

        /* ── Expander Override ── */
        [data-testid="stExpander"] {
            background: var(--white);
            border: 1px solid var(--slate-200) !important;
            border-radius: var(--radius-lg) !important;
            box-shadow: var(--shadow-sm) !important;
            margin-bottom: 12px !important;
        }
        [data-testid="stExpander"] summary {
            font-weight: 600 !important;
            font-size: 14px !important;
            color: var(--slate-900) !important;
            padding: 14px 20px !important;
        }

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--slate-200) !important;
            border-radius: var(--radius-md) !important;
        }

        /* ── Alert ── */
        [data-testid="stAlert"] {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--slate-200) !important;
            font-size: 14px !important;
        }

        /* ── Spinner ── */
        [data-testid="stSpinner"] > div {
            color: var(--blue-600) !important;
        }

        /* ── Divider ── */
        hr {
            border-color: var(--slate-200) !important;
            margin: 24px 0 !important;
        }

        /* ── Checkbox ── */
        [data-testid="stCheckbox"] label {
            font-size: 14px !important;
            color: var(--slate-700) !important;
        }

        /* ── Section Label ── */
        .section-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            color: var(--slate-400);
            margin-bottom: 10px;
            display: block;
        }

        /* ── Status Log ── */
        .log-box {
            background: var(--slate-900);
            border-radius: var(--radius-md);
            padding: 16px 18px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            line-height: 1.7;
            color: #94A3B8;
            max-height: 200px;
            overflow-y: auto;
        }
        .log-box .log-ok    { color: #34D399; }
        .log-box .log-info  { color: #60A5FA; }
        .log-box .log-warn  { color: #FBBF24; }
        .log-box .log-err   { color: #F87171; }

        /* ── Empty State ── */
        .empty-state {
            text-align: center;
            padding: 60px 40px;
            background: var(--white);
            border: 1px solid var(--slate-200);
            border-radius: var(--radius-lg);
        }
        .empty-state-icon { font-size: 48px; margin-bottom: 16px; display: block; }
        .empty-state-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--slate-900);
            margin-bottom: 6px;
        }
        .empty-state-desc { font-size: 14px; color: var(--slate-500); }

        /* ── Toast Override ── */
        [data-testid="stToast"] {
            border-radius: var(--radius-md) !important;
            font-size: 13px !important;
        }

        /* ── Progress Bar ── */
        [data-testid="stProgressBar"] > div > div {
            background: var(--blue-600) !important;
            border-radius: 99px !important;
        }
        [data-testid="stProgressBar"] > div {
            background: var(--slate-200) !important;
            border-radius: 99px !important;
            height: 6px !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--slate-100); }
        ::-webkit-scrollbar-thumb { background: var(--slate-300); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--slate-400); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── SESSION STATE ──────────────────────────────────────────────────────────────
def ensure_session_state():
    defaults = {
        "session_id": str(time.time()),
        "mock_mode": False,
        "pipeline_ran": False,
        "pipeline_steps": {},   # step_name -> "done" | "active" | "pending" | "error"
        "pipeline_log": [],
        "processing_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
            if k == "session_id":
                clear_session_data()


def clear_session_data():
    if TEMP_UPLOAD_DIR.exists():
        shutil.rmtree(TEMP_UPLOAD_DIR)
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    output_dirs = [
        Path(config.Task1_detection["output"]),
        Path(config.Task1_recognize["output"]),
        Path(config.Task2_role_classifier["output_dir"]),
        Path(config.Task3_axis_analysis["output_json"]),
        Path(config.Task4_legend_analysis["output_json"]),
        Path(config.Task5_detect_extraction["output_dir"]),
    ]
    for d in output_dirs:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    st.session_state["pipeline_ran"] = False
    st.session_state["pipeline_steps"] = {}
    st.session_state["pipeline_log"] = []
    st.session_state["processing_time"] = None


def ensure_folders():
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Path(config.Task1_detection["input"]).mkdir(parents=True, exist_ok=True)
    Path(config.Task1_detection["output"]).mkdir(parents=True, exist_ok=True)
    Path(config.Task1_recognize["output"]).mkdir(parents=True, exist_ok=True)
    Path(config.Task2_role_classifier["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config.Task3_axis_analysis["output_json"]).mkdir(parents=True, exist_ok=True)
    Path(config.Task4_legend_analysis["output_json"]).mkdir(parents=True, exist_ok=True)
    Path(config.Task5_detect_extraction["output_dir"]).mkdir(parents=True, exist_ok=True)


# ── PIPELINE ──────────────────────────────────────────────────────────────────
PIPELINE_STEPS = [
    ("detection",       "Text Detection"),
    ("recognition",     "Text Recognition"),
    ("classification",  "Role Classification"),
    ("extraction",      "Data Extraction"),
]


def add_log(msg: str, level: str = "info"):
    ts = time.strftime("%H:%M:%S")
    st.session_state["pipeline_log"].append((ts, level, msg))


@st.cache_resource
def load_pipeline_modules():
    try:
        import text_detector as t1
        import text_recognizer as t1_rec
        import role_classifier as t2_role
        import bar_detection_raw_data_extraction as t5_extract

        if not hasattr(t1_rec, "original_init_model"):
            t1_rec.original_init_model = t1_rec.init_model
        if not hasattr(t1_rec, "_app_cached_ocr"):
            t1_rec._app_cached_ocr = None

        def init_model_cached():
            if t1_rec._app_cached_ocr is None:
                t1_rec._app_cached_ocr = t1_rec.original_init_model()
            return t1_rec._app_cached_ocr

        t1_rec.init_model = init_model_cached
        return t1, t1_rec, t2_role, t5_extract
    except Exception as e:
        error_str = str(e)
        st.error(f"Failed to load pipeline modules: {e}")
        if "dll" in error_str.lower() or "procedure could not be found" in error_str.lower():
            st.warning("⚠️ PyTorch DLL issue detected.")
            st.code(
                "pip install --force-reinstall torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu118",
                language="bash",
            )
        return None, None, None, None


def run_extraction_pipeline(step_placeholders: dict, progress_bar, status_text):
    ensure_folders()
    start_time = time.time()
    step_keys = [s[0] for s in PIPELINE_STEPS]

    def set_step(key, state):
        st.session_state["pipeline_steps"][key] = state
        _render_step_tracker_content(step_placeholders)

    # ── MOCK MODE ──────────────────────────────────────────────────────────────
    if st.session_state.get("mock_mode", False):
        for i, (key, label) in enumerate(PIPELINE_STEPS):
            set_step(key, "active")
            status_text.markdown(f"<p style='font-size:13px;color:#1A56DB;font-weight:500;margin:0'>⏳ Running: {label}…</p>", unsafe_allow_html=True)
            add_log(f"[MOCK] Running {label}…", "info")
            time.sleep(0.8)
            set_step(key, "done")
            add_log(f"[MOCK] {label} complete.", "ok")
            progress_bar.progress(int((i + 1) / len(PIPELINE_STEPS) * 100))

        t5_output_csv = Path(config.Task5_detect_extraction["output_csv"]).resolve()
        t5_output_csv.parent.mkdir(parents=True, exist_ok=True)
        dummy_df = pd.DataFrame({"Category": ["Mock A", "Mock B", "Mock C"], "Value": [42, 78, 31]})
        dummy_df.to_csv(t5_output_csv, index=False)

        st.session_state["processing_time"] = round(time.time() - start_time, 1)
        st.session_state["pipeline_ran"] = True
        status_text.markdown("<p style='font-size:13px;color:#10B981;font-weight:500;margin:0'>✅ Pipeline completed successfully!</p>", unsafe_allow_html=True)
        add_log("Mock pipeline done.", "ok")
        return

    # ── REAL PIPELINE ──────────────────────────────────────────────────────────
    t1, t1_rec, t2_role, t5_extract = load_pipeline_modules()
    if not all([t1, t1_rec, t2_role, t5_extract]):
        st.error("Pipeline modules failed to load.")
        return

    # Build configs
    def _cfg(fn_new, fn_old, attr_old):
        try:    return getattr(config, fn_new)()
        except: return getattr(config, fn_old)() if hasattr(config, fn_old) else dict(getattr(config, attr_old))

    t1_cfg      = _cfg("return_task1_detection_config",       "returnTestTask2_1_Config", "Task1_detection")
    t1_rec_cfg  = _cfg("return_task1_recognize_config",       "returnTestTask2_Config",   "Task1_recognize")
    t2_role_cfg = _cfg("return_task2_role_classifier_config", "returnTestTask3_Config",   "Task2_role_classifier")
    t5_cfg      = _cfg("return_task5_detect_extraction_config","returnTestTask4_Config",  "Task5_detect_extraction")
    t3_axis_cfg = _cfg("return_task3_axis_analysis_config",   "",                         "Task3_axis_analysis")
    t4_legend_cfg= _cfg("return_task4_legend_analysis_config","",                         "Task4_legend_analysis")

    t1_output      = Path(t1_cfg["output"]).resolve()
    t1_rec_output  = Path(t1_rec_cfg["output"]).resolve()
    t2_role_output = Path(t2_role_cfg["output_dir"]).resolve()
    t3_axis_output = Path(t3_axis_cfg["output_json"]).resolve()
    t4_legend_output= Path(t4_legend_cfg["output_json"]).resolve()
    t5_output_dir  = Path(t5_cfg["output_dir"]).resolve()
    t5_output_csv  = Path(t5_cfg["output_csv"]).resolve()
    t5_output_json = Path(t5_cfg["output_json"]).resolve()

    t1_cfg.update({"input": str(TEMP_UPLOAD_DIR), "output": str(t1_output)})
    t1_rec_cfg.update({"input": str(TEMP_UPLOAD_DIR), "input_json": str(t1_output), "output": str(t1_rec_output)})
    t2_role_cfg.update({"data_dir_images": str(TEMP_UPLOAD_DIR), "data_dir_json": str(t1_rec_output), "output_dir": str(t2_role_output)})
    t3_axis_cfg.update({"input_images": str(TEMP_UPLOAD_DIR), "input_json": str(t2_role_output), "output_json": str(t3_axis_output)})
    t4_legend_cfg.update({"input_images": str(TEMP_UPLOAD_DIR), "input_json": str(t2_role_output), "output_json": str(t4_legend_output)})
    t5_cfg.update({
        "input_images": str(TEMP_UPLOAD_DIR), "input_json": str(t2_role_output),
        "output_dir": str(t5_output_dir), "output_csv": str(t5_output_csv),
        "output_json": str(t5_output_json),
        "output_paper_json": str(t5_output_dir / "result_paper_format.json"),
        "output_paper_txt": str(t5_output_dir / "result_paper_format.txt"),
        "axis_output_json": str(t3_axis_output), "legend_output_json": str(t4_legend_output),
    })

    for cfg_dict, updates in [
        (config.Task1_detection,       t1_cfg),
        (config.Task1_recognize,       t1_rec_cfg),
        (config.Task2_role_classifier, t2_role_cfg),
        (config.Task3_axis_analysis,   t3_axis_cfg),
        (config.Task4_legend_analysis, t4_legend_cfg),
        (config.Task5_detect_extraction, t5_cfg),
    ]:
        cfg_dict.update(updates)

    config.Dataset_Image       = t1_cfg["input"]
    config.Output_Json_Task_2_1= t1_cfg["output"]
    config.Output_Json_Task_2  = t1_rec_cfg["output"]
    config.Output_Json_Task_3  = t2_role_cfg["output_dir"]
    config.Output_Json_Task_4  = t5_cfg["output_dir"]
    config.Output_Excel_Task_4 = t5_cfg["output_csv"]

    t1.Task2_1Config   = t1_cfg
    t1_rec.Task2_Config= t1_rec_cfg
    t2_role.TEST_CONFIG= t2_role_cfg

    steps_fns = [
        ("detection",      t1.main,          t1_output,      "*.json", "Task 1 · Text Detection"),
        ("recognition",    t1_rec.main,      t1_rec_output,  "*.json", "Task 1 · Text Recognition"),
        ("classification", t2_role.main,     t2_role_output, "*.json", "Task 2 · Role Classification"),
        ("extraction",     t5_extract.main,  None,           None,     "Task 5 · Data Extraction"),
    ]

    for i, (key, fn, out_dir, pattern, label) in enumerate(steps_fns):
        try:
            set_step(key, "active")
            status_text.markdown(
                f"<p style='font-size:13px;color:#1A56DB;font-weight:500;margin:0'>⏳ {label}…</p>",
                unsafe_allow_html=True
            )
            add_log(f"Starting {label}…", "info")
            fn()

            if key == "extraction":
                if not t5_output_csv.exists():
                    raise FileNotFoundError("Output CSV not found after extraction.")
            else:
                files = list(out_dir.glob(pattern))
                if not files:
                    raise FileNotFoundError(f"No output files in {out_dir}")
                add_log(f"{label} → {len(files)} file(s) produced.", "ok")

            set_step(key, "done")
            progress_bar.progress(int((i + 1) / len(steps_fns) * 100))
            st.toast(f"✅ {label} complete", icon="✅")

        except Exception as e:
            set_step(key, "error")
            add_log(f"ERROR in {label}: {e}", "err")
            st.error(f"Pipeline failed at **{label}**: {e}")
            status_text.markdown(
                f"<p style='font-size:13px;color:#EF4444;font-weight:500;margin:0'>❌ Failed at {label}.</p>",
                unsafe_allow_html=True
            )
            return

    st.session_state["processing_time"] = round(time.time() - start_time, 1)
    st.session_state["pipeline_ran"] = True
    status_text.markdown(
        "<p style='font-size:13px;color:#10B981;font-weight:500;margin:0'>✅ All steps completed successfully!</p>",
        unsafe_allow_html=True
    )
    add_log("Pipeline completed successfully.", "ok")


# ── UI HELPERS ─────────────────────────────────────────────────────────────────
def draw_ocr_boxes(image_path: Path, json_path: Path) -> Image.Image:
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text_blocks = []
        if isinstance(data, dict):
            if "task2" in data:
                text_blocks = data["task2"].get("output", {}).get("text_blocks", [])
            else:
                text_blocks = data.get("text_blocks", [])
        elif isinstance(data, list):
            text_blocks = data
        for item in text_blocks:
            poly = item.get("polygon")
            if poly:
                points = [(poly["x0"], poly["y0"]), (poly["x1"], poly["y1"]),
                          (poly["x2"], poly["y2"]), (poly["x3"], poly["y3"])]
                draw.polygon(points, outline="#3F83F8", width=2)
        return image
    except Exception:
        return Image.open(image_path)


def save_uploaded_files(uploaded_files):
    ensure_folders()
    for file in uploaded_files:
        dest = TEMP_UPLOAD_DIR / file.name
        with open(dest, "wb") as f:
            f.write(file.getbuffer())


def list_uploaded_images() -> List[Path]:
    if not TEMP_UPLOAD_DIR.exists():
        return []
    return sorted(
        [p for p in TEMP_UPLOAD_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS],
        key=lambda p: p.name,
    )


def _render_step_tracker_content(placeholders: dict):
    steps_state = st.session_state.get("pipeline_steps", {})
    dots_html = ""
    for key, label in PIPELINE_STEPS:
        state = steps_state.get(key, "pending")
        icon  = {"done": "✓", "active": "●", "error": "✗"}.get(state, str(PIPELINE_STEPS.index((key, label)) + 1))
        dots_html += f"""
        <div class="step-item {state}">
            <div class="step-dot {state}">{icon}</div>
            <div class="step-label {state}">{label.replace(' ', '<br>')}</div>
        </div>"""
    html = f"""
    <div class="step-tracker">
        <div class="step-tracker-title">Pipeline Progress</div>
        <div class="steps-row">{dots_html}</div>
    </div>"""
    for ph in placeholders.values():
        ph.markdown(html, unsafe_allow_html=True)


def render_step_tracker():
    placeholder = st.empty()
    steps_state = st.session_state.get("pipeline_steps", {})
    dots_html = ""
    for key, label in PIPELINE_STEPS:
        state = steps_state.get(key, "pending")
        idx   = [s[0] for s in PIPELINE_STEPS].index(key) + 1
        icon  = {"done": "✓", "active": "●", "error": "✗"}.get(state, str(idx))
        dots_html += f"""
        <div class="step-item {state}">
            <div class="step-dot {state}">{icon}</div>
            <div class="step-label {state}">{label.replace(' ', '<br>')}</div>
        </div>"""
    placeholder.markdown(f"""
    <div class="step-tracker">
        <div class="step-tracker-title">Pipeline Progress</div>
        <div class="steps-row">{dots_html}</div>
    </div>""", unsafe_allow_html=True)
    return {"main": placeholder}


def render_metrics(images: List[Path], csv_path: Path, elapsed: float):
    n_images = len(images)
    n_rows = 0
    try:
        df = pd.read_csv(csv_path)
        n_rows = len(df)
    except Exception:
        pass
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card blue">
            <span class="metric-icon">🖼️</span>
            <div class="metric-value">{n_images}</div>
            <div class="metric-label">Charts Processed</div>
        </div>
        <div class="metric-card green">
            <span class="metric-icon">📊</span>
            <div class="metric-value">{n_rows}</div>
            <div class="metric-label">Data Points Extracted</div>
        </div>
        <div class="metric-card amber">
            <span class="metric-icon">⏱️</span>
            <div class="metric-value">{elapsed}s</div>
            <div class="metric-label">Processing Time</div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_log():
    logs = st.session_state.get("pipeline_log", [])
    if not logs:
        return
    lines = ""
    for ts, lvl, msg in logs[-30:]:
        cls = {"ok": "log-ok", "info": "log-info", "warn": "log-warn", "err": "log-err"}.get(lvl, "")
        lines += f'<div class="{cls}">[{ts}] {msg}</div>'
    with st.expander("📋 Pipeline Log", expanded=False):
        st.markdown(f'<div class="log-box">{lines}</div>', unsafe_allow_html=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-icon">📊</div>
            <div>
                <div class="logo-text">ChartIQ</div>
                <div class="logo-sub">v1.0 · Extraction Engine</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Settings
        st.markdown('<span class="section-label">Settings</span>', unsafe_allow_html=True)
        st.session_state["mock_mode"] = st.checkbox(
            "🚀 UI Demo Mode (no model)",
            value=st.session_state.get("mock_mode", False),
            help="Run the pipeline with mock data — no PyTorch required.",
        )

        st.markdown("---")

        # Instructions
        st.markdown('<span class="section-label">How It Works</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:13px;color:#64748B;line-height:1.7">
        <b style="color:#0F172A">① Upload</b> your bar chart images<br>
        <b style="color:#0F172A">② Run</b> the extraction pipeline<br>
        <b style="color:#0F172A">③ Review</b> and download results
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Clear session
        st.markdown('<span class="section-label">Session</span>', unsafe_allow_html=True)
        if st.button("🗑 Clear All Data", type="secondary", use_container_width=True):
            clear_session_data()
            st.rerun()

        st.markdown("---")

        # Pipeline step status in sidebar
        if st.session_state.get("pipeline_ran") or st.session_state.get("pipeline_steps"):
            st.markdown('<span class="section-label">Last Run Status</span>', unsafe_allow_html=True)
            steps_state = st.session_state.get("pipeline_steps", {})
            for key, label in PIPELINE_STEPS:
                state = steps_state.get(key, "pending")
                icon  = {"done": "✅", "active": "🔵", "error": "❌", "pending": "⬜"}.get(state, "⬜")
                color = {"done": "#10B981", "active": "#1A56DB", "error": "#EF4444", "pending": "#94A3B8"}.get(state, "#94A3B8")
                st.markdown(
                    f'<div style="font-size:13px;color:{color};padding:3px 0">{icon} {label}</div>',
                    unsafe_allow_html=True
                )


# ── MAIN CONTENT ───────────────────────────────────────────────────────────────
def render_main():
    # ── Hero ──
    st.markdown("""
    <div class="hero-section">
        <div class="hero-eyebrow">📊 AI-Powered Analysis</div>
        <h1 class="hero-title">Bar Chart Data Extraction</h1>
        <p class="hero-subtitle">Upload bar chart images and automatically extract structured data using computer vision and OCR.</p>
    </div>""", unsafe_allow_html=True)

    # ── Phase 1: Upload ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="phase-header">
        <div class="phase-badge">1</div>
        <div>
            <div class="phase-title">Upload Images</div>
            <div class="phase-desc">Supported formats: PNG, JPG, JPEG, BMP, TIFF</div>
        </div>
    </div>""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop images here or click to browse",
        type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        save_uploaded_files(uploaded_files)

    images = list_uploaded_images()

    if not images:
        st.markdown("""
        <div style="background:var(--white);border:2px dashed #CBD5E1;border-radius:12px;
                    padding:48px 32px;text-align:center;margin-bottom:24px">
            <span style="font-size:40px;display:block;margin-bottom:12px">🖼️</span>
            <div style="font-size:15px;font-weight:600;color:#0F172A;margin-bottom:4px">No images uploaded yet</div>
            <div style="font-size:13px;color:#64748B">Use the uploader above to get started</div>
        </div>""", unsafe_allow_html=True)
    else:
        # Image grid
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin:12px 0 16px">
            <span style="font-size:13px;font-weight:600;color:#0F172A">{len(images)} image{'s' if len(images)>1 else ''} ready</span>
            <span style="background:#ECFDF5;color:#10B981;font-size:11px;font-weight:600;
                         padding:2px 8px;border-radius:99px">● Loaded</span>
        </div>""", unsafe_allow_html=True)

        cols = st.columns(min(4, len(images)))
        for idx, img_path in enumerate(images):
            with cols[idx % 4]:
                img = Image.open(img_path)
                w, h = img.size
                st.image(img, use_container_width=True)
                st.markdown(
                    f'<div style="font-size:11px;color:#64748B;text-align:center;'
                    f'margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                    f'{img_path.name}</div>'
                    f'<div style="font-size:10px;color:#94A3B8;text-align:center;'
                    f'font-family:DM Mono,monospace">{w}×{h}px</div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ── Phase 2: Run Pipeline ──────────────────────────────────────────────────
    st.markdown("""
    <div class="phase-header">
        <div class="phase-badge">2</div>
        <div>
            <div class="phase-title">Run Extraction Pipeline</div>
            <div class="phase-desc">4-step AI pipeline: detection → recognition → classification → extraction</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Step tracker (static display before run)
    tracker_ph = render_step_tracker()

    col_run, col_status = st.columns([2, 5])
    with col_run:
        run_disabled = len(images) == 0
        run_clicked  = st.button(
            "▶  Run Extraction",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
        )
        if run_disabled:
            st.markdown(
                '<div style="font-size:12px;color:#94A3B8;margin-top:4px">Upload images to enable</div>',
                unsafe_allow_html=True
            )

    with col_status:
        status_text = st.empty()

    if run_clicked:
        st.session_state["pipeline_steps"] = {}
        st.session_state["pipeline_log"]   = []
        progress_bar = st.progress(0)
        with st.spinner(""):
            run_extraction_pipeline(tracker_ph, progress_bar, status_text)
        st.rerun()

    # Show log if available
    if st.session_state.get("pipeline_log"):
        render_log()

    st.markdown("---")

    # ── Phase 3: Results ───────────────────────────────────────────────────────
    ph_state = "done" if st.session_state.get("pipeline_ran") else "pending"
    st.markdown(f"""
    <div class="phase-header">
        <div class="phase-badge {'done' if ph_state=='done' else ''}">{'✓' if ph_state=='done' else '3'}</div>
        <div>
            <div class="phase-title">Extraction Results</div>
            <div class="phase-desc">Review extracted data and download per-image CSV files</div>
        </div>
    </div>""", unsafe_allow_html=True)

    result_csv = get_result_csv_path()

    if not result_csv.exists():
        st.markdown("""
        <div class="empty-state">
            <span class="empty-state-icon">📂</span>
            <div class="empty-state-title">No results yet</div>
            <div class="empty-state-desc">Run the pipeline above to see extracted data here.</div>
        </div>""", unsafe_allow_html=True)
        return

    # Metrics row
    elapsed = st.session_state.get("processing_time") or 0
    render_metrics(images, result_csv, elapsed)
    render_paper_format_output_section()

    # Per-image or combined results
    individual_dir = result_csv.parent / "individual_results"
    has_individual = individual_dir.exists() and any(individual_dir.glob("*.csv"))

    if not has_individual:
        # Fallback: combined view
        try:
            df = pd.read_csv(result_csv)
            if df.empty:
                st.info("No data was extracted. The pipeline ran but found no values.")
            else:
                st.warning("⚠️ Showing combined results (older format). Re-run for per-image breakdown.")
                st.dataframe(df, use_container_width=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download All Results", csv_bytes, "results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error loading results: {e}")
        return

    # Per-image cards
    csv_files  = sorted(individual_dir.glob("*.csv"))
    img_map    = {p.stem: p for p in images}
    task2_json_dir = (BASE_DIR / config.Output_Json_Task_2).resolve()

    all_dfs = []
    for csv_path in csv_files:
        stem = csv_path.stem
        try:
            sub_df = pd.read_csv(csv_path)
            all_dfs.append(sub_df)
            n_rows = len(sub_df)
        except Exception:
            sub_df = pd.DataFrame()
            n_rows = 0

        badge_cls  = "success" if n_rows > 0 else "empty"
        badge_text = f"{n_rows} row{'s' if n_rows!=1 else ''}" if n_rows > 0 else "No data"

        with st.expander(f"📄 {stem}", expanded=True):
            # Header inside expander
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
                <span style="font-size:14px;font-weight:600;color:#0F172A">{stem}</span>
                <span class="result-badge {badge_cls}">{badge_text}</span>
            </div>""", unsafe_allow_html=True)

            col_img, col_data = st.columns([1, 2], gap="large")

            with col_img:
                if stem in img_map:
                    orig = img_map[stem]
                    # Try OCR annotation
                    json_c = task2_json_dir / (orig.name + ".json")
                    if not json_c.exists():
                        json_c = task2_json_dir / (orig.stem + ".json")
                    if json_c.exists():
                        annotated = draw_ocr_boxes(orig, json_c)
                        tab_orig, tab_ann = st.tabs(["Original", "OCR Overlay"])
                        with tab_orig:
                            st.image(str(orig), use_container_width=True)
                        with tab_ann:
                            st.image(annotated, use_container_width=True)
                    else:
                        st.image(str(orig), use_container_width=True)
                    st.markdown(
                        f'<div style="font-size:11px;color:#94A3B8;margin-top:4px">{orig.name}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="color:#94A3B8;font-size:13px">Image not found: {stem}</div>',
                        unsafe_allow_html=True
                    )

            with col_data:
                if sub_df.empty:
                    st.markdown("""
                    <div style="text-align:center;padding:40px 20px;color:#94A3B8">
                        <div style="font-size:24px;margin-bottom:8px">🔍</div>
                        <div style="font-size:13px">No data extracted from this chart.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.dataframe(sub_df, use_container_width=True, height=280)
                    csv_bytes = sub_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"⬇ Download {stem}.csv",
                        data=csv_bytes,
                        file_name=f"{stem}.csv",
                        mime="text/csv",
                        key=f"dl_{stem}",
                    )

    # Download ALL button
    if all_dfs:
        st.markdown("---")
        combined = pd.concat(all_dfs, ignore_index=True)
        cb = combined.to_csv(index=False).encode("utf-8")
        col_dl, _ = st.columns([2, 5])
        with col_dl:
            st.download_button(
                "⬇ Download All Results (combined)",
                data=cb,
                file_name="all_results.csv",
                mime="text/csv",
                key="dl_all",
                type="primary",
                use_container_width=True,
            )


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
def main():
    ensure_session_state()
    inject_custom_css()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
