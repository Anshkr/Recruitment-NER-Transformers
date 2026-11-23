from pathlib import Path
import html
import json
import time

import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

# ==============================================================================
# 1. APPLICATION & SYSTEM CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Hugging Face Hub repo that holds the fine-tuned model.
HF_MODEL_ID = "boss012/recruitment-bert"

# Optional local copy. If this folder exists it is used, otherwise the model
# is downloaded from the Hugging Face Hub (needed for cloud deployment).
LOCAL_MODEL_DIR = BASE_DIR / "models" / "recruitment_bert"
HAS_LOCAL_WEIGHTS = (
    (LOCAL_MODEL_DIR / "model.safetensors").exists()
    or (LOCAL_MODEL_DIR / "pytorch_model.bin").exists()
)
MODEL_SOURCE = str(LOCAL_MODEL_DIR) if HAS_LOCAL_WEIGHTS else HF_MODEL_ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512

st.set_page_config(
    page_title="RecruitmentAI | Neural Resume Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(html_content):
    """
    Render raw HTML safely in Streamlit without markdown interpreting indented lines
    as code blocks (<pre><code>).
    """
    clean_lines = [line.strip() for line in html_content.strip().splitlines() if line.strip()]
    st.markdown(" ".join(clean_lines), unsafe_allow_html=True)

# ==============================================================================
# 2. DESIGN SYSTEM & REFINED CSS STYLESHEET
# ==============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --primary: #2563EB;
    --primary-hover: #1D4ED8;
    --slate-900: #0F172A;
    --slate-800: #1E293B;
    --slate-700: #334155;
    --slate-600: #475569;
    --slate-500: #64748B;
    --slate-400: #94A3B8;
    --slate-200: #E2E8F0;
    --slate-100: #F1F5F9;
    --slate-50: #F8FAFC;
    --card-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05), 0 1px 2px -1px rgba(15, 23, 42, 0.05);
    --card-shadow-hover: 0 8px 24px -4px rgba(15, 23, 42, 0.08), 0 4px 8px -2px rgba(15, 23, 42, 0.04);
}

html, body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #0F172A !important;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace;
}

/* ==========================================================================
   FORCE HIGH CONTRAST DARK TEXT ON LIGHT BACKGROUNDS (THEME AGNOSTIC)
   ========================================================================== */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

[data-testid="stHeader"] {
    background: rgba(248, 250, 252, 0.9) !important;
    backdrop-filter: blur(8px) !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3.5rem;
    max-width: 1420px;
}

/* Force dark text for all main content elements */
[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6,
[data-testid="stMain"] li {
    color: #0F172A !important;
}

/* Widget labels & descriptions */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Captions and subtext */
.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {
    color: #475569 !important;
    font-size: 12.5px !important;
}

/* Markdown paragraphs inside main area */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #1E293B;
}

/* Text area styling */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
}

/* Text Inputs */
.stTextInput input {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

.stTextInput input:focus {
    border-color: #2563EB !important;
}

/* File Uploader styling */
[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

[data-testid="stFileUploader"] section * {
    color: #334155 !important;
}

[data-testid="stFileUploader"] button {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    font-weight: 600 !important;
}

/* Multiselect styling */
[data-baseweb="select"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="select"] * {
    color: #0F172A !important;
}

[data-baseweb="tag"] {
    background-color: #E2E8F0 !important;
    color: #0F172A !important;
}

[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #E2E8F0 !important;
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 10px 18px !important;
    border-radius: 8px !important;
    color: #475569 !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #2563EB !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
}

/* Buttons */
.stButton > button {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    min-height: 42px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #F8FAFC !important;
    color: #2563EB !important;
    border-color: #94A3B8 !important;
}

.stButton > button[kind="primary"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1D4ED8 !important;
    color: #FFFFFF !important;
    border-color: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
}

.stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    min-height: 42px !important;
}

.stDownloadButton > button:hover {
    background-color: #F8FAFC !important;
    color: #2563EB !important;
    border-color: #94A3B8 !important;
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

[data-testid="stDataFrame"] * {
    color: #0F172A !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] * {
    color: #0F172A !important;
}

/* ==========================================================================
   SIDEBAR STYLING (DARK, SLEEK, REFINED)
   ========================================================================== */

[data-testid="stSidebar"] {
    background-color: #0B132B !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
    margin: 1.2rem 0 !important;
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 6px;
}

.brand-logo-box {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #3B82F6, #1D4ED8);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
}

.brand-title {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF !important;
    margin: 0;
    line-height: 1.1;
}

.brand-tagline {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #94A3B8 !important;
    margin-top: 3px;
}

.status-badge-online {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34D399 !important;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 20px;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
}

@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    70% {
        box-shadow: 0 0 0 7px rgba(16, 185, 129, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    margin-top: 14px;
}

.sidebar-prop {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    font-size: 12.5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.sidebar-prop:last-child {
    border-bottom: none;
}

.sidebar-prop-label {
    color: #94A3B8 !important;
    font-weight: 500;
}

.sidebar-prop-val {
    color: #F8FAFC !important;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

/* ==========================================================================
   CARDS, PANELS, HERO & DOSSIER
   ========================================================================== */

.hero-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: var(--card-shadow);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.hero-title-group h1 {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.7px;
    color: #0F172A !important;
    margin: 0;
    line-height: 1.2;
}

.hero-title-group p {
    color: #475569 !important;
    font-size: 14px;
    margin: 6px 0 0 0;
}

.meta-pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.meta-pill {
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    color: #1E293B !important;
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.2px;
}

/* Executive Metric Cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.stat-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: var(--card-shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}

.stat-box:hover {
    transform: translateY(-2px);
    box-shadow: var(--card-shadow-hover);
}

.stat-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--primary);
    opacity: 0.85;
}

.stat-label {
    color: #64748B !important;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}

.stat-value {
    color: #0F172A !important;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.8px;
    line-height: 1.1;
}

.stat-subtext {
    color: #64748B !important;
    font-size: 11.5px;
    margin-top: 6px;
    font-weight: 500;
}

/* Candidate Dossier Summary Card */
.dossier-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--card-shadow);
    margin-bottom: 24px;
}

.dossier-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 18px;
    margin-bottom: 18px;
    flex-wrap: wrap;
    gap: 16px;
}

.dossier-avatar-group {
    display: flex;
    align-items: center;
    gap: 16px;
}

.dossier-avatar {
    width: 52px;
    height: 52px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1E293B, #0F172A);
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 19px;
    letter-spacing: -0.5px;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.15);
}

.dossier-name {
    font-size: 20px;
    font-weight: 800;
    color: #0F172A !important;
    letter-spacing: -0.4px;
    margin: 0;
}

.dossier-role {
    font-size: 14px;
    font-weight: 600;
    color: var(--primary) !important;
    margin-top: 2px;
}

.dossier-meta-item {
    font-size: 13px;
    color: #334155 !important;
    display: flex;
    align-items: center;
    gap: 6px;
}

.dossier-meta-item b {
    color: #0F172A !important;
}

.dossier-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
}

.dossier-section-title {
    font-size: 11.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #64748B !important;
    margin-bottom: 8px;
}

.dossier-chip-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.dossier-chip {
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}

/* Content Panels */
.content-panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px;
    box-shadow: var(--card-shadow);
    margin-bottom: 20px;
}

/* Entity Highlighting Tags */
.ner-viewer-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 26px;
    color: #1E293B !important;
    line-height: 2.2;
    font-size: 14.5px;
    white-space: pre-wrap;
    box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.02);
}

.ner-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 7px;
    margin: 2px 1px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13.5px;
    line-height: 1.45;
    border: 1px solid transparent;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.ner-tag:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
}

.ner-badge {
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 1px 5px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.08);
}

/* Legend items */
.legend-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid transparent;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}

.legend-count {
    background: rgba(0, 0, 0, 0.08);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
}

/* Footer */
.app-footer {
    border-top: 1px solid #E2E8F0;
    padding-top: 24px;
    margin-top: 40px;
    text-align: center;
    color: #64748B !important;
    font-size: 12px;
    line-height: 1.8;
}

.app-footer b {
    color: #1E293B !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. PALETTE DEFINITIONS & STANDARD SAMPLE
# ==============================================================================

ENTITY_PALETTE = {
    "PERSON": {
        "bg": "#E0F2FE", "border": "#BAE6FD", "text": "#0369A1",
        "description": "Candidate full name or identification"
    },
    "DESIGNATION": {
        "bg": "#ECFDF5", "border": "#A7F3D0", "text": "#047857",
        "description": "Job title, professional role or position"
    },
    "COMPANY": {
        "bg": "#F3E8FF", "border": "#DDD6FE", "text": "#6D28D9",
        "description": "Employer, client or enterprise name"
    },
    "LOCATION": {
        "bg": "#FEF3C7", "border": "#FDE68A", "text": "#B45309",
        "description": "Geographical location, city or region"
    },
    "EMAIL": {
        "bg": "#FCE7F3", "border": "#FBCFE8", "text": "#BE185D",
        "description": "Electronic mail address"
    },
    "TECH_SKILL": {
        "bg": "#E0E7FF", "border": "#C7D2FE", "text": "#4338CA",
        "description": "Technical frameworks, languages and tools"
    },
    "JOB_SKILL": {
        "bg": "#CCFBF1", "border": "#99F6E4", "text": "#0F766E",
        "description": "Functional domain competencies and soft skills"
    },
    "EXPERIENCE": {
        "bg": "#F1F5F9", "border": "#CBD5E1", "text": "#334155",
        "description": "Years of service or tenure metrics"
    },
    "DEGREE": {
        "bg": "#FFE4E6", "border": "#FECDD3", "text": "#BE123C",
        "description": "Academic degree, qualification or credential"
    },
    "COLLEGE": {
        "bg": "#EEF2FF", "border": "#E0E7FF", "text": "#3730A3",
        "description": "University, college or academic institution"
    },
    "GRAD_YEAR": {
        "bg": "#FEF9C3", "border": "#FEF08A", "text": "#A16207",
        "description": "Year of graduation or study completion"
    }
}

SAMPLE_RESUME = """Alex Morgan
Software Engineer

Email: alex.morgan@example.com
Location: Bengaluru, Karnataka

Professional Experience:
3 years of experience in software development.

Technical Skills:
Python, Java, SQL, Machine Learning, Git.

Education:
B.Tech in Computer Science
North Maharashtra University
Graduation Year: 2023

Company:
ABC Technologies

Designation:
Software Engineer
"""

# ==============================================================================
# 4. MODEL LOADER & INFERENCE
# ==============================================================================

@st.cache_resource(show_spinner="Loading BERT model (first run downloads it from Hugging Face)...")
def load_bert_model():
    """Load and cache the fine-tuned BERT model and tokenizer.

    Uses the local folder when present, otherwise the Hugging Face Hub repo.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_SOURCE)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def extract_entities_with_diagnostics(text, tokenizer, model):
    """
    Perform token classification and aggregate subwords into entities with confidence scores.
    Also records inference execution latency.
    """
    start_time = time.perf_counter()

    untruncated = tokenizer(text, truncation=False)["input_ids"]
    token_count = len(untruncated)

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    offsets = encoding.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
        confidence_scores = probabilities[
            0,
            torch.arange(probabilities.shape[1]),
            predictions
        ].cpu().tolist()

    id2label = model.config.id2label
    entities = []
    current_entity = None

    def flush_entity():
        nonlocal current_entity
        if current_entity is not None:
            s = current_entity["start"]
            e = current_entity["end"]
            if s < e:
                current_entity["text"] = text[s:e].strip()
                current_entity["confidence"] = round(
                    sum(current_entity["scores"]) / len(current_entity["scores"]),
                    4
                )
                current_entity.pop("scores", None)
                if current_entity["text"]:
                    entities.append(current_entity.copy())
            current_entity = None

    for pred, conf, (start, end) in zip(predictions, confidence_scores, offsets):
        if start == end:
            continue

        tag = id2label.get(int(pred), "O")

        if tag == "O" or "-" not in tag:
            flush_entity()
            continue

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B":
            flush_entity()
            current_entity = {
                "label": entity_type,
                "start": start,
                "end": end,
                "scores": [conf]
            }
        elif prefix == "I":
            if (
                current_entity is not None
                and current_entity["label"] == entity_type
                and start >= current_entity["end"]
            ):
                current_entity["end"] = end
                current_entity["scores"].append(conf)
            else:
                flush_entity()
                current_entity = {
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "scores": [conf]
                }

    flush_entity()
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return entities, token_count, elapsed_ms


# ==============================================================================
# 5. DOSSIER AGGREGATION & FORMATTING
# ==============================================================================

def build_candidate_profile(entities):
    """Aggregate parsed entities into a structured executive dossier."""
    def get_unique_items(label):
        seen = set()
        items = []
        for ent in entities:
            if ent["label"] == label:
                val = ent["text"].strip()
                norm = val.lower()
                if norm not in seen and len(val) > 1:
                    seen.add(norm)
                    items.append(val)
        return items

    name_items = get_unique_items("PERSON")
    desig_items = get_unique_items("DESIGNATION")
    comp_items = get_unique_items("COMPANY")
    loc_items = get_unique_items("LOCATION")
    email_items = get_unique_items("EMAIL")
    exp_items = get_unique_items("EXPERIENCE")
    tech_items = get_unique_items("TECH_SKILL")
    job_items = get_unique_items("JOB_SKILL")
    deg_items = get_unique_items("DEGREE")
    col_items = get_unique_items("COLLEGE")
    grad_items = get_unique_items("GRAD_YEAR")

    candidate_name = name_items[0] if name_items else "Candidate Profile"
    initials = "".join([part[0] for part in candidate_name.split()[:2]]).upper() if name_items else "CP"

    return {
        "name": candidate_name,
        "initials": initials,
        "primary_role": desig_items[0] if desig_items else "Role Not Detected",
        "companies": comp_items,
        "locations": loc_items,
        "emails": email_items,
        "experience": exp_items,
        "tech_skills": tech_items,
        "job_skills": job_items,
        "education": {
            "degree": deg_items[0] if deg_items else None,
            "college": col_items[0] if col_items else None,
            "grad_year": grad_items[0] if grad_items else None
        }
    }


def render_highlighted_document(text, entities, active_categories=None):
    """Render text with styled, high-contrast entity tags."""
    if active_categories is None:
        active_categories = set(ENTITY_PALETTE.keys())

    sorted_entities = sorted(
        [e for e in entities if e["label"] in active_categories],
        key=lambda item: item["start"]
    )

    result = []
    position = 0

    for entity in sorted_entities:
        start = entity["start"]
        end = entity["end"]

        if start < position:
            continue

        result.append(html.escape(text[position:start]))

        label = entity["label"]
        palette = ENTITY_PALETTE.get(
            label,
            {"bg": "#F1F5F9", "border": "#CBD5E1", "text": "#334155"}
        )
        snippet = html.escape(text[start:end])

        tag_html = (
            f'<span class="ner-tag" style="background:{palette["bg"]}; border-color:{palette["border"]}; color:{palette["text"]};">'
            f'{snippet}'
            f'<span class="ner-badge" style="color:{palette["text"]};">[{html.escape(label)}]</span>'
            f'</span>'
        )
        result.append(tag_html)
        position = end

    result.append(html.escape(text[position:]))
    return "".join(result)


# ==============================================================================
# 6. SIDEBAR ARCHITECTURE (CLEAN & MINIMALIST)
# ==============================================================================

with st.sidebar:
    st.markdown("""
    <div class="brand-wrapper">
        <div class="brand-logo-box">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
        </div>
        <div>
            <div class="brand-title">RecruitmentAI</div>
            <div class="brand-tagline">Neural Resume Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 14px;">
        <span class="status-badge-online">
            <span class="pulse-dot"></span>
            BERT Engine Operational
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #94A3B8; margin-bottom: 8px;">
        Pipeline Specifications
    </div>
    <div class="sidebar-card">
        <div class="sidebar-prop">
            <span class="sidebar-prop-label">Base Model</span>
            <span class="sidebar-prop-val">BERT-Base-Cased</span>
        </div>
        <div class="sidebar-prop">
            <span class="sidebar-prop-label">Task Objective</span>
            <span class="sidebar-prop-val">Token Classification</span>
        </div>
        <div class="sidebar-prop">
            <span class="sidebar-prop-label">Tagging Scheme</span>
            <span class="sidebar-prop-val">BIO Scheme</span>
        </div>
        <div class="sidebar-prop">
            <span class="sidebar-prop-label">Target Classes</span>
            <span class="sidebar-prop-val">11 Categories</span>
        </div>
        <div class="sidebar-prop">
            <span class="sidebar-prop-label">Vocab Coverage</span>
            <span class="sidebar-prop-val">28,996 Tokens</span>
        </div>
        <div class="sidebar-prop">
            <span class="sidebar-prop-label">Compute Target</span>
            <span class="sidebar-prop-val">""" + str(DEVICE).upper() + """</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size: 11.5px; color: #94A3B8; line-height: 1.6;">
        <b>Academic NLP Project</b><br>
        Recruitment Information Extraction with Transformer-based Sequence Tagging.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 7. MAIN HEADER & HERO SECTION
# ==============================================================================

st.markdown("""
<div class="hero-header">
    <div class="hero-title-group">
        <h1>Recruitment Intelligence Platform</h1>
        <p>Transformer-based Named Entity Recognition for automated candidate profiling and recruitment intelligence.</p>
    </div>
    <div class="meta-pill-container">
        <span class="meta-pill">Fine-Tuned BERT</span>
        <span class="meta-pill">Context: 512 Tokens</span>
        <span class="meta-pill">11 Entity Categories</span>
        <span class="meta-pill">Subword Tokenization</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load the model (local folder if present, otherwise Hugging Face Hub)
try:
    tokenizer, model = load_bert_model()
except Exception as exc:
    st.error(
        f"Failed to load the model from '{MODEL_SOURCE}'. "
        f"Check your internet connection and that the repo is public. Details: {exc}"
    )
    st.stop()


# ==============================================================================
# 8. RESUME INPUT & ACTION WORKFLOW
# ==============================================================================

if "resume_input" not in st.session_state:
    st.session_state["resume_input"] = ""

def handle_file_upload():
    uploaded = st.session_state.get("resume_file_uploader")
    if uploaded is not None:
        try:
            content = uploaded.getvalue().decode("utf-8-sig")
            st.session_state["resume_input"] = content
        except UnicodeDecodeError:
            st.session_state["upload_error_msg"] = "Decoding error: Please ensure the document is encoded in UTF-8."

def set_sample_resume():
    st.session_state["resume_input"] = SAMPLE_RESUME

def clear_resume_workspace():
    st.session_state["resume_input"] = ""
    st.session_state.pop("analysis_result", None)

st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
    <div style="font-size: 16px; font-weight: 700; color: #0F172A;">Resume Input & Document Ingestion</div>
    <div style="font-size: 12.5px; color: #475569; font-weight: 500;">Supported format: Plain Text (.txt)</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Ingest Text Resume File",
    type=["txt"],
    key="resume_file_uploader",
    on_change=handle_file_upload,
    help="Upload candidate text file in UTF-8 format."
)

if "upload_error_msg" in st.session_state:
    st.error(st.session_state.pop("upload_error_msg"))

resume_text = st.text_area(
    "Candidate Resume Content",
    height=240,
    placeholder="Paste candidate resume text here (experience, skills, designations, education, contact details)...",
    key="resume_input"
)

char_count = len(resume_text)
word_count = len(resume_text.split()) if resume_text else 0
st.caption(f"Input Document Metrics: {char_count:,} characters | {word_count:,} words")

# Streamlined Action Row
btn_col1, btn_col2, btn_col3 = st.columns([1.1, 1, 1.6])

with btn_col1:
    st.button("Use Sample Resume", on_click=set_sample_resume, use_container_width=True)

with btn_col2:
    st.button("Clear Workspace", on_click=clear_resume_workspace, use_container_width=True)

with btn_col3:
    analyze_btn = st.button("Run Neural Extraction Pipeline", type="primary", use_container_width=True)

if analyze_btn:
    if not resume_text.strip():
        st.warning("Please provide resume text or click 'Use Sample Resume' before initiating analysis.")
    else:
        with st.spinner("Executing BERT token classification and entity aggregation..."):
            try:
                extracted, t_count, latency = extract_entities_with_diagnostics(
                    resume_text,
                    tokenizer,
                    model
                )
                st.session_state["analysis_result"] = {
                    "text": resume_text,
                    "entities": extracted,
                    "token_count": t_count,
                    "latency": latency
                }
            except Exception as e:
                st.error(f"Inference pipeline execution error: {e}")


# ==============================================================================
# 9. RESULTS, CANDIDATE DOSSIER & VISUAL ANALYTICS
# ==============================================================================

res = st.session_state.get("analysis_result")

if res is not None:
    raw_text = res["text"]
    active_entities = res["entities"]
    token_count = res["token_count"]
    latency = res["latency"]

    unique_categories = set(e["label"] for e in active_entities)

    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

    # Context window warning
    if token_count > MAX_LENGTH:
        st.warning(
            f"Context Ceiling Notice: Document contains {token_count} subword tokens. "
            f"The BERT architecture evaluated the initial {MAX_LENGTH} tokens. Subsequent tokens were truncated."
        )

    # Top Executive Metric Cards
    avg_conf = (
        sum(e["confidence"] for e in active_entities) / len(active_entities)
        if active_entities else 0.0
    )

    render_html(f"""
    <div class="metrics-grid">
        <div class="stat-box">
            <div class="stat-label">Total Entities</div>
            <div class="stat-value">{len(active_entities):,}</div>
            <div class="stat-subtext">Aggregated token spans</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Categories Discovered</div>
            <div class="stat-value">{len(unique_categories)} / 11</div>
            <div class="stat-subtext">Active recruitment classes</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Mean Confidence</div>
            <div class="stat-value">{avg_conf:.1%}</div>
            <div class="stat-subtext">Average token probability</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Inference Latency</div>
            <div class="stat-value">{latency} ms</div>
            <div class="stat-subtext">Hardware: {str(DEVICE).upper()}</div>
        </div>
    </div>
    """)

    # Executive Candidate Dossier
    profile = build_candidate_profile(active_entities)

    tech_chips = "".join([f'<span class="dossier-chip" style="background:#E0E7FF; color:#4338CA; border:1px solid #C7D2FE;">{html.escape(s)}</span>' for s in profile["tech_skills"]]) if profile["tech_skills"] else '<span style="color:#94A3B8; font-size:13px;">No technical skills detected</span>'
    job_chips = "".join([f'<span class="dossier-chip" style="background:#CCFBF1; color:#0F766E; border:1px solid #99F6E4;">{html.escape(s)}</span>' for s in profile["job_skills"]]) if profile["job_skills"] else '<span style="color:#94A3B8; font-size:13px;">No domain competencies detected</span>'
    grad_text = f'<br><span style="color:#A16207; font-weight:600; font-size:12px;">Class of {html.escape(profile["education"]["grad_year"])}</span>' if profile["education"]["grad_year"] else ''

    render_html(f"""
    <div class="dossier-card">
        <div class="dossier-header">
            <div class="dossier-avatar-group">
                <div class="dossier-avatar">{profile["initials"]}</div>
                <div>
                    <div class="dossier-name">{html.escape(profile["name"])}</div>
                    <div class="dossier-role">{html.escape(profile["primary_role"])}</div>
                </div>
            </div>
            <div style="display: flex; gap: 18px; flex-wrap: wrap;">
                <div class="dossier-meta-item">
                    <span style="color:#64748B;">Location:</span>
                    <b>{html.escape(", ".join(profile["locations"]) if profile["locations"] else "Not Specified")}</b>
                </div>
                <div class="dossier-meta-item">
                    <span style="color:#64748B;">Contact:</span>
                    <b>{html.escape(", ".join(profile["emails"]) if profile["emails"] else "Not Specified")}</b>
                </div>
                <div class="dossier-meta-item">
                    <span style="color:#64748B;">Experience:</span>
                    <b>{html.escape(", ".join(profile["experience"]) if profile["experience"] else "Not Detected")}</b>
                </div>
            </div>
        </div>
        <div class="dossier-grid">
            <div>
                <div class="dossier-section-title">Technical Skill Stack</div>
                <div class="dossier-chip-list">{tech_chips}</div>
            </div>
            <div>
                <div class="dossier-section-title">Job & Domain Competencies</div>
                <div class="dossier-chip-list">{job_chips}</div>
            </div>
            <div>
                <div class="dossier-section-title">Academic Background</div>
                <div style="font-size:13.5px; color:#334155; line-height:1.6;">
                    <b>{html.escape(profile["education"]["degree"] or "Degree Not Detected")}</b><br>
                    <span style="color:#64748B;">{html.escape(profile["education"]["college"] or "University Not Detected")}</span>
                    {grad_text}
                </div>
            </div>
        </div>
    </div>
    """)

    # 4 Modular Exploration Tabs
    tab_doc, tab_table, tab_charts, tab_export = st.tabs([
        "Document Annotations",
        "Entity Registry",
        "Distribution Analytics",
        "Export & Payload"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: DOCUMENT ANNOTATIONS
    # --------------------------------------------------------------------------
    with tab_doc:
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="font-size:15px; font-weight:700; color:#0F172A;">Interactive Visual Span Annotations</div>
            <div style="font-size:12.5px; color:#64748B;">Select entity categories below to highlight dynamically</div>
        </div>
        """, unsafe_allow_html=True)

        all_present_categories = sorted(list(set(e["label"] for e in active_entities)))
        selected_display_categories = st.multiselect(
            "Filter highlighted entity categories",
            options=all_present_categories,
            default=all_present_categories,
            help="Toggle specific entity tags to inspect categories in isolation."
        )

        rendered_document = render_highlighted_document(
            raw_text,
            active_entities,
            active_categories=set(selected_display_categories)
        )

        st.markdown(
            f'<div class="ner-viewer-box">{rendered_document}</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<b style='color:#0F172A;'>Category Legend & Frequency</b>", unsafe_allow_html=True)

        legend_cols = st.columns(3)
        cat_counts = pd.Series([e["label"] for e in active_entities]).value_counts().to_dict()

        for idx, cat in enumerate(all_present_categories):
            pal = ENTITY_PALETTE.get(cat, {"bg": "#F1F5F9", "border": "#CBD5E1", "text": "#334155"})
            cnt = cat_counts.get(cat, 0)
            with legend_cols[idx % 3]:
                render_html(f"""
                <div class="legend-card" style="background:{pal['bg']}; border-color:{pal['border']}; color:{pal['text']};">
                    <span>{cat}</span>
                    <span class="legend-count" style="color:{pal['text']};">{cnt} detected</span>
                </div>
                """)

    # --------------------------------------------------------------------------
    # TAB 2: ENTITY REGISTRY
    # --------------------------------------------------------------------------
    with tab_table:
        st.markdown("""
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:12px;">
            Structured Entity Records
        </div>
        """, unsafe_allow_html=True)

        if active_entities:
            df_records = pd.DataFrame([
                {
                    "Category": e["label"],
                    "Extracted Text": e["text"],
                    "Confidence": e["confidence"],
                    "Start Pos": e["start"],
                    "End Pos": e["end"],
                    "Length": len(e["text"])
                }
                for e in active_entities
            ])

            table_filter_col1, table_filter_col2 = st.columns([1.5, 2])
            with table_filter_col1:
                cat_filter = st.multiselect(
                    "Filter by Category",
                    options=sorted(df_records["Category"].unique().tolist()),
                    default=sorted(df_records["Category"].unique().tolist()),
                    key="reg_cat_filter"
                )
            with table_filter_col2:
                search_query = st.text_input(
                    "Search Extracted Text",
                    placeholder="Search for skills, locations, companies...",
                    key="reg_search"
                )

            filtered_records = df_records[df_records["Category"].isin(cat_filter)]
            if search_query.strip():
                filtered_records = filtered_records[
                    filtered_records["Extracted Text"].str.contains(search_query, case=False, na=False)
                ]

            st.caption(f"Showing {len(filtered_records)} of {len(df_records)} registered entities.")

            st.dataframe(
                filtered_records,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Model Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f"
                    ),
                    "Category": st.column_config.TextColumn("Entity Category"),
                    "Extracted Text": st.column_config.TextColumn("Extracted Entity Span"),
                    "Start Pos": st.column_config.NumberColumn("Start Offset"),
                    "End Pos": st.column_config.NumberColumn("End Offset")
                }
            )
        else:
            st.info("No entities detected in this resume.")

    # --------------------------------------------------------------------------
    # TAB 3: VISUAL DISTRIBUTION ANALYTICS
    # --------------------------------------------------------------------------
    with tab_charts:
        st.markdown("""
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:16px;">
            Statistical & Confidence Distribution Analytics
        </div>
        """, unsafe_allow_html=True)

        if active_entities:
            df_analytics = pd.DataFrame(active_entities)
            counts_series = df_analytics["label"].value_counts().reset_index()
            counts_series.columns = ["Category", "Count"]

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_bar = px.bar(
                    counts_series.sort_values(by="Count", ascending=True),
                    x="Count",
                    y="Category",
                    orientation="h",
                    title="Entity Frequency per Category",
                    color="Count",
                    color_continuous_scale="Blues"
                )
                fig_bar.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="Inter", size=12, color="#0F172A"),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="#F1F5F9", showline=True, linecolor="#E2E8F0"),
                    yaxis=dict(showgrid=False, showline=True, linecolor="#E2E8F0")
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with chart_col2:
                conf_by_cat = df_analytics.groupby("label")["confidence"].mean().reset_index()
                conf_by_cat.columns = ["Category", "Mean Confidence"]

                fig_conf = px.bar(
                    conf_by_cat.sort_values(by="Mean Confidence", ascending=True),
                    x="Mean Confidence",
                    y="Category",
                    orientation="h",
                    title="Mean Model Confidence per Category",
                    color="Mean Confidence",
                    color_continuous_scale="Tealgrn"
                )
                fig_conf.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="Inter", size=12, color="#0F172A"),
                    coloraxis_showscale=False,
                    xaxis=dict(range=[0, 1.05], gridcolor="#F1F5F9", showline=True, linecolor="#E2E8F0"),
                    yaxis=dict(showgrid=False, showline=True, linecolor="#E2E8F0")
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            # Subword Sequence Diagnostics
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            seq_col1, seq_col2 = st.columns(2)
            with seq_col1:
                token_utilization = min(token_count / MAX_LENGTH, 1.0)
                render_html(f"""
                <div class="content-panel" style="margin-bottom:0;">
                    <div style="font-size:13px; font-weight:700; color:#0F172A; margin-bottom:8px;">Subword Token Context Budget</div>
                    <div style="font-size:24px; font-weight:800; color:#2563EB;">{token_count} / {MAX_LENGTH} Tokens</div>
                    <div style="background:#E2E8F0; border-radius:8px; height:8px; width:100%; margin:10px 0; overflow:hidden;">
                        <div style="background:{'#EF4444' if token_count > MAX_LENGTH else '#2563EB'}; height:100%; width:{min(token_utilization*100, 100)}%;"></div>
                    </div>
                    <div style="font-size:12px; color:#475569;">Capacity utilization: {token_utilization:.1%} of 512 context limit</div>
                </div>
                """)

            with seq_col2:
                entity_density = (len(active_entities) / token_count * 100) if token_count else 0
                render_html(f"""
                <div class="content-panel" style="margin-bottom:0;">
                    <div style="font-size:13px; font-weight:700; color:#0F172A; margin-bottom:8px;">Entity Extraction Density</div>
                    <div style="font-size:24px; font-weight:800; color:#0F766E;">{entity_density:.1f}%</div>
                    <div style="font-size:12.5px; color:#475569; margin-top:8px; line-height:1.6;">
                        Average of {round(token_count / len(active_entities), 1) if active_entities else 0} tokens per extracted recruitment entity.
                    </div>
                </div>
                """)
        else:
            st.info("Visual charts are unavailable because no entities were detected.")

    # --------------------------------------------------------------------------
    # TAB 4: EXPORT & PAYLOAD
    # --------------------------------------------------------------------------
    with tab_export:
        st.markdown("""
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:12px;">
            Downstream Integration & Structured Export
        </div>
        """, unsafe_allow_html=True)

        if active_entities:
            df_export = pd.DataFrame([
                {
                    "entity": e["label"],
                    "text": e["text"],
                    "confidence": e["confidence"],
                    "start": e["start"],
                    "end": e["end"]
                }
                for e in active_entities
            ])

            csv_payload = df_export.to_csv(index=False).encode("utf-8-sig")

            full_payload = {
                "metadata": {
                    "model": "BERT-Base-Cased-Recruitment-NER",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                    "device": str(DEVICE),
                    "subword_tokens": token_count,
                    "inference_ms": latency
                },
                "candidate_profile": profile,
                "entities": active_entities
            }
            json_payload = json.dumps(full_payload, indent=2, ensure_ascii=False)

            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                render_html("""
                <div class="content-panel">
                    <div style="font-size:15px; font-weight:700; color:#0F172A;">Tabular CSV Export</div>
                    <div style="color:#475569; font-size:13px; margin:6px 0 16px 0;">
                        Structured entity observations suitable for spreadsheet analysis, ATS databases, or analytics pipelines.
                    </div>
                </div>
                """)
                st.download_button(
                    "Download Entities CSV",
                    data=csv_payload,
                    file_name="candidate_recruitment_entities.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with exp_col2:
                render_html("""
                <div class="content-panel">
                    <div style="font-size:15px; font-weight:700; color:#0F172A;">JSON API Payload</div>
                    <div style="color:#475569; font-size:13px; margin:6px 0 16px 0;">
                        Full structured document object containing candidate profile summary, span offsets, and token metrics.
                    </div>
                </div>
                """)
                st.download_button(
                    "Download Structured JSON",
                    data=json_payload,
                    file_name="candidate_recruitment_payload.json",
                    mime="application/json",
                    use_container_width=True
                )

            with st.expander("Preview JSON Payload Schema", expanded=False):
                st.code(json_payload[:2000] + ("\n..." if len(json_payload) > 2000 else ""), language="json")
        else:
            st.info("No entity data available for export.")


# ==============================================================================
# 10. SYSTEM FOOTER
# ==============================================================================

st.markdown("""
<div class="app-footer">
    <b>RecruitmentAI Platform</b> · Transformer-Based Natural Language Processing
    <br>
    Academic Research Demonstration · Sequence Tagging with Fine-Tuned BERT Architecture
    <br>
    <span style="font-size: 11px; color: #94A3B8;">
        Disclaimer: This system automates entity extraction and informational structuring. It does not perform automated ranking, scoring, or candidate selection.
    </span>
</div>
""", unsafe_allow_html=True)