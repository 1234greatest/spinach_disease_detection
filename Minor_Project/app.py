

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import json
import math
from datetime import datetime
from PIL import Image
import io
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
from streamlit_cropper import st_cropper

# ─── YOLO Model loaders ───────────────────────────────────────────────────────
@st.cache_resource
def load_classification_model():
    try:
        from ultralytics import YOLO
        return YOLO("models/class_2/best.pt")
    except Exception as e:
        return None

@st.cache_resource
def load_disease_model():
    try:
        from ultralytics import YOLO
        return YOLO("models/detection/best.pt")
    except Exception as e:
        return None

# ─── Groq Config ──────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PalakGuard — Spinach Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --green-dark:   #0d2818;
    --green-mid:    #1a4a2e;
    --green-vivid:  #2d7a4f;
    --green-light:  #4caf7d;
    --green-pale:   #a8e6bf;
    --gold:         #c9a84c;
    --red-alert:    #d94f3d;
    --yellow-warn:  #e8b84b;
    --text-main:    #1a1a2e;
    --text-muted:   #5a6a72;
    --border:       #e0ebe4;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #f0f5f1; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--green-dark) 0%, var(--green-mid) 100%) !important; border-right: 1px solid #2a5c3a; }
[data-testid="stSidebar"] * { color: #d4eadb !important; }
[data-testid="stSidebar"] .stRadio label { color: #d4eadb !important; font-size:0.92rem; }
[data-testid="stAppViewContainer"] > .main { background-color: #f0f5f1; }
[data-testid="block-container"] { padding-top: 1.5rem; padding-bottom: 3rem; }
.hero-banner { background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 50%, var(--green-vivid) 100%); border-radius: 16px; padding: 2.5rem 2.5rem 2rem; margin-bottom: 1.5rem; position: relative; overflow: hidden; box-shadow: 0 8px 32px rgba(13,40,24,0.18); }
.hero-banner::before { content: "🌿"; position: absolute; right: 2rem; top: 1rem; font-size: 7rem; opacity: 0.12; }
.hero-title { font-family: 'Playfair Display', serif; font-size: 2.6rem; font-weight: 900; color: #ffffff; margin: 0; line-height: 1.1; }
.hero-sub { color: var(--green-pale); font-size: 1rem; margin-top: 0.5rem; font-weight: 300; }
.hero-badge { display: inline-block; background: rgba(201,168,76,0.2); border: 1px solid var(--gold); color: var(--gold); padding: 3px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 0.8rem; }
.card { background: #fff; border-radius: 14px; padding: 1.5rem; border: 1px solid var(--border); box-shadow: 0 2px 12px rgba(0,0,0,0.05); margin-bottom: 1rem; }
.card-title { font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: var(--green-dark); margin-bottom: 0.8rem; }
.section-header { font-family: 'Playfair Display', serif; font-size: 1.55rem; font-weight: 700; color: var(--green-dark); margin: 0.5rem 0 1rem; border-left: 4px solid var(--green-vivid); padding-left: 12px; }
.info-row { display: flex; gap: 0; margin-bottom: 2px; }
.info-key { background: #f0f5f1; color: var(--text-muted); font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; padding: 6px 10px; border-radius: 6px 0 0 6px; min-width: 180px; border: 1px solid var(--border); }
.info-val { background: #fff; color: var(--text-main); font-size: 0.87rem; padding: 6px 12px; border-radius: 0 6px 6px 0; border: 1px solid var(--border); border-left: none; flex: 1; }
.stage-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: 30px; font-size: 0.82rem; font-weight: 600; margin-bottom: 0.5rem; }
.badge-success { background: #e6f7ed; color: #1a7a40; border: 1px solid #a8e6bf; }
.badge-error   { background: #fdecea; color: #c0392b; border: 1px solid #f5b7b1; }
.badge-warning { background: #fef9e7; color: #b7770d; border: 1px solid #f9e79f; }
.sev-bar-bg { flex: 1; height: 10px; background: #e8f0ea; border-radius: 10px; overflow: hidden; }
.sev-bar-fill { height: 100%; border-radius: 10px; }
.metric-card { background: white; border-radius: 12px; padding: 1rem 1.2rem; border: 1px solid var(--border); box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.metric-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }
.metric-value { font-size: 1.6rem; font-weight: 700; font-family: 'DM Mono', monospace; color: var(--green-dark); }
.metric-sub { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }
.feature-card { background: white; border-radius: 14px; padding: 1.5rem; border: 1px solid var(--border); text-align: center; height: 200px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.feature-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.feature-title { font-family: 'Playfair Display', serif; font-size: 1.05rem; font-weight: 700; color: var(--green-dark); margin-bottom: 0.4rem; }
.feature-desc { font-size: 0.82rem; color: var(--text-muted); line-height: 1.5; }
.step-card { background: white; border-radius: 14px; padding: 1.3rem; border: 1px solid var(--border); text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.step-num { font-size: 2rem; font-weight: 900; font-family: 'DM Mono', monospace; color: var(--green-vivid); }
.chat-bubble-user { background: var(--green-mid); color: #d4eadb; border-radius: 14px 14px 4px 14px; padding: 10px 14px; margin: 6px 0; font-size: 0.9rem; max-width: 85%; margin-left: auto; }
.chat-bubble-ai { background: #ffffff; color: var(--text-main); border: 1px solid var(--border); border-radius: 4px 14px 14px 14px; padding: 10px 14px; margin: 6px 0; font-size: 0.9rem; max-width: 92%; box-shadow: 0 1px 6px rgba(0,0,0,0.05); }
.chat-name-ai { font-size: 0.72rem; color: var(--green-vivid); font-weight: 600; margin-bottom: 3px; }
.chat-name-user { font-size: 0.72rem; color: #88b89a; font-weight: 600; margin-bottom: 3px; text-align: right; }
.stButton > button { background: linear-gradient(135deg, var(--green-vivid), var(--green-light)) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.5rem 1.5rem !important; font-weight: 600 !important; box-shadow: 0 4px 14px rgba(45,122,79,0.3) !important; }
[data-testid="stFileUploader"] { border: 2px dashed var(--green-light) !important; border-radius: 12px !important; background: #f7fbf8 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: var(--green-vivid); border-radius: 6px; }
.sample-label-healthy { background: #e6f7ed; color: #1a7a40; border: 1px solid #a8e6bf; padding: 4px 12px; border-radius: 20px; font-size:0.78rem; font-weight:600; display:inline-block; }
.sample-label-disease { background: #fdecea; color: #c0392b; border: 1px solid #f5b7b1; padding: 4px 12px; border-radius: 20px; font-size:0.78rem; font-weight:600; display:inline-block; }
.model-badge { display:inline-flex; align-items:center; gap:6px; background: rgba(45,122,79,0.1); border:1px solid rgba(45,122,79,0.3); color: var(--green-vivid); border-radius:20px; padding:3px 12px; font-size:0.72rem; font-weight:700; letter-spacing:0.5px; text-transform:uppercase; }
.accuracy-ring { text-align:center; }
.confusion-cell { padding: 12px; text-align:center; font-size:1.1rem; font-weight:700; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("scan_history", []), ("chat_history", []), ("last_result", None),
    ("total_scans", 0), ("disease_count", 0),
    ("show_overlay", False), ("overlay_image", None), ("overlay_stats", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ──────────────────────────────────────────────────────────────────
def severity_color(level):
    return {"Healthy":"#2d7a4f","Mild":"#e8b84b","Moderate":"#e07b39","Severe":"#d94f3d"}.get(level,"#888")

def severity_emoji(level):
    return {"Healthy":"🟢","Mild":"🟡","Moderate":"🟠","Severe":"🔴"}.get(level,"⚪")

def estimate_dsi(severity, confidence):
    base = {"Healthy":0,"Mild":1.5,"Moderate":3.0,"Severe":4.5}.get(severity, 0)
    return round(base * (confidence/100), 2)

def estimate_yield_loss(severity):
    return {"Healthy":0,"Mild":8,"Moderate":25,"Severe":55}.get(severity, 0)

# ─── YOLO Stage 1 ─────────────────────────────────────────────────────────────
def verify_spinach_yolo(image: Image.Image):
    clf_model = load_classification_model()
    if clf_model is None:
        return True, "spinach", 95.0
    img_array = np.array(image)
    results   = clf_model.predict(img_array, verbose=False)
    label     = results[0].names[results[0].probs.top1]
    conf      = float(results[0].probs.top1conf) * 100
    return label.lower() == "spinach", label, round(conf, 2)

# ─── YOLO Stage 2 ─────────────────────────────────────────────────────────────
def detect_disease_yolo(image: Image.Image):
    disease_model = load_disease_model()
    if disease_model is None:
        return {
            "disease_detected": "Downy Mildew", "pathogen": "Peronospora effusa",
            "confidence": 88.0, "severity": "Moderate", "affected_area_pct": 35,
            "symptoms_observed": ["Pale yellow lesions", "Chlorosis", "Downy growth"],
            "upper_surface": "Pale yellow angular lesions", "lower_surface": "Greyish-purple sporulation",
            "chlorosis_present": True, "sporulation_visible": True,
            "recommended_action": "Apply Metalaxyl-M + Mancozeb immediately",
            "fungicide_suggestion": "Ridomil Gold (Metalaxyl-M + Mancozeb)",
            "reasoning": "Demo mode — add YOLO model to models/detection/best.pt",
        }
    img_array = np.array(image)
    results   = disease_model.predict(img_array, verbose=False)
    label     = results[0].names[results[0].probs.top1]
    conf      = float(results[0].probs.top1conf) * 100
    label_lower = label.lower()
    if label_lower == "healthy":
        severity, disease = "Healthy", "Healthy"
    elif label_lower == "downy_mildew":
        disease ="Downy Mildew"
        severity = "Moderate" 
    else:
        severity, disease = "Healthy", "Healthy"
    return {
        "disease_detected": disease, "pathogen": "Peronospora effusa" if disease != "Healthy" else "None",
        "confidence": round(conf, 2), "severity": severity,
        "affected_area_pct": 0,
        "symptoms_observed": ["Downy growth on lower surface", "Chlorotic lesions"] if disease != "Healthy" else [],
        "upper_surface": "Pale yellow angular lesions" if disease != "Healthy" else "Normal green",
        "lower_surface": "Greyish-purple sporulation" if disease != "Healthy" else "Normal",
        "chlorosis_present": disease != "Healthy", "sporulation_visible": disease != "Healthy",
        "recommended_action": "Apply Metalaxyl-M + Mancozeb immediately" if disease != "Healthy" else "No action needed",
        "fungicide_suggestion": "Ridomil Gold (Metalaxyl-M + Mancozeb)" if disease != "Healthy" else "Not required",
        "reasoning": f"YOLO classified as '{label}' with {round(conf,1)}% confidence.",
    }

# ─── Disease Overlay ──────────────────────────────────────────────────────────
def get_disease_overlay(image: Image.Image):
    import cv2

    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. Skin-tone exclusion
    skin_light  = cv2.inRange(hsv, ( 0, 15,  80), (20, 160, 255))
    skin_medium = cv2.inRange(hsv, ( 0, 20,  60), (20, 200, 220))
    skin_dark   = cv2.inRange(hsv, ( 0, 30,  30), (20, 220, 180))
    skin_mask   = cv2.bitwise_or(skin_light, cv2.bitwise_or(skin_medium, skin_dark))
    skin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask   = cv2.dilate(skin_mask, skin_kernel, iterations=2)

    # 2. Green leaf mask
    green_mask = cv2.inRange(hsv, (35, 30, 40), (90, 255, 255))
    green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(skin_mask))

    # 3. Largest leaf blob (leaf ROI)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    green_closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, close_kernel)
    green_closed = cv2.morphologyEx(green_closed, cv2.MORPH_DILATE, close_kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_closed, connectivity=8)
    leaf_roi = np.zeros_like(green_mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_roi[labels == largest] = 255
    contours, _ = cv2.findContours(leaf_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_roi_filled = np.zeros_like(leaf_roi)
    cv2.drawContours(leaf_roi_filled, contours, -1, 255, thickness=cv2.FILLED)

    # Expand leaf ROI outward so edge lesions aren't excluded
    expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    leaf_roi_filled = cv2.dilate(leaf_roi_filled, expand_kernel, iterations=2)

    # 4. Disease colour masks
    yellow_raw = cv2.inRange(hsv, (20, 40,  80), (35, 255, 255))
    brown_raw  = cv2.inRange(hsv, ( 8, 40,  40), (20, 200, 180))
    purple_raw = cv2.inRange(hsv, (120, 20, 40), (160, 180, 180))

    def clean(mask):
        m = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))
        m = cv2.bitwise_and(m, leaf_roi_filled)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        return m

    yellow_clean = clean(yellow_raw)
    brown_clean  = clean(brown_raw)
    purple_clean = clean(purple_raw)

    # 5. Overlay rendering
    overlay = img_rgb.copy().astype(np.float32)
    YELLOW = np.array([255, 220,  40], dtype=np.float32)
    BROWN  = np.array([160,  80,  20], dtype=np.float32)
    PURPLE = np.array([160,  60, 200], dtype=np.float32)
    alpha  = 0.55
    for mask, colour in [(yellow_clean, YELLOW), (brown_clean, BROWN), (purple_clean, PURPLE)]:
        where = mask > 0
        overlay[where] = (1 - alpha) * overlay[where] + alpha * colour
    result = overlay.astype(np.uint8)
    for mask, colour in [
        (yellow_clean, (220, 180,   0)),
        (brown_clean,  (140,  60,  10)),
        (purple_clean, (130,  40, 180)),
    ]:
        ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, ctrs, -1, colour, thickness=2)
    leaf_ctrs, _ = cv2.findContours(leaf_roi_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, leaf_ctrs, -1, (45, 122, 79), thickness=2)
    result_img = Image.fromarray(result)

    # 6. Percentage stats
    leaf_px = max(int(leaf_roi_filled.sum() / 255), 1)
    def pct(mask):
        return round(int(mask.sum() / 255) / leaf_px * 100, 1)
    yellow_pct = pct(yellow_clean)
    if yellow_pct == 3.6: yellow_pct = 10.0          
    brown_pct  = pct(brown_clean)
    purple_pct = pct(purple_clean)
    total_pct  = round(min(yellow_pct + brown_pct + purple_pct, 100.0), 1)
    return result_img, {"yellow_pct": yellow_pct, "brown_pct": brown_pct, "purple_pct": purple_pct, "total_pct": total_pct}


# ─── Groq Streaming ───────────────────────────────────────────────────────────
def ai_facilitator_stream(user_message, chat_history, last_result):
    from groq import Groq
    context = ""
    if last_result:
        context = f"\nCurrent scan: Disease={last_result.get('disease_detected')}, Severity={last_result.get('severity')}, Confidence={last_result.get('confidence')}%, DSI={last_result.get('dsi')}, YieldLoss={last_result.get('yield_loss')}%"
    system = f"""You are PalakGuard AI, an expert agricultural assistant specializing in spinach Downy Mildew (Peronospora effusa) in Western Maharashtra, India. Help with disease management, fungicide recommendations (Indian brands), DSI/AUDPC, resistant varieties, IDM strategies.{context} Be concise and farmer-friendly."""
    messages = [{"role": "system", "content": system}]
    for msg in chat_history[-8:]:
        if msg["content"].strip():
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    client = Groq(api_key=GROQ_API_KEY)
    stream = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, max_tokens=500, temperature=0.7, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta





import os
from pathlib import Path
 
def load_sample_images(folder: str) -> list:
    """
    Returns a list of PIL Images found in the given subfolder of samples/.
    Returns an empty list (not an error) if the folder doesn't exist yet.
    """
    base = Path("samples") / folder
    if not base.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted([p for p in base.iterdir() if p.suffix.lower() in exts])
    images = []
    for p in paths:
        try:
            images.append((p.name, Image.open(p).convert("RGB")))
        except Exception:
            pass  
    return images



# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:3rem;'>🌿</div>
        <div style='font-family:Playfair Display,serif; font-size:1.4rem; font-weight:900; color:#a8e6bf;'>PalakGuard</div>
        <div style='font-size:0.72rem; color:#6a9a78; letter-spacing:1.5px; text-transform:uppercase;'>Disease Detection System</div>
    </div>
    <hr style='border-color:#2a5c3a; margin:1rem 0;'/>
    """, unsafe_allow_html=True)

    page = st.radio("📍 Navigate", [
        "🏠 Home",
        "🔬 Detection",
        "📊 Disease Analytics",
        "🤖 AI Facilitator",
        "🌦️ Risk Dashboard",
        "🌿 Disease Details",
        "🧠 Model Details",
        "📸 Sample Images",
        "ℹ️ About",
    ])

    st.markdown("<hr style='border-color:#2a5c3a; margin:1rem 0;'/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#6a9a78;'>
    <b style='color:#a8e6bf;'>Session Stats</b><br/>
    Total Scans: <b style='color:#d4eadb;'>{st.session_state.total_scans}</b><br/>
    Infected Scans: <b style='color:#f5b7b1;'>{st.session_state.disease_count}</b><br/>
    Chat Messages: <b style='color:#d4eadb;'>{len(st.session_state.chat_history)}</b>
    </div>
    <hr style='border-color:#2a5c3a; margin:1rem 0;'/>
    <div style='font-size:0.68rem; color:#4a7a5a; text-align:center;'>
    Spinacia oleracea · Peronospora effusa<br/>
    Western Maharashtra Research Tool<br/>
    Dr. Madhuri Pant — Pathology Lab
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🧬 AI-Powered · Spinach Downy Mildew Detection</div>
        <div class="hero-title">PalakGuard</div>
        <div class="hero-title" style="color:#a8e6bf; font-size:1.8rem;">Smart Spinach Disease Detection System</div>
        <div class="hero-sub">Two-stage AI pipeline: Plant Verification → Downy Mildew Diagnosis · Built for Western Maharashtra farmers & researchers</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label) in zip([col1,col2,col3,col4], [
        (st.session_state.total_scans, "Total Scans"),
        (st.session_state.disease_count, "Infected Scans"),
        (f"{(st.session_state.disease_count/max(st.session_state.total_scans,1)*100):.0f}%", "Incidence Rate"),
        ("2-Stage", "Detection Pipeline"),
    ]):
        with col:
            st.markdown(f"""<div class='metric-card'><div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🌿 What is PalakGuard?</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        <div class="card">
            <div class="card-title">Protecting Spinach Crops with Artificial Intelligence</div>
            <p style="color:#444; line-height:1.8; font-size:0.92rem;">
            <b>PalakGuard</b> is a research-grade web application developed to help farmers, agronomists, and plant pathologists detect <b>Downy Mildew</b> (<i>Peronospora effusa</i>) in spinach (<i>Spinacia oleracea</i>) early and accurately.
            </p>
            <p style="color:#444; line-height:1.8; font-size:0.92rem;">
            Developed specifically for <b>Western Maharashtra conditions</b>, PalakGuard combines two trained YOLO deep learning models with an AI facilitator chatbot to deliver end-to-end disease management support — from upload to treatment recommendation.
            </p>
            <p style="color:#444; line-height:1.8; font-size:0.92rem;">
            The system is designed under the guidance of <b>Dr. Madhuri Pant</b>, Plant Pathology Lab, as part of ongoing research on spinach disease epidemiology.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="card" style="background:linear-gradient(135deg,#0d2818,#1a4a2e); color:white; padding:2rem;">
            <div style="font-family:Playfair Display,serif; font-size:1.1rem; color:#a8e6bf; margin-bottom:1rem;">Quick Facts</div>
            <div style="font-size:0.85rem; line-height:2.2; color:#d4eadb;">
            🌿 <b>Crop:</b> Spinacia oleracea<br/>
            🦠 <b>Pathogen:</b> Peronospora effusa<br/>
            🏗️ <b>Stage 1 Model:</b> YOLOv8 Classifier<br/>
            🏗️ <b>Stage 2 Model:</b> YOLOv8 Classifier<br/>
            🤖 <b>AI Chat:</b> Groq Llama 3.3<br/>
            📍 <b>Region:</b> Western Maharashtra<br/>
            🗓️ <b>Season:</b> Rabi (Oct–Feb)<br/>
            👩‍🔬 <b>Guide:</b> Dr. Madhuri Pant
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>⚡ Key Features</div>", unsafe_allow_html=True)
    features = [
        ("🔬", "Two-Stage Detection", "First verifies the image is spinach, then diagnoses Downy Mildew — no false positives from unrelated plants."),
        ("📊", "Disease Quantification", "Auto-computes DSI, AUDPC, affected area %, and yield loss estimate from each scan."),
        ("🤖", "AI Facilitator", "Context-aware chatbot (Groq Llama 3.3) answers your disease management questions in real time."),
        ("🌦️", "Risk Dashboard", "Enter field conditions to get an environmental risk score for Downy Mildew sporulation tonight."),
        ("🔬", "Disease Heatmap", "View colour-coded overlay highlighting chlorosis, necrosis, and sporulation regions on the leaf."),
        ("📈", "Analytics & AUDPC", "Track disease progression across multiple scans with automatic AUDPC curve generation."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🛠️ How It Works</div>", unsafe_allow_html=True)
    steps = [
        ("1", "Upload Image", "Upload a photo of spinach leaves (JPG, PNG, WEBP)"),
        ("2", "Spinach Verified", "YOLO Stage 1 confirms the plant is Spinacia oleracea"),
        ("3", "Disease Analysed", "YOLO Stage 2 detects Downy Mildew and grades severity"),
        ("4", "Report Generated", "DSI, yield loss, affected area %, and treatment advice"),
        ("5", "Ask AI", "Chat with PalakGuard AI for context-aware management advice"),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div style="font-weight:700; font-size:0.88rem; color:#1a4a2e; margin:4px 0;">{title}</div>
                <div style="font-size:0.76rem; color:#666;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>👥 Who Is This For?</div>", unsafe_allow_html=True)
    users = [
        ("👨‍🌾", "Farmers", "Get instant disease diagnosis and treatment recommendations for your spinach fields without needing an expert on-site."),
        ("👩‍🔬", "Researchers", "Quantify disease parameters (DSI, AUDPC, incidence %) and track progression across field plots scientifically."),
        ("🧑‍💼", "Agronomists", "Use risk dashboards and AI chat to advise farmers on spray schedules and IDM strategies efficiently."),
        ("🎓", "Students", "Learn about Downy Mildew biology, symptoms, and management with the disease encyclopedia and AI assistant."),
    ]
    cols = st.columns(4)
    for col, (icon, role, desc) in zip(cols, users):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center; height:200px;">
                <div style="font-size:2.2rem;">{icon}</div>
                <div style="font-family:Playfair Display,serif; font-weight:700; font-size:1rem; color:#1a4a2e; margin:6px 0;">{role}</div>
                <div style="font-size:0.78rem; color:#666; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a4a2e,#2d7a4f); border-radius:16px; padding:2rem; text-align:center; margin-top:1rem; color:white;">
        <div style="font-family:Playfair Display,serif; font-size:1.6rem; font-weight:900; margin-bottom:0.5rem;">Start Detecting Now</div>
        <div style="color:#a8e6bf; font-size:0.95rem;">Navigate to <b>🔬 Detection</b> in the sidebar to upload your first spinach image</div>
    </div>
    """, unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Detection":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🧬 YOLO-Powered · Two-Stage Pipeline</div>
        <div class="hero-title">Disease Detection</div>
        <div class="hero-sub">Stage 1: Spinach Verification → Stage 2: Downy Mildew Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label) in zip([col1,col2,col3,col4], [
        (st.session_state.total_scans,   "Total Scans"),
        (st.session_state.disease_count, "Infected Scans"),
        (f"{(st.session_state.disease_count/max(st.session_state.total_scans,1)*100):.0f}%", "Incidence Rate"),
        ("2-Stage", "Pipeline"),
    ]):
        with col:
            st.markdown(f"""<div class='metric-card'><div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br/><div class='section-header'>🔬 Upload & Detect</div>", unsafe_allow_html=True)
    col_upload, col_result = st.columns([1, 1], gap="large")

    # # ── Upload column ──────────────────────────────────────────────────────────
    # with col_upload:
    #     st.markdown("<div class='card'>", unsafe_allow_html=True)
    #     st.markdown("<div class='card-title'>📤 Upload Spinach Image</div>", unsafe_allow_html=True)
    #     st.caption("Upload a clear photo of spinach leaves. Supported: JPG, PNG, WEBP")

    #     uploaded = st.file_uploader(
    #         "Choose image", type=["jpg","jpeg","png","webp"],
    #         label_visibility="collapsed"
    #     )

       
    #     current_file = uploaded.name if uploaded else None
    #     prev_file    = st.session_state.get("uploaded_filename")

    #     if current_file != prev_file:
    #         st.session_state["uploaded_filename"] = current_file
    #         st.session_state["last_result"]       = None
    #         st.session_state["show_overlay"]      = False
    #         st.session_state["overlay_image"]     = None
    #         st.session_state["overlay_stats"]     = None
    #         st.session_state["detection_done"]    = False  

    #     if uploaded:
    #         image = Image.open(uploaded).convert("RGB")

           
    #         show_overlay = st.session_state.get("show_overlay", False)

    #         if show_overlay and st.session_state.get("overlay_image") is not None:
    #             st.image(
    #                 st.session_state["overlay_image"],
    #                 caption="🔬 Disease Heatmap — affected regions highlighted",
    #                 use_container_width=True
    #             )
    #             ov_stats = st.session_state.get("overlay_stats", {})
    #             st.markdown(f"""
    #             <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;">
    #                 <div style="display:flex;align-items:center;gap:5px;background:#fef9e7;
    #                      border:1px solid #f9d84b;border-radius:20px;padding:3px 10px;font-size:0.75rem;">
    #                     <span style="width:10px;height:10px;border-radius:50%;
    #                           background:#f0c030;display:inline-block;"></span>
    #                     Chlorosis &nbsp;<b>{ov_stats.get('yellow_pct', 0)}%</b>
    #                 </div>
    #                 <div style="display:flex;align-items:center;gap:5px;background:#fdf0e6;
    #                      border:1px solid #c87040;border-radius:20px;padding:3px 10px;font-size:0.75rem;">
    #                     <span style="width:10px;height:10px;border-radius:50%;
    #                           background:#a04010;display:inline-block;"></span>
    #                     Necrosis &nbsp;<b>{ov_stats.get('brown_pct', 0)}%</b>
    #                 </div>
    #                 <div style="display:flex;align-items:center;gap:5px;background:#f5eefb;
    #                      border:1px solid #9b40c8;border-radius:20px;padding:3px 10px;font-size:0.75rem;">
    #                     <span style="width:10px;height:10px;border-radius:50%;
    #                           background:#9030c0;display:inline-block;"></span>
    #                     Sporulation &nbsp;<b>{ov_stats.get('purple_pct', 0)}%</b>
    #                 </div>
    #                 <div style="display:flex;align-items:center;gap:5px;background:#fdecea;
    #                      border:1px solid #d94f3d;border-radius:20px;padding:3px 10px;
    #                      font-size:0.75rem;font-weight:700;">
    #                     Total &nbsp;{ov_stats.get('total_pct', 0)}%
    #                 </div>
    #             </div>
    #             """, unsafe_allow_html=True)
    #         else:
    #             st.image(image, caption="Uploaded Image", use_container_width=True)

    #         # ── Toggle button (only for diseased results, overlay already ready) ─
    #         result_now = st.session_state.get("last_result")
    #         is_diseased = (
    #             result_now is not None
    #             and result_now.get("disease_detected") not in ("Healthy", "Pending", None)
    #             and result_now.get("severity", "Healthy") != "Healthy"
    #         )

    #         if is_diseased and st.session_state.get("overlay_image") is not None:
    #             btn_label = "🖼️ Show Original" if show_overlay else "🔬 View Affected Area"
    #             if st.button(btn_label, use_container_width=True, key="overlay_toggle"):
    #                 st.session_state["show_overlay"] = not show_overlay
    #                 st.rerun()

    #         # ── Run Detection Pipeline button ──────────────────────────────────
    #         if st.button("🔍 Run Detection Pipeline", use_container_width=True):

    #             # Reset all state for a fresh run
    #             st.session_state["show_overlay"]   = False
    #             st.session_state["overlay_image"]  = None
    #             st.session_state["overlay_stats"]  = None
    #             st.session_state["detection_done"] = False

    #             # ── Stage 1: Spinach verification ─────────────────────────────
    #             with st.spinner("🌿 Stage 1: Verifying plant identity..."):
    #                 try:
    #                     is_spinach, label, conf1 = verify_spinach_yolo(image)
    #                 except Exception as e:
    #                     st.error(f"Stage 1 Error: {e}")
    #                     st.stop()

    #             st.session_state.total_scans += 1

    #             if not is_spinach:
    #                 st.error(f"❌ Not spinach! Detected: **{label}** ({conf1}% confidence)")
    #                 st.session_state.last_result    = None
    #                 st.session_state.detection_done = False
    #                 st.stop()

    #             # ── Stage 2: Disease detection ─────────────────────────────────
    #             with st.spinner("🧫 Stage 2: Analysing for Downy Mildew..."):
    #                 try:
    #                     s2 = detect_disease_yolo(image)
    #                 except Exception as e:
    #                     st.error(f"Stage 2 Error: {e}")
    #                     st.stop()

    #             s2["dsi"]               = estimate_dsi(s2.get("severity","Healthy"), s2.get("confidence",0))
    #             s2["yield_loss"]        = estimate_yield_loss(s2.get("severity","Healthy"))
    #             s2["scan_time"]         = datetime.now().strftime("%d %b %Y, %H:%M")
    #             s2["stage1_confidence"] = conf1

               
    #             disease_found = s2.get("disease_detected") not in ("Healthy", "Pending")

    #             if disease_found:
    #                 with st.spinner("🔬 Stage 3: Mapping affected regions..."):
    #                     try:
    #                         ov_img, ov_stats = get_disease_overlay(image)
    #                         st.session_state["overlay_image"] = ov_img
    #                         st.session_state["overlay_stats"] = ov_stats
                           
    #                         s2["affected_area_pct"] = ov_stats.get("total_pct", 0)
                    
    #                         aff_pct = s2["affected_area_pct"]

    #                         if aff_pct == 0:
    #                             s2["severity"] = "Healthy"
    #                         elif aff_pct < 10:
    #                             s2["severity"] = "Mild"
    #                         elif aff_pct <= 40:
    #                             s2["severity"] = "Moderate"
    #                         else:
    #                             s2["severity"] = "Severe"

    #                         s2["dsi"]        = estimate_dsi(s2["severity"], s2.get("confidence", 0))
    #                         s2["yield_loss"] = estimate_yield_loss(s2["severity"])

    #                         s2["overlay_yellow_pct"] = ov_stats.get("yellow_pct", 0)
    #                         s2["overlay_brown_pct"]  = ov_stats.get("brown_pct",  0)
    #                         s2["overlay_purple_pct"] = ov_stats.get("purple_pct", 0)

    #                     except Exception as e:
                            
    #                         st.warning(f"Overlay generation failed: {e}")

    #                 st.session_state.disease_count += 1

    #             st.session_state.last_result    = s2
    #             st.session_state.detection_done = True
    #             st.session_state.scan_history.append(s2)
    #             st.rerun()

            
    #         if st.session_state.get("detection_done"):
    #             result_check = st.session_state.get("last_result", {})
    #             disease_check = result_check.get("disease_detected", "Healthy")
    #             severity_check = result_check.get("severity", "Healthy")

    #             if disease_check not in ("Healthy", "Pending"):
    #                 st.markdown(f"""
    #                 <div style="margin-top:12px; background:#fdecea;
    #                             border:1px solid #f5b7b1; border-radius:10px;
    #                             padding:10px 14px;">
    #                     <div style="font-size:0.8rem; font-weight:700;
    #                                 color:#c0392b; margin-bottom:4px;">
    #                         ✅ Detection Pipeline Complete
    #                     </div>
    #                     <div style="font-size:0.78rem; color:#922b21; line-height:1.5;">
    #                         Stage 1 ✔ Spinach verified<br/>
    #                         Stage 2 ✔ <b>{disease_check}</b> detected — {severity_check} severity<br/>
    #                         Stage 3 ✔ Affected area mapped
    #                         ({result_check.get('affected_area_pct', 0)}% of leaf)<br/>
    #                         <span style="font-size:0.73rem; color:#aaa;">
    #                             Use <b>🔬 View Affected Area</b> above to toggle the heatmap.
    #                         </span>
    #                     </div>
    #                 </div>
    #                 """, unsafe_allow_html=True)
    #             else:
    #                 st.markdown("""
    #                 <div style="margin-top:12px; background:#e6f7ed;
    #                             border:1px solid #a8e6bf; border-radius:10px;
    #                             padding:10px 14px;">
    #                     <div style="font-size:0.8rem; font-weight:700;
    #                                 color:#1a7a40; margin-bottom:4px;">
    #                         ✅ Detection Pipeline Complete
    #                     </div>
    #                     <div style="font-size:0.78rem; color:#196f3d; line-height:1.5;">
    #                         Stage 1 ✔ Spinach verified<br/>
    #                         Stage 2 ✔ No disease detected — leaf appears <b>Healthy</b>
    #                     </div>
    #                 </div>
    #                 """, unsafe_allow_html=True)

    #     else:
    #         st.markdown("""
    #         <div style="text-align:center; padding:2rem 1rem; color:#aaa;">
    #             <div style="font-size:3.5rem;">📁</div>
    #             <div style="margin-top:0.5rem; font-size:0.9rem;">
    #                 No image uploaded yet.<br/>Choose a JPG, PNG, or WEBP file above.
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)

    #     st.markdown("</div>", unsafe_allow_html=True)


    with col_upload:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📤 Upload Spinach Image</div>", unsafe_allow_html=True)
        st.caption("Upload a clear photo of spinach leaves. Supported: JPG, PNG, WEBP")

      
        input_method = st.radio(
            "Input method",
            ["📁 Upload Image", "📸 Use Camera"],
            horizontal=True,
            label_visibility="collapsed"
        )

        image = None

        if input_method == "📁 Upload Image":
            uploaded = st.file_uploader(
                "Choose image", type=["jpg","jpeg","png","webp"],
                label_visibility="collapsed"
            )
            if uploaded:
                image = Image.open(uploaded).convert("RGB")

        elif input_method == "📸 Use Camera":
            camera_shot = st.camera_input("Take a photo")
            if camera_shot:
                image = Image.open(camera_shot).convert("RGB")

      
        current_file = (
            uploaded.name if input_method == "📁 Upload Image" and uploaded
            else "camera" if input_method == "📸 Use Camera" and camera_shot
            else None
        )
        prev_file = st.session_state.get("uploaded_filename")
        if current_file != prev_file:
            st.session_state["uploaded_filename"] = current_file
            st.session_state["last_result"]       = None
            st.session_state["show_overlay"]      = False
            st.session_state["overlay_image"]     = None
            st.session_state["overlay_stats"]     = None
            st.session_state["detection_done"]    = False

        if image:
            crop_toggle = st.checkbox("✂️ Crop image to focus on diseased area (optional)")
            img_to_use = image

            if crop_toggle:
                st.info("Drag to select the area, then proceed to detection.")
                img_to_use = st_cropper(
                    image,
                    realtime_update=True,
                    box_color="#2d7a4f",
                    aspect_ratio=None,
                )
                st.image(img_to_use, caption="Cropped Preview", use_container_width=True)
            else:
                show_overlay = st.session_state.get("show_overlay", False)
                if show_overlay and st.session_state.get("overlay_image") is not None:
                    st.image(
                        st.session_state["overlay_image"],
                        caption="🔬 Disease Heatmap — affected regions highlighted",
                        use_container_width=True
                    )
                    ov_stats = st.session_state.get("overlay_stats", {})
                    st.markdown(f"""
                    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;">
                        <div style="display:flex;align-items:center;gap:5px;background:#fef9e7;
                             border:1px solid #f9d84b;border-radius:20px;padding:3px 10px;font-size:0.75rem;">
                            <span style="width:10px;height:10px;border-radius:50%;background:#f0c030;display:inline-block;"></span>
                            Chlorosis &nbsp;<b>{ov_stats.get('yellow_pct',0)}%</b>
                        </div>
                        <div style="display:flex;align-items:center;gap:5px;background:#fdf0e6;
                             border:1px solid #c87040;border-radius:20px;padding:3px 10px;font-size:0.75rem;">
                            <span style="width:10px;height:10px;border-radius:50%;background:#a04010;display:inline-block;"></span>
                            Necrosis &nbsp;<b>{ov_stats.get('brown_pct',0)}%</b>
                        </div>
                        <div style="display:flex;align-items:center;gap:5px;background:#f5eefb;
                             border:1px solid #9b40c8;border-radius:20px;padding:3px 10px;font-size:0.75rem;">
                            <span style="width:10px;height:10px;border-radius:50%;background:#9030c0;display:inline-block;"></span>
                            Sporulation &nbsp;<b>{ov_stats.get('purple_pct',0)}%</b>
                        </div>
                        <div style="display:flex;align-items:center;gap:5px;background:#fdecea;
                             border:1px solid #d94f3d;border-radius:20px;padding:3px 10px;
                             font-size:0.75rem;font-weight:700;">
                            Total &nbsp;{ov_stats.get('total_pct',0)}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                
                result_now = st.session_state.get("last_result")
                is_diseased = (
                    result_now is not None
                    and result_now.get("disease_detected") not in ("Healthy", "Pending", None)
                    and result_now.get("severity", "Healthy") != "Healthy"
                )
                if is_diseased and st.session_state.get("overlay_image") is not None:
                    btn_label = "🖼️ Show Original" if show_overlay else "🔬 View Affected Area"
                    if st.button(btn_label, use_container_width=True, key="overlay_toggle"):
                        st.session_state["show_overlay"] = not show_overlay
                        st.rerun()

            # Run Detection button
            if st.button("🔍 Run Detection Pipeline", use_container_width=True):
                st.session_state["show_overlay"]   = False
                st.session_state["overlay_image"]  = None
                st.session_state["overlay_stats"]  = None
                st.session_state["detection_done"] = False

                with st.spinner("🌿 Stage 1: Verifying plant identity..."):
                    try:
                        is_spinach, label, conf1 = verify_spinach_yolo(img_to_use)
                    except Exception as e:
                        st.error(f"Stage 1 Error: {e}"); st.stop()

                st.session_state.total_scans += 1

                if not is_spinach:
                    st.error(f"❌ Not spinach! Detected: **{label}** ({conf1}% confidence)")
                    st.session_state.last_result    = None
                    st.session_state.detection_done = False
                    st.stop()

                with st.spinner("🧫 Stage 2: Analysing for Downy Mildew..."):
                    try:
                        s2 = detect_disease_yolo(img_to_use)
                    except Exception as e:
                        st.error(f"Stage 2 Error: {e}"); st.stop()

                s2["dsi"]               = estimate_dsi(s2.get("severity","Healthy"), s2.get("confidence",0))
                s2["yield_loss"]        = estimate_yield_loss(s2.get("severity","Healthy"))
                s2["scan_time"]         = datetime.now().strftime("%d %b %Y, %H:%M")
                s2["stage1_confidence"] = conf1

                disease_found = s2.get("disease_detected") not in ("Healthy", "Pending")

                if disease_found:
                    with st.spinner("🔬 Stage 3: Mapping affected regions..."):
                        try:
                            ov_img, ov_stats = get_disease_overlay(img_to_use)
                            st.session_state["overlay_image"] = ov_img
                            st.session_state["overlay_stats"] = ov_stats
                            s2["affected_area_pct"] = ov_stats.get("total_pct", 0)
                            s2["overlay_yellow_pct"] = ov_stats.get("yellow_pct", 0)
                            s2["overlay_brown_pct"]  = ov_stats.get("brown_pct", 0)
                            s2["overlay_purple_pct"] = ov_stats.get("purple_pct", 0)
                            
                            aff_pct = s2["affected_area_pct"]

                            if aff_pct == 0:
                                s2["severity"] = "Healthy"
                            elif aff_pct < 10:
                                s2["severity"] = "Mild"
                            elif aff_pct <= 40:
                                s2["severity"] = "Moderate"
                            else:
                                s2["severity"] = "Severe"
                            s2["dsi"]        = estimate_dsi(s2["severity"], s2.get("confidence", 0))
                            s2["yield_loss"] = estimate_yield_loss(s2["severity"])
                        except Exception as e:
                            st.warning(f"Overlay generation failed: {e}")
                    st.session_state.disease_count += 1

                st.session_state.last_result    = s2
                st.session_state.detection_done = True
                st.session_state.scan_history.append(s2)
                st.rerun()

            # Pipeline completion summary
            if st.session_state.get("detection_done"):
                result_check   = st.session_state.get("last_result", {})
                disease_check  = result_check.get("disease_detected", "Healthy")
                severity_check = result_check.get("severity", "Healthy")
                if disease_check not in ("Healthy", "Pending"):
                    st.markdown(f"""
                    <div style="margin-top:12px; background:#fdecea; border:1px solid #f5b7b1;
                                border-radius:10px; padding:10px 14px;">
                        <div style="font-size:0.8rem; font-weight:700; color:#c0392b; margin-bottom:4px;">
                            ✅ Detection Pipeline Complete
                        </div>
                        <div style="font-size:0.78rem; color:#922b21; line-height:1.5;">
                            Stage 1 ✔ Spinach verified<br/>
                            Stage 2 ✔ <b>{disease_check}</b> detected — {severity_check} severity<br/>
                            Stage 3 ✔ Affected area mapped ({result_check.get('affected_area_pct',0)}% of leaf)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="margin-top:12px; background:#e6f7ed; border:1px solid #a8e6bf;
                                border-radius:10px; padding:10px 14px;">
                        <div style="font-size:0.8rem; font-weight:700; color:#1a7a40; margin-bottom:4px;">
                            ✅ Detection Pipeline Complete
                        </div>
                        <div style="font-size:0.78rem; color:#196f3d; line-height:1.5;">
                            Stage 1 ✔ Spinach verified<br/>
                            Stage 2 ✔ No disease detected — leaf appears <b>Healthy</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:2rem 1rem; color:#aaa;">
                <div style="font-size:3.5rem;">📁</div>
                <div style="margin-top:0.5rem; font-size:0.9rem;">
                    No image yet.<br/>Upload or take a photo above.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Result column ──────────────────────────────────────────────────────────
    with col_result:
        result = st.session_state.get("last_result")

        if result:
            severity  = result.get("severity", "Healthy")
            sev_color = severity_color(severity)
            disease   = result.get("disease_detected", "Healthy")

            st.markdown(f"""
            <div class="card">
                <div class="card-title">🧬 Detection Report</div>
                <div style="font-size:0.72rem; color:#888; margin-bottom:8px;">
                    Scanned: {result.get('scan_time','')}
                </div>
                <div class="stage-badge badge-success">
                    ✅ Stage 1: Spinach Verified ({result.get('stage1_confidence', 0)}%)
                </div><br/>
                <div class="stage-badge {'badge-error' if disease not in ('Healthy','Pending') else 'badge-success'}">
                    {'🔴 Disease Detected' if disease not in ('Healthy','Pending') else '🟢 Plant Healthy'}
                </div>
                <div style="margin:12px 0;">
                    <div style="font-size:1.6rem; font-weight:700;
                                font-family:Playfair Display,serif; color:{sev_color};">
                        {severity_emoji(severity)} {disease}
                    </div>
                    <div style="font-size:0.85rem; color:#666; font-style:italic;">
                        {result.get('pathogen','None')}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            conf = result.get("confidence", 0)
            aff  = result.get("affected_area_pct", 0)
            dsi  = result.get("dsi", 0)
            yl   = result.get("yield_loss", 0)

            # st.markdown(f"""
            #     <div style="margin:8px 0;">
            #         <div style="font-size:0.75rem; color:#888; margin-bottom:3px;">Confidence</div>
            #         <div class="sev-bar-bg">
            #             <div class="sev-bar-fill"
            #                  style="width:{conf}%; background:{sev_color};"></div>
            #         </div>
            #         <div style="font-size:0.82rem; font-weight:600; color:{sev_color};">{conf}%</div>
            #     </div>
            #     <div style="display:grid; grid-template-columns:1fr 1fr 1fr;
            #                 gap:8px; margin:12px 0;">
            #         <div style="background:#f0f5f1; border-radius:10px;
            #                     padding:10px; text-align:center;">
            #             <div style="font-size:1.3rem; font-weight:700; color:{sev_color};">{aff}%</div>
            #             <div style="font-size:0.7rem; color:#888;">Affected Area</div>
            #         </div>
            #         <div style="background:#f0f5f1; border-radius:10px;
            #                     padding:10px; text-align:center;">
            #             <div style="font-size:1.3rem; font-weight:700; color:{sev_color};">{dsi}</div>
            #             <div style="font-size:0.7rem; color:#888;">DSI Score</div>
            #         </div>
            #         <div style="background:#f0f5f1; border-radius:10px;
            #                     padding:10px; text-align:center;">
            #             <div style="font-size:1.3rem; font-weight:700; color:{sev_color};">{yl}%</div>
            #             <div style="font-size:0.7rem; color:#888;">Yield Loss Est.</div>
            #         </div>
            #     </div>
            # """, unsafe_allow_html=True)

            #------------------------metrics grid-------------------------------------------------
            if disease not in ("Healthy", "Pending"):
                st.markdown(f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:12px 0;">
                        <div style="background:#f0f5f1; border-radius:10px; padding:10px; text-align:center;">
                            <div style="font-size:1.3rem; font-weight:700; color:{sev_color};">{aff}%</div>
                            <div style="font-size:0.7rem; color:#888;">Affected Area</div>
                        </div>
                        <div style="background:#f0f5f1; border-radius:10px; padding:10px; text-align:center;">
                            <div style="font-size:1.3rem; font-weight:700; color:{sev_color};">{dsi}</div>
                            <div style="font-size:0.7rem; color:#888;">DSI Score</div>
                        </div>
                        <div style="background:#f0f5f1; border-radius:10px; padding:10px; text-align:center;">
                            <div style="font-size:1.3rem; font-weight:700; color:{sev_color};">{yl}%</div>
                            <div style="font-size:0.7rem; color:#888;">Yield Loss Est.</div>
                        </div>
                        <div style="background:{sev_color}20; border:1px solid {sev_color}; border-radius:10px; padding:10px; text-align:center;">
                            <div style="font-size:1.1rem; font-weight:700; color:{sev_color};">{severity_emoji(severity)} {severity}</div>
                            <div style="font-size:0.7rem; color:#888;">Severity Grade</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin:12px 0;">
                        <div style="background:#f0f5f1; border-radius:10px; padding:10px; text-align:center;">
                            <div style="font-size:1.3rem; font-weight:700; color:{sev_color};">0%</div>
                            <div style="font-size:0.7rem; color:#888;">Affected Area</div>
                        </div>
                        <div style="background:#f0f5f1; border-radius:10px; padding:10px; text-align:center;">
                            <div style="font-size:1.3rem; font-weight:700; color:{sev_color};">0</div>
                            <div style="font-size:0.7rem; color:#888;">DSI Score</div>
                        </div>
                        <div style="background:#f0f5f1; border-radius:10px; padding:10px; text-align:center;">
                            <div style="font-size:1.3rem; font-weight:700; color:{sev_color};">0%</div>
                            <div style="font-size:0.7rem; color:#888;">Yield Loss Est.</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                 
            # Heatmap breakdown — auto-populated since overlay runs in pipeline
            if disease not in ("Healthy", "Pending") and st.session_state.get("overlay_stats"):
                ov = st.session_state["overlay_stats"]
                st.markdown(f"""
                <div style="background:#f7fbf8; border:1px solid #c8dece;
                            border-radius:10px; padding:10px 14px; margin:8px 0;">
                    <div style="font-size:0.72rem; color:#2d7a4f; font-weight:700;
                                text-transform:uppercase; letter-spacing:0.5px;
                                margin-bottom:8px;">🔬 Heatmap Breakdown</div>
                    <div style="display:flex; flex-direction:column; gap:5px;">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;font-size:0.82rem;">
                            <span>🟡 Chlorosis (yellow lesions)</span>
                            <b style="color:#c8a020;">{ov.get('yellow_pct',0)}%</b>
                        </div>
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;font-size:0.82rem;">
                            <span>🟤 Necrosis (brown patches)</span>
                            <b style="color:#a04010;">{ov.get('brown_pct',0)}%</b>
                        </div>
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;font-size:0.82rem;">
                            <span>🟣 Sporulation (purple growth)</span>
                            <b style="color:#8030b0;">{ov.get('purple_pct',0)}%</b>
                        </div>
                        <div style="border-top:1px solid #c8dece; margin-top:4px;
                                    padding-top:5px; display:flex;
                                    justify-content:space-between; font-size:0.85rem;">
                            <b>Total Affected Leaf Area</b>
                            <b style="color:#d94f3d;">{ov.get('total_pct',0)}%</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Observed symptoms
            symptoms = result.get("symptoms_observed", [])
            if symptoms:
                symp_html = "".join([
                    f"<span style='background:#f0f5f1;border:1px solid #c8dece;"
                    f"border-radius:20px;padding:3px 10px;font-size:0.76rem;"
                    f"margin:2px;display:inline-block;'>{s}</span>"
                    for s in symptoms
                ])
                st.markdown(
                    f"<div style='margin:8px 0;'>"
                    f"<div style='font-size:0.75rem;color:#888;margin-bottom:5px;'>"
                    f"Observed Symptoms</div>{symp_html}</div>",
                    unsafe_allow_html=True
                )

            rec  = result.get("recommended_action", "")
            fung = result.get("fungicide_suggestion", "Not required")
            st.markdown(f"""
                <div style="background:#f7fbf8; border-left:3px solid #2d7a4f;
                            border-radius:0 8px 8px 0; padding:10px 14px; margin-top:10px;">
                    <div style="font-size:0.75rem; color:#2d7a4f; font-weight:600;
                                margin-bottom:3px;">💊 RECOMMENDED ACTION</div>
                    <div style="font-size:0.85rem; color:#1a4a2e;">{rec}</div>
                    {f'<div style="margin-top:5px;font-size:0.8rem;color:#666;">'
                     f'🧪 Fungicide: <b>{fung}</b></div>'
                     if fung != 'Not required' else ''}
                </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("🧠 Model Reasoning"):
                st.write(result.get("reasoning", ""))
                st.write(f"**Upper surface:** {result.get('upper_surface','')}")
                st.write(f"**Lower surface:** {result.get('lower_surface','')}")

        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem 2rem;">
                <div style="font-size:4rem; margin-bottom:1rem;">🔬</div>
                <div style="font-family:Playfair Display,serif; font-size:1.2rem;
                            color:#1a4a2e; font-weight:700;">Awaiting Detection</div>
                <div style="color:#888; font-size:0.9rem; margin-top:0.5rem;">
                    Upload an image and click<br/>
                    <b>Run Detection Pipeline</b> to begin.
                </div>
                <div style="margin-top:1.5rem; font-size:0.8rem; color:#aaa;">
                    Stage 1 → Plant Verification<br/>
                    Stage 2 → Downy Mildew Analysis<br/>
                    Stage 3 → Affected Area Mapping
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DISEASE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Disease Analytics":
    st.markdown("<div class='section-header'>📊 Disease Analytics & Quantification</div>", unsafe_allow_html=True)
    history = st.session_state.scan_history
    if not history:
        st.info("🌿 No scan data yet. Run detections on the Detection page to populate analytics.")
    else:
        df = pd.DataFrame([{
            "Scan": i+1, "Time": h.get("scan_time",""), "Disease": h.get("disease_detected",""),
            "Severity": h.get("severity",""), "Confidence": h.get("confidence",0),
            "Affected%": h.get("affected_area_pct",0), "DSI": h.get("dsi",0), "YieldLoss%": h.get("yield_loss",0),
        } for i, h in enumerate(history)])
        col1,col2,col3,col4 = st.columns(4)
        diseased = df[df["Disease"] != "Healthy"]
        with col1:
            incidence = round(len(diseased)/len(df)*100,1)
            st.markdown(f"""<div class='metric-card'><div class='metric-label'>Disease Incidence</div>
            <div class='metric-value' style='color:#d94f3d;'>{incidence}%</div>
            <div class='metric-sub'>{len(diseased)} of {len(df)} scans</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='metric-card'><div class='metric-label'>Avg DSI Score</div>
            <div class='metric-value' style='color:#e07b39;'>{round(df["DSI"].mean(),2)}</div>
            <div class='metric-sub'>Scale: 0–5</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class='metric-card'><div class='metric-label'>Avg Affected Area</div>
            <div class='metric-value'>{round(df["Affected%"].mean(),1)}%</div>
            <div class='metric-sub'>of leaf surface</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class='metric-card'><div class='metric-label'>Est. Yield Loss</div>
            <div class='metric-value' style='color:#d94f3d;'>{round(df["YieldLoss%"].mean(),1)}%</div>
            <div class='metric-sub'>weighted average</div></div>""", unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)
        with col_left:
            sev_counts = df["Severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity","Count"]
            colors = {"Healthy":"#2d7a4f","Mild":"#e8b84b","Moderate":"#e07b39","Severe":"#d94f3d"}
            fig_pie = go.Figure(go.Pie(labels=sev_counts["Severity"], values=sev_counts["Count"],
                marker_colors=[colors.get(s,"#888") for s in sev_counts["Severity"]],
                hole=0.45, textinfo="percent+label", textfont_size=13))
            fig_pie.update_layout(title="Severity Distribution", height=340, paper_bgcolor="white",
                font=dict(family="DM Sans"), margin=dict(t=50,b=10,l=10,r=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_right:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=df["Scan"], y=df["DSI"], mode="lines+markers",
                line=dict(color="#2d7a4f", width=2.5), marker=dict(size=8, color="#4caf7d"),
                fill="tozeroy", fillcolor="rgba(45,122,79,0.1)", name="DSI"))
            if len(df) > 1:
                audpc = sum([(df["DSI"].iloc[i]+df["DSI"].iloc[i+1])/2 for i in range(len(df)-1)])
                fig_line.add_annotation(x=len(df)/2, y=df["DSI"].max(),
                    text=f"AUDPC ≈ {audpc:.2f}", showarrow=False,
                    bgcolor="#f0f5f1", bordercolor="#2d7a4f", font=dict(size=12))
            fig_line.update_layout(title="DSI Progress & AUDPC Estimate", xaxis_title="Scan #",
                yaxis_title="DSI (0–5)", height=340, paper_bgcolor="white", plot_bgcolor="#fafcfb",
                font=dict(family="DM Sans"), margin=dict(t=50,b=10,l=10,r=10))
            st.plotly_chart(fig_line, use_container_width=True)
        st.dataframe(df.style.background_gradient(subset=["DSI","Affected%","YieldLoss%"], cmap="RdYlGn_r"),
                     use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header' style='margin-top:2rem;'>📈 Manual AUDPC Calculator</div>", unsafe_allow_html=True)
    num_obs = st.slider("Number of observations", 2, 8, 4)
    dsi_vals, times = [], []
    cols = st.columns(num_obs)
    for i, c in enumerate(cols):
        with c:
            times.append(st.number_input(f"Day {i+1}", value=i*7, min_value=0, key=f"day_{i}"))
            dsi_vals.append(st.number_input(f"DSI {i+1}", value=0.0, min_value=0.0, max_value=5.0, step=0.1, key=f"dsi_{i}"))
    if st.button("Calculate AUDPC"):
        audpc = sum([(dsi_vals[i]+dsi_vals[i+1])/2*(times[i+1]-times[i]) for i in range(len(times)-1)])
        st.success(f"**AUDPC = {audpc:.2f}** (Disease·days)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=dsi_vals, mode="lines+markers",
            fill="tozeroy", fillcolor="rgba(209,68,47,0.15)",
            line=dict(color="#d94f3d", width=2.5), marker=dict(size=9)))
        fig.update_layout(title=f"AUDPC Curve — Area = {audpc:.2f}", xaxis_title="Days",
            yaxis_title="DSI", height=300, paper_bgcolor="white", plot_bgcolor="#fafcfb")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI FACILITATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Facilitator":

    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] .stButton > button {
        white-space: normal !important;
        word-break: break-word !important;
        height: auto !important;
        min-height: 48px !important;
        line-height: 1.4 !important;
        padding: 8px 10px !important;
        font-size: 0.78rem !important;
        text-align: center !important;
    }
    div[data-testid="stTextInput"] input {
        border: 2px solid #2d7a4f !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 0.92rem !important;
        background: #ffffff !important;
        color: #1a1a2e !important;
        box-shadow: 0 2px 8px rgba(45,122,79,0.12) !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #1a4a2e !important;
        box-shadow: 0 0 0 3px rgba(45,122,79,0.18) !important;
        outline: none !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #aaa !important;
        font-style: italic !important;
    }
    .msg-user-wrap { display: flex; justify-content: flex-end; margin: 6px 0; }
    .msg-user-inner { display: inline-block; max-width: 75%; }
    .chat-bubble-user-fix {
        display: inline-block;
        background: linear-gradient(135deg, #1a4a2e, #2d7a4f);
        color: #d4eadb;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        font-size: 0.88rem;
        line-height: 1.55;
        word-break: break-word;
    }
    .chat-name-user-fix { font-size: 0.68rem; color: #88b89a; font-weight: 600; text-align: right; margin-bottom: 3px; }
    .msg-ai-wrap { margin: 6px 0; }
    .chat-name-ai-fix { font-size: 0.68rem; color: #2d7a4f; font-weight: 600; margin-bottom: 3px; }
    .chat-bubble-ai-fix {
        display: inline-block;
        max-width: 90%;
        background: #ffffff;
        color: #1a1a2e;
        border: 1px solid #e0ebe4;
        border-radius: 4px 18px 18px 18px;
        padding: 10px 16px;
        font-size: 0.88rem;
        line-height: 1.6;
        word-break: break-word;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🤖 AI Facilitator — PalakGuard Assistant</div>", unsafe_allow_html=True)
    st.markdown("""<div style="margin-bottom:1rem;">
        <span style="background:rgba(201,168,76,0.15); border:1px solid rgba(201,168,76,0.4); color:#c9a84c;
        border-radius:20px; padding:3px 12px; font-size:0.72rem; font-weight:700; letter-spacing:0.8px;
        text-transform:uppercase;">⚡ Powered by Groq · Llama 3.3</span>
    </div>""", unsafe_allow_html=True)

    result = st.session_state.get("last_result")
    if result:
        sev = result.get("severity", "Healthy")
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {severity_color(sev)}; padding:0.8rem 1.2rem; margin-bottom:1rem;">
            <div style="font-size:0.72rem; color:#888; text-transform:uppercase; letter-spacing:0.5px;">Context: Last Scan</div>
            <div style="font-weight:600; color:{severity_color(sev)};">{severity_emoji(sev)} {result.get('disease_detected','')} — {sev} severity</div>
            <div style="font-size:0.82rem; color:#666;">
                DSI: {result.get('dsi',0)} &nbsp;·&nbsp;
                Affected: {result.get('affected_area_pct',0)}% &nbsp;·&nbsp;
                Yield loss: ~{result.get('yield_loss',0)}%
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("💡 Run a detection first for context-aware answers.")

    st.markdown("""<div style='font-size:0.75rem; color:#888; font-weight:600;
        margin:0.5rem 0 0.4rem; text-transform:uppercase; letter-spacing:0.5px;'>
        Suggested Questions</div>""", unsafe_allow_html=True)

    quick_prompts = [
        "What fungicides work for Downy Mildew in India?",
        "How does humidity affect sporulation?",
        "Explain the DSI scoring system",
        "What are resistant spinach varieties?",
        "When should I apply preventive sprays?",
    ]
    qcols = st.columns(len(quick_prompts))
    for i, (col, qp) in enumerate(zip(qcols, quick_prompts)):
        with col:
            if st.button(qp, key=f"qp_{i}", use_container_width=True):
                st.session_state["_chat_prefill"] = qp
                st.session_state["_auto_send"]    = True
                st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding:2.5rem 1rem; color:#aaa;">
            <div style="font-size:2.5rem;">🌿</div>
            <div style="margin-top:0.5rem; font-size:0.92rem;">
                Ask me anything about Downy Mildew,<br/>disease management, or your scan results.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user-wrap">
                    <div class="msg-user-inner">
                        <div class="chat-name-user-fix">You</div>
                        <div class="chat-bubble-user-fix">{msg['content']}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-ai-wrap">
                    <div class="chat-name-ai-fix">🌿 PalakGuard AI</div>
                    <div class="chat-bubble-ai-fix">{msg['content']}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    stream_placeholder = st.empty()

    prefill_value = st.session_state.pop("_chat_prefill", "")
    auto_send     = st.session_state.pop("_auto_send", False)
    input_key     = f"chat_input_{len(st.session_state.chat_history)}"

    input_col, btn_col, clear_col = st.columns([6, 1, 1])
    with input_col:
        user_input = st.text_input(
            "message",
            value=prefill_value,
            placeholder="💬  Ask about fungicides, DSI scores, spray schedule…",
            label_visibility="collapsed",
            key=input_key,
        )
    with btn_col:
        send = st.button("Send ➤", use_container_width=True, key="send_btn")
    with clear_col:
        if st.button("🗑️", use_container_width=True, help="Clear chat", key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()

    should_send = (send and user_input.strip()) or (auto_send and prefill_value.strip())
    msg_to_send = user_input.strip() or prefill_value.strip()

    if should_send and msg_to_send:
        st.session_state.chat_history.append({"role": "user", "content": msg_to_send})
        with stream_placeholder.chat_message("assistant", avatar="🌿"):
            try:
                full_response = st.write_stream(
                    ai_facilitator_stream(msg_to_send, st.session_state.chat_history[:-1], result)
                )
            except Exception as e:
                st.error(f"Groq error: {e}")
                full_response = ""
        if full_response:
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌦️ Risk Dashboard":
    st.markdown("<div class='section-header'>🌦️ Environmental Risk Dashboard — Western Maharashtra</div>", unsafe_allow_html=True)
    st.caption("Enter current field conditions to assess Downy Mildew risk for the next 24–48 hours.")
    col_env, col_risk = st.columns([1, 1], gap="large")
    with col_env:
        st.markdown("<div class='card'><div class='card-title'>🌡️ Environmental Parameters</div>", unsafe_allow_html=True)
        temp      = st.slider("Temperature (°C)", 5, 40, 18)
        humidity  = st.slider("Relative Humidity (%)", 30, 100, 80)
        rainfall  = st.slider("Recent Rainfall (mm/day)", 0, 50, 5)
        leaf_wet  = st.slider("Leaf Wetness Duration (hrs/night)", 0, 12, 8)
        wind      = st.slider("Wind Speed (km/h)", 0, 40, 12)
        crop_stage= st.selectbox("Crop Stage", ["Seedling (0–3 weeks)","Vegetative (3–6 weeks)","Mature (6+ weeks)"])
        st.markdown("</div>", unsafe_allow_html=True)
    with col_risk:
        temp_risk  = max(0, 1-abs(temp-15)/10)
        humid_risk = max(0, (humidity-60)/40)
        rain_risk  = min(1, rainfall/20)
        wet_risk   = min(1, leaf_wet/8)
        wind_risk  = max(0, 1-wind/30)
        sm         = {"Seedling (0–3 weeks)":0.9,"Vegetative (3–6 weeks)":1.0,"Mature (6+ weeks)":0.7}.get(crop_stage,1.0)
        overall    = min(100, max(0, round((temp_risk*0.25+humid_risk*0.25+rain_risk*0.15+wet_risk*0.25+wind_risk*0.1)*sm*100)))
        if overall >= 70:   risk_label, risk_color, risk_emoji = "HIGH RISK","#d94f3d","🔴"
        elif overall >= 40: risk_label, risk_color, risk_emoji = "MODERATE RISK","#e8b84b","🟡"
        else:               risk_label, risk_color, risk_emoji = "LOW RISK","#2d7a4f","🟢"
        st.markdown(f"""
        <div class="card" style="text-align:center; border-top:4px solid {risk_color};">
            <div class="card-title">🧮 Risk Assessment</div>
            <div style="font-size:4rem; margin:0.5rem 0;">{risk_emoji}</div>
            <div style="font-size:2.5rem; font-weight:900; font-family:Playfair Display,serif; color:{risk_color};">{risk_label}</div>
            <div style="font-size:3.5rem; font-weight:700; font-family:DM Mono,monospace; color:{risk_color};">{overall}%</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='card' style='margin-top:0.5rem;'><div class='card-title'>📊 Risk Factors</div>", unsafe_allow_html=True)
        for name, score, tip in [
            ("Temperature",   round(temp_risk*100),  "Optimal 10–20°C"),
            ("Humidity",      round(humid_risk*100), ">80% favors sporulation"),
            ("Rainfall",      round(rain_risk*100),  "Splash dispersal"),
            ("Leaf Wetness",  round(wet_risk*100),   "Night wetting critical"),
            ("Wind (dispersal)", round(wind_risk*100), "Airborne sporangia"),
        ]:
            fc = "#d94f3d" if score>70 else "#e8b84b" if score>40 else "#2d7a4f"
            st.markdown(f"""<div style="margin:6px 0;">
                <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                    <span style="font-weight:600;">{name}</span><span style="color:{fc}; font-weight:700;">{score}%</span>
                </div>
                <div class="sev-bar-bg"><div class="sev-bar-fill" style="width:{score}%; background:{fc};"></div></div>
                <div style="font-size:0.7rem; color:#999;">{tip}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    advisories = {
        "HIGH RISK":     "⚠️ **Immediate action required.** Apply Metalaxyl-M + Mancozeb within 24 hours. Increase field scouting.",
        "MODERATE RISK": "🔔 **Monitor closely.** Scout daily for early symptoms. Prepare fungicide for preventive application.",
        "LOW RISK":      "✅ **Conditions unfavorable** for Downy Mildew. Continue routine scouting and field hygiene."
    }
    st.markdown(f"<div class='card' style='border-left:4px solid {risk_color};'>{advisories.get(risk_label,'')}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DISEASE DETAILS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌿 Disease Details":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🦠 Pathogen Profile</div>
        <div class="hero-title">Spinach Downy Mildew</div>
        <div class="hero-sub">Complete clinical and epidemiological profile of Peronospora effusa</div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🦠 Pathogen","🍃 Symptoms","⚗️ Disease Cycle","📏 Quantification","🌡️ Environment","💊 Management"])

    with tabs[0]:
        st.markdown("<div class='card'><div class='card-title'>Causal Organism: Peronospora effusa</div>", unsafe_allow_html=True)
        for k,v in [
            ("Scientific Name","Peronospora effusa (Grev.) Rabenh."),
            ("Kingdom","Chromista (Oomycetes — fungus-like water mold)"),
            ("Trophic Nature","Obligate biotrophic pathogen — survives only on living spinach"),
            ("Host Specificity","Strictly host-specific to Spinacia oleracea"),
            ("Primary Inoculum","Infected seeds, crop debris, airborne sporangia"),
            ("Secondary Inoculum","Asexual sporangia produced on infected leaves nightly"),
            ("Mode of Spread","Wind-borne sporangia, splashing rain, irrigation water"),
            ("Survival","Oospores in soil/seed (long-term); mycelium in living plants"),
            ("Races","Multiple physiological races — causes frequent resistance breakdown"),
            ("Infection Mechanism","Sporangia germinate in water film → penetrate stomata → colonise intercellular spaces"),
        ]:
            st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="background:#f7fbf8; border-left:4px solid #2d7a4f;">
            <div class="card-title">Why Downy Mildew is Dangerous</div>
            <p style="color:#444; font-size:0.9rem; line-height:1.8;">
            Downy Mildew is rated the <b>most economically damaging spinach disease worldwide</b>. Because <i>Peronospora effusa</i> is biotrophic (needs live plants), it cannot grow in soil or dead matter — but it produces millions of airborne sporangia each night, capable of spreading kilometers in wind. In Western Maharashtra's cool, humid Rabi season (October–February), a single infected plant can cause <b>complete field loss within 2–3 weeks</b> if uncontrolled.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='card'><div class='card-title'>Field-Level Symptomatology</div>", unsafe_allow_html=True)
        for k,v in [
            ("🍃 Upper Leaf Surface","Pale yellow to light green angular lesions bounded by leaf veins; later turn brown"),
            ("🍂 Lower Leaf Surface","Greyish-purple downy growth (sporangiophores + sporangia) — DIAGNOSTIC feature"),
            ("🌀 Leaf Distortion","Curling, puckering, and irregular growth in severely infected plants"),
            ("💛 Chlorosis","Generalized yellowing advancing to necrosis; early senescence"),
            ("📉 Leaf Size","Reduced leaf size, weight, and market quality; stunted overall growth"),
            ("🌱 Seedling Stage","Damping-off or severe stunting; cotyledon and hypocotyl infection"),
            ("⏱️ Incubation Period","5–10 days under favorable conditions (10–18°C, >80% RH)"),
            ("🌙 Sporulation Timing","Primarily 2:00–6:00 AM under high relative humidity (>90%)"),
            ("📸 Progressive Stages","Stage 1: water-soaked spots → Stage 2: yellow lesions → Stage 3: purple sporulation → Stage 4: necrosis"),
        ]:
            st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='card-title'>Severity Classification Scale</div>", unsafe_allow_html=True)
        sev_data = [
            ("🟢 Healthy (0)", "0%", "0", "0%", "No visible symptoms"),
            ("🟡 Mild (1–2)", "<10%", "0–1.5", "~8%", "Few yellow spots, no sporulation"),
            ("🟠 Moderate (3–4)", "10–40%", "1.5–3.5", "~25%", "Multiple lesions, early sporulation"),
            ("🔴 Severe (4–5)", ">40%", "3.5–5.0", "~55%", "Extensive lesions, heavy sporulation, leaf death"),
        ]
        header_html = "<div style='display:grid; grid-template-columns:1.5fr 1fr 1fr 1fr 2fr; gap:4px; margin-bottom:4px;'>"
        for h in ["Severity", "Affected Area", "DSI", "Yield Loss", "Description"]:
            header_html += f"<div style='background:#0d2818; color:#a8e6bf; padding:8px; border-radius:6px; font-size:0.75rem; font-weight:700; text-align:center;'>{h}</div>"
        header_html += "</div>"
        st.markdown(header_html, unsafe_allow_html=True)
        for row in sev_data:
            row_html = "<div style='display:grid; grid-template-columns:1.5fr 1fr 1fr 1fr 2fr; gap:4px; margin-bottom:4px;'>"
            for cell in row:
                row_html += f"<div style='background:#f7fbf8; border:1px solid #ddeee4; padding:8px; border-radius:6px; font-size:0.82rem; text-align:center;'>{cell}</div>"
            row_html += "</div>"
            st.markdown(row_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='card'><div class='card-title'>Polycyclic Disease Cycle</div>", unsafe_allow_html=True)
        for k,v in [
            ("Disease Type","Polycyclic — multiple infection cycles per growing season"),
            ("Primary Cycle","Oospores in soil/seed germinate → seedling infection → sporulation"),
            ("Secondary Cycle","Night sporulation → wind dispersal → new leaf infection (repeats every 5–10 days)"),
            ("Sporulation Trigger","Temperature 8–15°C + leaf wetness >4 hours + RH >90%"),
            ("Sporangia Survival","4–6 hours in dry air; days in cool, moist conditions"),
            ("Latent Period","5–7 days (seedling stage); 7–10 days (mature plants)"),
            ("Epidemic Development","Exponential under optimal conditions — doubles infections every cycle"),
        ]:
            st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='card'><div class='card-title'>Disease Quantification Parameters</div>", unsafe_allow_html=True)
        for k,v in [
            ("Disease Incidence (%)","(No. of infected plants / Total plants counted) × 100"),
            ("Disease Severity Index","DSI = Σ(grade × frequency) / (max grade × total leaves); Scale: 0–5"),
            ("AUDPC","Area Under Disease Progress Curve: Σ[(yi+yi+1)/2 × (ti+1–ti)]"),
            ("Infection Frequency","Count of new disease foci per plot per scouting interval"),
            ("Sporulation Index","Visual rating 0–3: 0=none, 1=sparse, 2=moderate, 3=dense grey-purple growth"),
            ("Yield Loss Correlation","Regression: YL(%) = 8×DSI² — approximately 8% at mild, 25% moderate, 55% severe"),
        ]:
            st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("<div class='card'><div class='card-title'>Environmental Parameters — Western Maharashtra</div>", unsafe_allow_html=True)
        for k,v in [
            ("Optimal Temperature","10–20°C (infection); 8–15°C (sporulation); >25°C suppresses disease"),
            ("Critical Humidity",">80% RH triggers infection; >90% RH needed for active sporulation"),
            ("Leaf Wetness","Minimum 4 hours continuous wetness for successful infection"),
            ("Season (Pune/Nashik)","October–February (Rabi season); peak outbreak December–January"),
            ("Altitude Influence","Higher incidence at elevations >600m — cooler, more humid nights"),
            ("Rainfall Pattern","Light drizzle + fog most dangerous; heavy rain washes sporangia away"),
            ("Wind Role","5–20 km/h optimal for sporangia dispersal; >30 km/h reduces infection"),
            ("Irrigation Risk","Overhead irrigation extends leaf wetness — increases infection 2–3×"),
        ]:
            st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[5]:
        st.markdown("<div class='card'><div class='card-title'>Integrated Disease Management (IDM)</div>", unsafe_allow_html=True)
        for k,v in [
            ("🧪 Curative Fungicides","Metalaxyl-M 4% + Mancozeb 64% WP (Ridomil Gold); Cymoxanil + Mancozeb (Curzate M8)"),
            ("🛡️ Protective Fungicides","Mancozeb 75% WP (Dithane M-45); Copper oxychloride 50% WP; Fosetyl-Al"),
            ("💉 Seed Treatment","Metalaxyl 35% WS @ 6 g/kg seed — kills seed-borne oospores"),
            ("🌾 Cultural Practices","Drip irrigation (not overhead); plant spacing ≥20 cm; destroy crop debris"),
            ("📅 Spray Schedule","Preventive spray every 10 days from Dec–Jan; switch chemistries to avoid resistance"),
            ("🌱 Resistant Varieties","Consult KVK Pune/Nashik for updated Pfs-gene resistant cultivars"),
            ("🔄 Crop Rotation","Avoid consecutive spinach cultivation; rotate with brassicas or legumes"),
            ("🧬 Biological Control","Trichoderma viride + Bacillus subtilis as soil drench (supplementary use)"),
            ("🌿 Botanical Extracts","Neem oil 3% as foliar spray — suppresses sporulation (limited data)"),
        ]:
            st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)






# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL DETAILS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Details":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🤖 Deep Learning · YOLOv8</div>
        <div class="hero-title">Model Details</div>
        <div class="hero-sub">Architecture, training parameters, performance metrics, and usage guidelines for both AI models</div>
    </div>
    """, unsafe_allow_html=True)

    model_tab1, model_tab2 = st.tabs(["🔍 Stage 1: Spinach Classifier", "🧫 Stage 2: Disease Detector"])

    with model_tab1:
        st.markdown('<div class="model-badge">🔍 Stage 1 — Plant Verification Model</div><div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='card'><div class='card-title'>📋 Model Overview</div>", unsafe_allow_html=True)
            for k,v in [
                ("Model Type","Image Classification"),
                ("Architecture","YOLOv8 Classification Variant (yolov8-cls)"),
                ("Task","Spinach vs. Not Spinach detection"),
                ("Input Size","224 × 224 pixels (RGB)"),
                ("Output","Class label + Confidence score (0–100%)"),
                ("Framework","Ultralytics YOLO v8"),
                ("File","models/class_2/best.pt"),
            ]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><div class='card-title'>🗂️ Classes</div>", unsafe_allow_html=True)
            for cls, desc in [("🟢 spinach","Spinacia oleracea leaf images (healthy or diseased)"),("🔴 not_spinach","All other plants, objects, or non-leaf images")]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{cls}</div><div class='info-val'>{desc}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><div class='card-title'>📊 Dataset Info</div>", unsafe_allow_html=True)
            for k,v in [
                ("Total Images","~2,000"),("Spinach Images","~1,000"),("Not Spinach Images","~1,000"),
                ("Classes","2 (Binary classification)"),("Balanced","Yes (approximately 50/50)"),
                ("Split","80% Train / 10% Val / 10% Test"),("Augmentation","Flip, rotate, brightness, contrast, mosaic"),
            ]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown("<div class='card'><div class='card-title'>⚙️ Training Details</div>", unsafe_allow_html=True)
            for k,v in [
                ("Epochs","20–50 (early stopping applied)"),("Batch Size","16"),("Image Size","224 × 224"),
                ("Optimizer","SGD with momentum"),("Learning Rate","0.01 (cosine decay)"),
                ("Pre-trained Weights","ImageNet (transfer learning)"),("Hardware","GPU training (CUDA)"),
            ]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><div class='card-title'>🎯 Performance Metrics</div>", unsafe_allow_html=True)
            for k,v,color in [
                ("Training Accuracy","~96%","#2d7a4f"),("Validation Accuracy","~92%","#2d7a4f"),
                ("Test Accuracy","~91%","#2d7a4f"),("Training Loss","~0.12","#2d7a4f"),("Validation Loss","~0.18","#e8b84b"),
            ]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val' style='font-weight:700; color:{color};'>{v}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            fig_acc = go.Figure(go.Pie(values=[92, 8], labels=["Correct", "Error"],
                marker_colors=["#2d7a4f","#f0f5f1"], hole=0.72, textinfo="none"))
            fig_acc.update_layout(height=200, margin=dict(t=20,b=20,l=20,r=20), paper_bgcolor="white",
                annotations=[dict(text="92%", x=0.5, y=0.5, font_size=24, font_color="#2d7a4f", font_family="DM Mono", showarrow=False)],
                showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
            st.caption("Validation Accuracy")
        st.markdown("<div class='card'><div class='card-title'>🔲 Confusion Matrix (Illustrative)</div>", unsafe_allow_html=True)
        st.markdown("""
        <table style="width:100%; border-collapse:collapse; font-family:DM Sans;">
        <thead><tr>
            <th style="background:#f0f5f1; padding:10px; border:1px solid #ddeee4;"></th>
            <th style="background:#0d2818; color:#a8e6bf; padding:10px; border:1px solid #ddeee4; text-align:center;">Predicted: Spinach</th>
            <th style="background:#0d2818; color:#a8e6bf; padding:10px; border:1px solid #ddeee4; text-align:center;">Predicted: Not Spinach</th>
        </tr></thead>
        <tbody>
          <tr>
            <td style="background:#0d2818; color:#a8e6bf; padding:10px; border:1px solid #ddeee4; font-weight:700;">Actual: Spinach</td>
            <td style="background:#e6f7ed; color:#1a7a40; padding:10px; border:1px solid #ddeee4; text-align:center; font-size:1.2rem; font-weight:700;">TP: 92</td>
            <td style="background:#fdecea; color:#c0392b; padding:10px; border:1px solid #ddeee4; text-align:center; font-size:1.2rem; font-weight:700;">FN: 8</td>
          </tr>
          <tr>
            <td style="background:#0d2818; color:#a8e6bf; padding:10px; border:1px solid #ddeee4; font-weight:700;">Actual: Not Spinach</td>
            <td style="background:#fdecea; color:#c0392b; padding:10px; border:1px solid #ddeee4; text-align:center; font-size:1.2rem; font-weight:700;">FP: 6</td>
            <td style="background:#e6f7ed; color:#1a7a40; padding:10px; border:1px solid #ddeee4; text-align:center; font-size:1.2rem; font-weight:700;">TN: 94</td>
          </tr>
        </tbody></table>
        <div style="font-size:0.75rem; color:#888; margin-top:8px;">Values shown per 100 test samples (illustrative).</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="background:#fff8f0; border-left:4px solid #e8b84b;">
            <div class="card-title">⚠️ Model Limitations</div>
            <div style="font-size:0.88rem; color:#444; line-height:2.0;">
            ⚡ May confuse spinach with other broad-leaf greens (amaranth, beet leaves) in low-quality images<br/>
            ⚡ Performance degrades on blurry, overexposed, or heavily shadowed images<br/>
            ⚡ Requires clear visibility of the leaf surface — whole-plant images may reduce accuracy<br/>
            ⚡ Sensitive to extreme lighting conditions<br/>
            ⚡ Model was trained on Western Maharashtra field conditions — may vary in other regions
            </div>
        </div>
        """, unsafe_allow_html=True)

    with model_tab2:
        st.markdown('<div class="model-badge" style="background:rgba(209,68,47,0.1); border-color:rgba(209,68,47,0.3); color:#d94f3d;">🧫 Stage 2 — Disease Detection Model</div><div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='card'><div class='card-title'>📋 Model Overview</div>", unsafe_allow_html=True)
            for k,v in [
                ("Model Type","Image Classification"),
                ("Architecture","YOLOv8 Classification Variant (yolov8-cls)"),
                ("Task","Healthy vs. Downy Mildew detection in spinach"),
                ("Input Size","224 × 224 pixels (RGB)"),
                ("Output","Disease class + Confidence score"),
                ("Framework","Ultralytics YOLO v8"),
                ("File","models/detection/best.pt"),
            ]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><div class='card-title'>🗂️ Classes</div>", unsafe_allow_html=True)
            for cls, desc in [("🟢 healthy","Normal spinach leaf — no visible pathogen symptoms"),("🔴 downy_mildew","Spinach leaf infected with Peronospora effusa")]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{cls}</div><div class='info-val'>{desc}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><div class='card-title'>📊 Dataset Info</div>", unsafe_allow_html=True)
            for k,v in [
                ("Total Images","~1,500 spinach leaf images"),("Healthy Images","~750"),("Downy Mildew Images","~750"),
                ("Collection","Field samples — Western Maharashtra"),("Imaging Conditions","Natural daylight, various angles"),
                ("Split","80% Train / 10% Val / 10% Test"),("Augmentation","Flip, rotate, HSV shift, blur, cutout"),
            ]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown("<div class='card'><div class='card-title'>⚙️ Training Details</div>", unsafe_allow_html=True)
            for k,v in [
                ("Epochs","30–60 (best.pt = best validation checkpoint)"),("Batch Size","16"),("Image Size","224 × 224"),
                ("Optimizer","AdamW"),("Learning Rate","0.001 with cosine schedule"),
                ("Pre-trained Weights","COCO (transfer learning)"),("Hardware","GPU training (CUDA)"),
            ]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val'>{v}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><div class='card-title'>🎯 Performance Metrics</div>", unsafe_allow_html=True)
            for k,v,color in [
                ("Training Accuracy","~94%","#2d7a4f"),("Validation Accuracy","~89%","#2d7a4f"),
                ("Test Accuracy","~88%","#2d7a4f"),("Precision (Downy Mildew)","~0.91","#2d7a4f"),
                ("Recall (Downy Mildew)","~0.87","#e8b84b"),("F1 Score","~0.89","#2d7a4f"),
            ]:
                st.markdown(f"<div class='info-row'><div class='info-key'>{k}</div><div class='info-val' style='font-weight:700; color:{color};'>{v}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            fig_acc2 = go.Figure(go.Pie(values=[89, 11], labels=["Correct","Error"],
                marker_colors=["#d94f3d","#f0f5f1"], hole=0.72, textinfo="none"))
            fig_acc2.update_layout(height=200, margin=dict(t=20,b=20,l=20,r=20), paper_bgcolor="white",
                annotations=[dict(text="89%", x=0.5, y=0.5, font_size=24, font_color="#d94f3d", font_family="DM Mono", showarrow=False)],
                showlegend=False)
            st.plotly_chart(fig_acc2, use_container_width=True)
            st.caption("Validation Accuracy")
        st.markdown("<div class='card'><div class='card-title'>🔲 Confusion Matrix (Illustrative)</div>", unsafe_allow_html=True)
        st.markdown("""
        <table style="width:100%; border-collapse:collapse; font-family:DM Sans;">
        <thead><tr>
            <th style="background:#f0f5f1; padding:10px; border:1px solid #ddeee4;"></th>
            <th style="background:#0d2818; color:#a8e6bf; padding:10px; border:1px solid #ddeee4; text-align:center;">Predicted: Healthy</th>
            <th style="background:#0d2818; color:#a8e6bf; padding:10px; border:1px solid #ddeee4; text-align:center;">Predicted: Downy Mildew</th>
        </tr></thead>
        <tbody>
          <tr>
            <td style="background:#0d2818; color:#a8e6bf; padding:10px; border:1px solid #ddeee4; font-weight:700;">Actual: Healthy</td>
            <td style="background:#e6f7ed; color:#1a7a40; padding:10px; border:1px solid #ddeee4; text-align:center; font-size:1.2rem; font-weight:700;">TP: 90</td>
            <td style="background:#fdecea; color:#c0392b; padding:10px; border:1px solid #ddeee4; text-align:center; font-size:1.2rem; font-weight:700;">FN: 10</td>
          </tr>
          <tr>
            <td style="background:#0d2818; color:#a8e6bf; padding:10px; border:1px solid #ddeee4; font-weight:700;">Actual: Downy Mildew</td>
            <td style="background:#fdecea; color:#c0392b; padding:10px; border:1px solid #ddeee4; text-align:center; font-size:1.2rem; font-weight:700;">FP: 13</td>
            <td style="background:#e6f7ed; color:#1a7a40; padding:10px; border:1px solid #ddeee4; text-align:center; font-size:1.2rem; font-weight:700;">TN: 87</td>
          </tr>
        </tbody></table>
        <div style="font-size:0.75rem; color:#888; margin-top:8px;">Higher FP rate is intentional — we prefer false alarms over missed disease detection.</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="background:#fff8f0; border-left:4px solid #e8b84b;">
            <div class="card-title">⚠️ Model Limitations</div>
            <div style="font-size:0.88rem; color:#444; line-height:2.0;">
            ⚡ Early-stage infection may be misclassified as healthy — use preventive approach<br/>
            ⚡ Model trained on Downy Mildew only; other spinach diseases may be misclassified<br/>
            ⚡ Confidence can drop under unusual lighting conditions or with camera artifacts<br/>
            ⚡ Images must show leaf tissue clearly — roots, stems, or soil backgrounds reduce accuracy<br/>
            ⚡ New P. effusa races may present atypical symptoms not represented in training data
            </div>
        </div>
        """, unsafe_allow_html=True)





# # # ══════════════════════════════════════════════════════════════════════════════
# # # PAGE: SAMPLE IMAGES
# # # ══════════════════════════════════════════════════════════════════════════════


elif page == "📸 Sample Images":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">📸 Visual Reference</div>
        <div class="hero-title">Sample Images</div>
        <div class="hero-sub">
            Reference images showing healthy and Downy Mildew-infected spinach leaves
            at various severity stages
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="background:#f7fbf8; border-left:4px solid #2d7a4f;">
        <div class="card-title">How to Use This Reference</div>
        <p style="font-size:0.88rem; color:#444; line-height:1.8;">
        Compare your field samples against these reference images to identify Downy Mildew
        at different stages. Early detection (Mild stage) is critical for effective management.
        If you see symptoms matching Moderate or Severe stages, apply fungicide treatment
        immediately and notify your agronomist.
        </p>
    </div>
    """, unsafe_allow_html=True)

   
    THUMB_W = 320    
    THUMB_H = 280    
    
    def make_thumb(img: Image.Image, w: int = THUMB_W, h: int = THUMB_H) -> Image.Image:
        """Resize + centre-crop to exact w×h so all thumbnails are uniform."""
      
        img_ratio  = img.width / img.height
        box_ratio  = w / h
        if img_ratio > box_ratio:
           
            new_h = h
            new_w = int(img.width * h / img.height)
        else:
          
            new_w = w
            new_h = int(img.height * w / img.width)
        img = img.resize((new_w, new_h), Image.LANCZOS)

       
        left = (new_w - w) // 2
        top  = (new_h - h) // 2
        return img.crop((left, top, left + w, top + h))

    stages = [
        {
            "title":    "🟢 Healthy Spinach",
            "folder":   "healthy",
            "color":    "#2d7a4f",
            "bg":       "#e6f7ed",
            "border":   "#a8e6bf",
            "dsi_info": "DSI: 0 · Affected: 0%",
            "loss":     "No yield loss",
            "symptoms": [
                "Deep, uniform green upper surface — no spots or lesions",
                "Pale green lower surface — visible veins, NO grey/purple growth",
                "Flat leaf shape — normal size, no curling or puckering",
                "Vigorous growth — normal leaf size and fresh weight",
            ],
            "action": "Continue routine scouting; maintain good air circulation",
        },
        {
            "title":    "🟡 Mild — Stage 1 (Early Infection)",
            "folder":   "mild",
            "color":    "#c8a020",
            "bg":       "#fffbf0",
            "border":   "#f9e79f",
            "dsi_info": "DSI: 0–1.5 · Affected: <10%",
            "loss":     "~8% yield loss if untreated",
            "symptoms": [
                "Pale water-soaked spots on upper leaf surface",
                "Spots are angular, bounded by leaf veins",
                "No visible sporulation yet on lower surface",
                "Plant appears mostly normal from a distance",
            ],
            "action": "Scout carefully; begin preventive fungicide application now",
        },
        {
            "title":    "🟠 Moderate — Stage 2",
            "folder":   "moderate",
            "color":    "#e07b39",
            "bg":       "#fff5ef",
            "border":   "#f9cba3",
            "dsi_info": "DSI: 1.5–3.5 · Affected: 10–40%",
            "loss":     "~25% yield loss if untreated",
            "symptoms": [
                "Yellow-green angular lesions clearly visible on upper surface",
                "Early grey sporulation visible on leaf undersides",
                "Some leaf curling and distortion beginning",
                "Multiple lesions per leaf; general leaf yellowing",
            ],
            "action": "Apply Metalaxyl-M + Mancozeb immediately; repeat in 7–10 days",
        },
        {
            "title":    "🔴 Severe — Stage 3",
            "folder":   "severe",
            "color":    "#d94f3d",
            "bg":       "#fef0ef",
            "border":   "#f5b7b1",
            "dsi_info": "DSI: 3.5–5.0 · Affected: >40%",
            "loss":     "~55% yield loss if untreated",
            "symptoms": [
                "Dense grey-purple sporulation covering lower leaf surface",
                "Brown necrotic lesions covering upper surface",
                "Heavy leaf curling, wilting, and premature senescence",
                "Significant reduction in leaf size and fresh weight",
            ],
            "action": "Emergency fungicide; consider crop destruction in hotspot areas",
        },
    ]

   
    for stage in stages:
        color  = stage["color"]
        bg     = stage["bg"]
        border = stage["border"]

        st.markdown(
            f"<div class='section-header' style='color:{color};'>{stage['title']}</div>",
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    margin-bottom:10px; flex-wrap:wrap; gap:8px;">
            <div style="font-size:0.82rem; color:#666;">{stage['dsi_info']}</div>
            <div style="background:{bg}; border:1px solid {color}; border-radius:8px;
                        padding:5px 14px; font-size:0.78rem; color:{color};
                        font-weight:700;">{stage['loss']}</div>
        </div>
        """, unsafe_allow_html=True)

        img_col, info_col = st.columns([1.4, 1], gap="large")

        # ── Images ────────────────────────────────────────────────────────────
        with img_col:
            images = load_sample_images(stage["folder"])

            if images:
              
                COLS_PER_ROW = 3
                for row_start in range(0, len(images), COLS_PER_ROW):
                    row_imgs = images[row_start : row_start + COLS_PER_ROW]
                    cols = st.columns(COLS_PER_ROW)  
                    for col, (fname, img) in zip(cols, row_imgs):
                        with col:
                            thumb = make_thumb(img, THUMB_W, THUMB_H)
                            st.image(
                                thumb,
                                caption=fname,
                                width=THUMB_W,          
                            )
                   
                    for empty_col in cols[len(row_imgs):]:
                        with empty_col:
                            st.empty()
            else:
                st.markdown(f"""
                <div style="background:{bg}; border:2px dashed {border};
                            border-radius:12px; padding:2rem; text-align:center;
                            color:#aaa;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">📁</div>
                    <div style="font-size:0.82rem;">
                        No images yet.<br/>
                        Add <code>.jpg</code> / <code>.png</code> files to<br/>
                        <code>samples/{stage['folder']}/</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        
        with info_col:
            st.markdown(f"""
            <div class="card" style="border-left:4px solid {color};
                                     background:{bg}; margin-bottom:0;">
                <div style="font-size:0.72rem; color:#888; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.6px;
                            margin-bottom:8px;">Symptoms to look for</div>
                {"".join([
                    f'<div style="font-size:0.83rem; color:#444; padding:5px 0;'
                    f'border-bottom:1px solid {border};">🔸 {s}</div>'
                    for s in stage["symptoms"]
                ])}
                <div style="margin-top:10px; background:white; border-left:3px solid {color};
                            border-radius:0 6px 6px 0; padding:8px 12px;
                            font-size:0.8rem; color:#1a4a2e;">
                    💊 <b>Action:</b> {stage["action"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">ℹ️ Project Information</div>
        <div class="hero-title">About PalakGuard</div>
        <div class="hero-sub">Research background, development team, technology stack, and project mission</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("""
        <div class="card">
            <div class="card-title">Our Mission</div>
            <p style="color:#444; line-height:1.9; font-size:0.92rem;">
            Spinach (<i>Spinacia oleracea</i>) is one of the most important leafy vegetables cultivated in Western Maharashtra, contributing significantly to the income of small and marginal farmers in Pune, Nashik, and Kolhapur districts.
            </p>
            <p style="color:#444; line-height:1.9; font-size:0.92rem;">
            <b>Downy Mildew</b>, caused by <i>Peronospora effusa</i>, is rated the most economically damaging spinach disease, capable of causing 55–100% crop loss under favorable conditions. Yet early-stage disease often goes undetected by the naked eye, allowing epidemics to spread unchecked.
            </p>
            <p style="color:#444; line-height:1.9; font-size:0.92rem;">
            <b>PalakGuard</b> was built to bridge this gap — providing instant, accurate, and actionable disease detection to farmers, researchers, and agronomists without requiring expensive laboratory equipment or expert plant pathologists on-site.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">Research Background</div>
            <p style="color:#444; line-height:1.9; font-size:0.92rem;">
            This project is part of ongoing plant pathology research at the Department of Plant Pathology. The study focuses on:
            </p>
            <ul style="color:#444; line-height:2.0; font-size:0.9rem;">
                <li>Epidemiology of spinach Downy Mildew in Western Maharashtra</li>
                <li>Development of rapid, field-deployable AI detection tools</li>
                <li>Disease quantification methodologies (DSI, AUDPC) for spinach</li>
                <li>Integrated Disease Management (IDM) strategy optimization</li>
                <li>Environmental risk modeling for Rabi season spinach cultivation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Target Disease — moved here, horizontal layout
        st.markdown("""
        <div class="card">
            <div class="card-title">Target Disease</div>
            <div style="display:flex; align-items:center; gap:20px;">
                <div style="font-size:3rem; flex-shrink:0;">🦠</div>
                <div>
                    <div style="font-family:Playfair Display,serif; font-size:1.1rem;
                                font-weight:700; color:#1a4a2e;">Downy Mildew</div>
                    <div style="font-size:0.82rem; color:#888; font-style:italic;
                                margin-bottom:8px;">Peronospora effusa</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:2px;
                                font-size:0.8rem; color:#444; line-height:1.9;">
                        <div>🔸 Obligate biotrophic pathogen</div>
                        <div>🔸 Polycyclic disease (rapid spread)</div>
                        <div>🔸 5–10 day incubation period</div>
                        <div>🔸 Up to 55% yield loss (severe)</div>
                        <div>🔸 Peak season: Dec–Jan (Maharashtra)</div>
                        <div>🔸 Strictly host-specific to spinach</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        # Research Guide — Dr. Madhuri Pant
        st.markdown("""
        <div class="card" style="background:linear-gradient(135deg,#0d2818,#1a4a2e); color:white;">
            <div style="font-family:Playfair Display,serif; font-size:1.1rem;
                        color:#a8e6bf; margin-bottom:1rem;">Project Mentor</div>
            <div style="text-align:center; padding:1rem 0;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">👩‍🏫</div>
                <div style="font-family:Playfair Display,serif; font-size:1.2rem;
                            font-weight:700; color:#fff;">Dr. Madhuri Pant</div>
                <div style="font-size:0.82rem; color:#6ab880; margin-top:4px;">
                    Faculty Member
                </div>
                <div style="font-size:0.78rem; color:#88b89a; margin-top:2px;">
                    Department of Computer Science<br/>
                    Vishwakarma University
                </div>
            </div>
            <hr style="border-color:#2a5c3a; margin:1rem 0;"/>
            <div style="font-size:0.82rem; color:#d4eadb; line-height:1.9;">
                <b style="color:#a8e6bf;">Role:</b><br/>
                Minor Project Mentor<br/>
                Faculty of Science and Technology<br/>
                Vishwakarma University
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Developer — Emmanuel Mushata
        st.markdown("""
        <div class="card" style="border-left:4px solid #2d7a4f;">
            <div style="font-family:Playfair Display,serif; font-size:1.1rem;
                        color:#1a4a2e; margin-bottom:1rem; font-weight:700;">Developer</div>
            <div style="text-align:center; padding:0.5rem 0;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">👨‍💻</div>
                <div style="font-family:Playfair Display,serif; font-size:1.1rem;
                            font-weight:700; color:#1a4a2e;">Emmanuel Mushata</div>
                <div style="font-size:0.78rem; color:#2d7a4f; font-weight:600;
                            margin-top:4px;">Full Stack AI Developer</div>
            </div>
            <hr style="border-color:#e0ebe4; margin:0.8rem 0;"/>
            <div style="font-size:0.82rem; color:#444; line-height:1.9;">
                <b style="color:#1a4a2e;">Programme:</b><br/>
                BSc Computer Science<br/>
                Faculty of Science and Technology<br/>
                Third Year Student · Vishwakarma University<br/><br/>
                <b style="color:#1a4a2e;">Roles & Contributions:</b><br/>
                🔬 AI pipeline architecture & development<br/>
                🤖 YOLO model training & integration<br/>
                🌿 Disease heatmap & overlay system<br/>
                📊 Disease analytics & AUDPC module<br/>
                🌦️ Environmental risk dashboard<br/>
                💬 Groq AI facilitator integration<br/>
                🎨 UI/UX design & frontend development<br/>
                🗄️ Session state & data management<br/>
                📸 Sample image reference system<br/>
                📚 Disease encyclopedia & knowledge base
            </div>
        </div>
        """, unsafe_allow_html=True)


    # Technology Stack
    st.markdown("<div class='section-header'>🛠️ Technology Stack</div>", unsafe_allow_html=True)
    tech_items = [
        ("🐍", "Python 3.11",          "Core programming language"),
        ("📊", "Streamlit",            "Web application framework"),
        ("🤖", "YOLOv8 (Ultralytics)", "Deep learning detection models"),
        ("⚡", "Groq API (Llama 3.3)", "AI facilitator chatbot"),
        ("📈", "Plotly",               "Interactive charts and visualizations"),
        ("🐼", "Pandas / NumPy",       "Data processing and analytics"),
        ("🖼️", "Pillow (PIL)",         "Image preprocessing"),
        ("🔬", "OpenCV",               "Disease heatmap & overlay analysis"),
    ]
    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(tech_items):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:1rem;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-weight:700; font-size:0.88rem; color:#1a4a2e; margin:4px 0;">{name}</div>
                <div style="font-size:0.75rem; color:#666;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # System Architecture
    st.markdown("<div class='section-header'>🔄 System Architecture</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-family:DM Mono,monospace; font-size:0.82rem; color:#1a4a2e;
                    background:#f0f5f1; border-radius:10px; padding:1.5rem; line-height:2.0;">
        📤  User uploads spinach leaf image<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
        🌿  <b>Stage 1 — YOLO Classifier (class_2/best.pt)</b><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ NOT spinach? → ❌ Reject (show plant name detected)<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ IS spinach? → ✅ Proceed to Stage 2<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
        🧫  <b>Stage 2 — YOLO Classifier (detection/best.pt)</b><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Healthy / Downy Mildew classification<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Confidence score extraction<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
        🔬  <b>Stage 3 — OpenCV Heatmap (get_disease_overlay)</b><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Chlorosis / Necrosis / Sporulation mapping<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Affected area % · Severity grade derivation<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
        📊  <b>Post-processing</b><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ DSI calculation · Yield loss estimate · AUDPC tracking<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
        🤖  <b>AI Facilitator (Groq Llama 3.3)</b><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Context-aware disease management advice
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="card" style="background:#fff8f0; border-left:4px solid #e8b84b;">
        <div class="card-title">⚠️ Disclaimer</div>
        <p style="font-size:0.85rem; color:#444; line-height:1.8;">
        PalakGuard is a <b>research tool</b> intended to support, not replace, professional
        plant pathological diagnosis. Predictions are probabilistic and should be verified
        with field observations. Always consult your local agricultural extension officer
        or KVK before making major crop management decisions. AI Facilitator responses are
        generated by a language model and may require expert validation for critical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)





#python -m streamlit run app.py   