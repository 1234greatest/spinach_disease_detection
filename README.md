# 🌿 PalakGuard — Spinach Downy Mildew Detection System

AI-powered Streamlit web application for detecting Downy Mildew (*Peronospora effusa*) in spinach (*Spinacia oleracea*), developed for plant pathology research in Western Maharashtra.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Open in Browser
App opens automatically at: `http://localhost:8501`

### 4. Enter API Key
- Go to [console.anthropic.com](https://console.anthropic.com) to get your API key
- Paste it in the sidebar of the running app

---

## 📱 Features

### 🏠 Home & Detection (Two-Stage Pipeline)
- **Stage 1** — Plant Verification: Confirms the image is spinach, not another plant
- **Stage 2** — Disease Detection: Detects Downy Mildew with severity classification
- Outputs: Confidence %, DSI score, Affected area %, Yield loss estimate, Fungicide recommendation

### 📊 Disease Analytics
- Disease Incidence %, Average DSI, AUDPC curve
- Severity distribution pie chart
- Scan history table with color-coded metrics
- Manual AUDPC calculator

### 🤖 AI Facilitator
- Context-aware chatbot powered by Claude
- Knows your last scan result
- Quick-prompt buttons for common questions
- Covers: fungicide schedules, environmental risk, IDM strategies, research queries

### 🌦️ Environmental Risk Dashboard
- Input: Temperature, Humidity, Rainfall, Leaf wetness, Wind speed, Crop stage
- Outputs: Risk score (0–100%), Risk level (Low/Moderate/High)
- Factor breakdown with individual risk bars
- Field advisory text

### 📚 Disease Encyclopedia
- Pathogen biology (*Peronospora effusa*)
- Field symptomatology
- Environmental parameters for Western Maharashtra
- Disease quantification methods (DSI, AUDPC, Incidence)
- Integrated Disease Management (IDM) table

---

## 🧬 Technical Details

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| AI Vision | Claude claude-sonnet-4-20250514 (Anthropic) |
| AI Chatbot | Claude claude-sonnet-4-20250514 (Anthropic) |
| Charts | Plotly |
| Data | Pandas |
| Image Processing | Pillow |

### Detection Pipeline
```
Upload Image
    │
    ▼
Stage 1: Claude Vision → "Is this spinach?"
    │ NO → Reject with plant name identified
    │ YES ↓
Stage 2: Claude Vision → Downy Mildew Analysis
    │
    ▼
Report: Severity + DSI + Affected% + Recommendations
```

---

## 📋 Disease Parameters Covered

Based on research parameters by Dr. Madhuri Pant, Plant Pathology Lab:

1. **Pathogen**: *Peronospora effusa* — obligate biotrophic oomycete
2. **Host**: *Spinacia oleracea* (Spinach)  
3. **Symptoms**: Yellow lesions (upper), Purple downy growth (lower), chlorosis
4. **Quantification**: Disease Incidence, DSI, AUDPC, Infection frequency, Yield loss
5. **Environment**: Western Maharashtra conditions (Oct–Feb, cool+humid)

---

## 🔑 API Key
Get your free Anthropic API key: https://console.anthropic.com

---

*Developed for spinach disease research — Western Maharashtra, India*
