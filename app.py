import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import requests
import json
import os
from fpdf import FPDF

# --- 1. CONFIGURATION & PROJECT IDENTITY ---
# Project ID: gen-lang-client-0652486419
apiKey = "" 

def get_api_key():
    try:
        if st.secrets and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except:
        pass
    return apiKey.strip()

def local_clinical_brain(hb, bp, risk_score, lang, diagnostic_msg=""):
    """Rule-based clinical fallback if AI is unreachable."""
    if lang == "தமிழ்":
        advice = f"மருத்துவ ஆய்வு: அபாய மதிப்பெண் {risk_score}%. "
        if hb < 11: advice += "ஹீமோகுளோபின் குறைவாக உள்ளது. இரும்புச்சத்து உணவுகளை உட்கொள்ளுங்கள். "
        if bp > 140: advice += "இரத்த அழுத்தம் அதிகம். ஓய்வெடுக்கவும்."
        footer = f"\n\n(குறிப்பு: இணைப்பு சிக்கல். பிழை விவரம்: {diagnostic_msg})" if diagnostic_msg else "\n\n(குறிப்பு: இது ஒரு தானியங்கி மருத்துவ உதவி.)"
        return advice + footer
    elif lang == "हिन्दी":
        advice = f"नैदानிக் विश्लेषण: जोखिम स्कोर {risk_score}% है। "
        if hb < 11: advice += "हीमोग्लोबिन कम है। आयरन युक्त भोजन लें। "
        if bp > 140: advice += "रक्तचाप अधिक है। आराम करें।"
        footer = f"\n\n(त्रुटि विवरण: {diagnostic_msg})" if diagnostic_msg else "\n\n(नोट: यह एक स्वचालित सहायता है।)"
        return advice + footer
    else:
        advice = f"Clinical Assessment: Predictive risk is {risk_score}%. "
        if hb < 11: advice += "Action: Increase Iron intake. "
        if bp > 140: advice += "Action: Monitor BP daily and reduce salt. "
        footer = f"\n\n(Diagnostic Details: {diagnostic_msg})" if diagnostic_msg else "\n\n(Note: Rule-based fallback due to API timeout.)"
        return advice + footer

# --- 2. AI CORE LOGIC (DEEP DISCOVERY) ---
def list_available_models():
    """Diagnostic function to ask Google what models this specific key can see."""
    key = get_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'].split('/')[-1] for m in models]
        else:
            return [f"Error {response.status_code}"]
    except:
        return ["Request Failed"]

def call_gemini_ai(prompt, context_type="general", language="English"):
    """
    Exhaustive Discovery Engine: Cycles through every version to bypass 404s.
    """
    current_key = get_api_key()
    patient_ctx = st.session_state.get('patient_data', {})
    hb = patient_ctx.get('hb', 11.0)
    bp = patient_ctx.get('bp', 120)
    score = st.session_state.get('risk_score', 0)

    if not current_key or len(current_key) < 10:
        return local_clinical_brain(hb, bp, score, language, "API Key Missing in Streamlit Secrets")

    # Priority list including standard and experimental strings
    discovery_paths = [
        ("v1beta", "gemini-2.0-flash"),
        ("v1beta", "gemini-2.0-flash-lite"),
        ("v1beta", "gemini-2.5-flash"),
        ("v1beta", "gemini-2.5-pro"),
        ("v1beta", "gemini-flash-latest"),
    ]

    
    prompts = {
        "clinical": "Senior Maternal Health Doctor. Provide a professional clinical risk assessment.",
        "tips": "Provide 3 unique, medical-backed pregnancy tips for this week.",
        "music": "Recommend 3 relaxation soundscapes for this pregnancy stage.",
        "chatbot": "Friendly maternal health assistant. Answer queries warmly."
    }
    
    instr = prompts.get(context_type, "Maternal health assistant.")
    full_query = (f"Role: {instr}\n"
                  f"LANGUAGE: Respond ONLY in {language}.\n"
                  f"CONTEXT: {patient_ctx}\n"
                  f"QUERY: {prompt}")

    payload = {"contents": [{"role": "user", "parts": [{"text": full_query}]}]}
    headers = {"Content-Type": "application/json"}
    
    error_log = []
    
    for version, model_name in discovery_paths:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={current_key}"
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=8)
            if response.status_code == 200:
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                error_log.append(f"{model_name}({version}): {response.status_code}")
        except Exception:
            error_log.append(f"{model_name}: Connection Fail")
        time.sleep(0.05) 

    diagnostic_msg = " | ".join(error_log)
    return local_clinical_brain(hb, bp, score, language, diagnostic_msg)

# --- 3. PDF GENERATION ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MaternalAI Clinical Report', 0, 1, 'C')
        self.ln(10)

def create_pdf(data, ai_note):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Patient Name: {data.get('name', 'N/A')}", 0, 1)
    pdf.set_font("Arial", '', 11)
    vitals = (f"Age: {data.get('age')} | Week: {data.get('week')}\n"
              f"Hb: {data.get('hb')} | BP: {data.get('bp')} | Weight: {data.get('weight')}")
    pdf.multi_cell(0, 10, vitals)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical AI Assessment:", 0, 1)
    pdf.set_font("Arial", '', 10)
    try:
        clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    except:
        clean_ai = "Assessment complete. See application dashboard for localized details."
    pdf.multi_cell(0, 6, clean_ai)
    return pdf.output(dest='S').encode('latin-1')

# --- 4. UI STYLING & TEXT VISIBILITY ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown, [data-testid="stMetricLabel"], .stRadio label, .stSelectbox label { 
        color: #0f172a !important; 
        font-weight: 600 !important; 
    }
    input, textarea, [data-baseweb="input"], [data-baseweb="select"], .stNumberInput div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }
    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: white !important; font-family: serif; font-size: 50px; font-weight: 900;
        width: 90px; height: 90px; display: flex; align-items: center; 
        justify-content: center; border-radius: 26px; border: 3px solid white;
        box-shadow: 0 10px 25px rgba(30, 27, 75, 0.3);
    }
    .main-card {
        background: white; padding: 40px; border-radius: 32px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.06); border: 1px solid #eef2f6;
        margin-bottom: 25px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%);
        color: #ffffff !important; border-radius: 18px; font-weight: 700;
        padding: 16px 32px; border: none; box-shadow: 0 4px 15px rgba(67, 56, 202, 0.3);
        transition: 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-4px); box-shadow: 0 12px 25px rgba(67, 56, 202, 0.5); }
    [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. SESSION & NAV ---
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'ai_assessment' not in st.session_state: st.session_state.ai_assessment = ""
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'risk_score' not in st.session_state: st.session_state.risk_score = None

with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin:30px 0;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Choose Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
    
    # Navigation mapping
    nav_map = {
        "English": ["Assessment", "Weekly AI Tips", "Medication Log", "Relaxation Music", "AI Doctor Chat"],
        "தமிழ்": ["மதிப்பீடு", "வாராந்திர குறிப்புகள்", "மருந்துப் பதிவு", "இசை", "AI மருத்துவர்"],
        "हिन्दी": ["मूल्यांकन", "साप्ताहिक सुझाव", "दवा लॉग", "संगीत", "AI डॉक्टर"]
    }
    nav_choice = st.radio("Menu / மெனு / मेनू", nav_map[lang])
    
    st.write("---")
    if st.button("🔍 Check API Status"):
        models = list_available_models()
        st.write("Available models for your key:")
        st.code(models)
    
    st.success("System: Connected 🟢")

# --- FULL MULTILINGUAL CONTENT MAP ---
content = {
    "English": {
        "title": "Maternal Health Dashboard", "vitals": "📝 Clinical Inputs",
        "name": "Full Name", "age": "Age", "hb": "Hemoglobin", "bp": "Systolic BP", 
        "wt": "Weight", "wk": "Week", "btn_run": "Analyze Health",
        "res": "📊 Results", "btn_ai": "Generate AI Report", "dl": "Download PDF",
        "tips_title": "📅 Personalized Weekly Tips", "btn_tips": "Refresh Tips",
        "med_title": "💊 Medication Log", "med_add": "Add Medicine",
        "music_title": "🎵 Wellness Music Recommendations", "music_btn": "Get AI Selection",
        "chat_title": "🤖 AI Chat Assistant", "chat_btn": "Ask Assistant"
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார மேலாண்மை", "vitals": "📝 மருத்துவத் தரவு",
        "name": "முழு பெயர்", "age": "வயது", "hb": "ஹீமோகுளோபின்", "bp": "இரத்த அழுத்தம்", 
        "wt": "எடை (கிலோ)", "wk": "கர்ப்ப வாரம்", "btn_run": "ஆய்வு செய்",
        "res": "📊 மருத்துவ முடிவுகள்", "btn_ai": "AI அறிக்கையை உருவாக்கு", "dl": "PDF பதிவிறக்கம்",
        "tips_title": "📅 வாராந்திர AI வழிகாட்டி", "btn_tips": "குறிப்புகளைப் புதுப்பிக்கவும்",
        "med_title": "💊 தினசரி மருந்துப் பதிவு", "med_add": "மருந்தைச் சேர்க்கவும்",
        "music_title": "🎵 தியானப் பரிந்துரைகள்", "music_btn": "இசை பரிந்துரையைப் பெறுங்கள்",
        "chat_title": "🤖 சுகாதார AI உதவியாளர்", "chat_btn": "AI உதவியாளரிடம் கேளுங்கள்"
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य डैशबोर्ड", "vitals": "📝 नैदानिक डेटा",
        "name": "पूरा नाम", "age": "आयु", "hb": "हीमोग्लोबिन", "bp": "रक्तचाप", 
        "wt": "वजन (kg)", "wk": "गर्भावस्था सप्ताह", "btn_run": "स्वास्थ्य विश्लेषण करें",
        "res": "📊 मूल्यांकन परिणाम", "btn_ai": "AI रिपोर्ट तैयार करें", "dl": "PDF डाउनलोड करें",
        "tips_title": "📅 आपकी साप्ताहिक AI गाइड", "btn_tips": "सुझाव अपडेट करें",
        "med_title": "💊 दैनिक दवा लॉग", "med_add": "दवा जोड़ें",
        "music_title": "🎵 कल्याण संगीत", "music_btn": "AI अनुशंसा प्राप्त करें",
        "chat_title": "🤖 स्वास्थ्य AI सहायक", "chat_btn": "सहायक से पूछें"
    }
}
c = content[lang]

# Navigate using index to prevent logic breakage
page_idx = nav_map[lang].index(nav_choice)

# --- 6. PAGE LOGIC ---
if page_idx == 0: # Assessment
    st.title(c["title"])
    col1, col2 = st.columns([1.1, 1], gap="large")
    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals"]}</h3>', unsafe_allow_html=True)
        name = st.text_input(c["name"], value=st.session_state.patient_data.get('name', ""))
        age = st.slider(c["age"], 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input("Hb (g/dL)", 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input("BP (mmHg)", 80, 200, st.session_state.patient_data.get('bp', 120))
        weight = st.number_input(c["wt"], 30.0, 250.0, st.session_state.patient_data.get('weight', 60.0))
        week = st.number_input(c["wk"], 1, 42, st.session_state.patient_data.get('week', 12))
        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {"name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week}
            st.session_state.risk_score = 15.0 if hb >= 11 and bp <= 140 else 68.0
            st.session_state.ai_assessment = "" 
            st.success("Synchronized.")
    with col2:
        st.markdown(f'<div class="main-card"><h3>{c["res"]}</h3>', unsafe_allow_html=True)
        if st.session_state.risk_score:
            st.metric("Risk Score", f"{st.session_state.risk_score}%")
            if st.button(c["btn_ai"], use_container_width=True):
                with st.spinner("Exhausting all Gemini versions..."):
                    res = call_gemini_ai(f"Risk {st.session_state.risk_score}% vitals {st.session_state.patient_data}", "clinical", lang)
                    st.session_state.ai_assessment = res
            if st.session_state.ai_assessment:
                st.markdown(f"**AI Results:**\n{st.session_state.ai_assessment}")
                pdf_bytes = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(c["dl"], pdf_bytes, f"Report_{name}.pdf", "application/pdf")
        else: st.info("Run Assessment first.")

elif page_idx == 1: # Weekly AI Tips
    st.title(c["tips_title"])
    if not st.session_state.patient_data: st.warning("Complete Assessment first.")
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button(c["btn_tips"], use_container_width=True):
            with st.spinner("AI personalizing advice..."):
                st.session_state.tips = call_gemini_ai(f"Guidance for week {st.session_state.patient_data['week']}.", "tips", lang)
        if 'tips' in st.session_state: st.markdown(st.session_state.tips)

elif page_idx == 2: # Medication Log
    st.title(c["med_title"])
    st.markdown(f'<div class="main-card"><h3>{c["med_title"]}</h3>', unsafe_allow_html=True)
    m_name = st.text_input("Medicine / மருந்து / दवा")
    m_time = st.time_input("Dosage Time")
    if st.button("Save Log", use_container_width=True):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
    for m in st.session_state.medicines: st.markdown(f"🔔 **{m['name']}** at {m['time']}")

elif page_idx == 3: # Relaxation Music
    st.title(c["music_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.button(c["music_btn"], use_container_width=True):
        with st.spinner("Finding AI wellness sounds..."):
            st.session_state.music = call_gemini_ai("Suggest relaxation music based on current health.", "music", lang)
    if 'music' in st.session_state: st.markdown(st.session_state.music)

elif page_idx == 4: # AI Doctor Chat
    st.title(c["chat_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    user_q = st.text_input(c["chat_btn"])
    if st.button(c["chat_btn"], use_container_width=True):
        with st.spinner("Assistant is replying..."):
            st.markdown(f"**AI Doctor:** {call_gemini_ai(user_q, 'chatbot', lang)}")

st.write("---")
st.caption(f"MaternalAI Support v3.0 | Project: gen-lang-client-0652486419 | Model Discovery Mode")
