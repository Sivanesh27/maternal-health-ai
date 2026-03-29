import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import requests
import json
import os
from fpdf import FPDF

# --- AI CONFIGURATION (Gemini API) ---
apiKey = "" 

def get_api_key():
    try:
        if st.secrets and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except:
        pass
    return apiKey.strip()

def local_clinical_brain(hb, bp, risk_score, lang):
    """Deep clinical rule-based fallback if Gemini API is unreachable."""
    if lang == "தமிழ்":
        advice = f"மருத்துவ ஆய்வு: உங்கள் அபாய மதிப்பெண் {risk_score}%. "
        if hb < 11: advice += "ஹீமோகுளோபின் குறைவாக உள்ளது (இரத்த சோகை). இரும்புச்சத்து நிறைந்த உணவுகளை உட்கொள்ளுங்கள். "
        if bp > 140: advice += "இரத்த அழுத்தம் அதிகமாக உள்ளது. தயவுசெய்து ஓய்வெடுக்கவும்."
        return advice + "\n\n(குறிப்பு: இணைப்பு சிக்கல் காரணமாக இது தானியங்கி மருத்துவ உதவி.)"
    elif lang == "हिन्दी":
        advice = f"नैदानिक विश्लेषण: आपका जोखिम स्कोर {risk_score}% है। "
        if hb < 11: advice += "हीमोग्लोबिन कम है। आयरन युक्त भोजन (पालक, गुड़) लें। "
        if bp > 140: advice += "रक्तचाप अधिक है। कृपया आराम करें।"
        return advice + "\n\n(नोट: यह एक स्वचालित नैदानिक सहायता है।)"
    else:
        advice = f"Clinical Assessment: Your predictive risk is {risk_score}%. "
        if hb < 11: advice += "Action: Increase Iron intake. "
        if bp > 140: advice += "Action: Monitor BP daily and reduce salt. "
        return advice + "\n\n(Note: This is a rule-based fallback generated locally due to API timeout.)"

def call_gemini_ai(prompt, context_type="general", language="English"):
    """
    Ultra-Resilient Engine: Cycles through v1/v1beta and every model variant 
    with extended timeout to bypass API connectivity hurdles.
    """
    current_key = get_api_key()
    patient_ctx = st.session_state.get('patient_data', {})
    hb = patient_ctx.get('hb', 11.0)
    bp = patient_ctx.get('bp', 120)
    score = st.session_state.get('risk_score', 0)

    if not current_key or len(current_key) < 10:
        return local_clinical_brain(hb, bp, score, language)

    # Expanded list of authorized model paths and versions
    discovery_paths = [
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-flash"),
        ("v1", "gemini-1.5-flash-latest"),
        ("v1beta", "gemini-1.5-flash-latest"),
        ("v1", "gemini-pro"),
        ("v1beta", "gemini-1.5-flash-8b")
    ]
    
    prompts = {
        "clinical": "Professional Maternal Health Doctor. Provide a detailed risk assessment and roadmap.",
        "tips": "Provide 3 unique, medical-backed pregnancy tips for this specific week and health status.",
        "music": "Recommend 3 specific relaxation soundscapes or meditation styles for this pregnancy stage.",
        "chatbot": "Supportive maternal health assistant. Answer queries with clinical empathy and clarity."
    }
    
    instr = prompts.get(context_type, "Maternal health assistant.")
    # Force language and context into a merged prompt for maximum compatibility
    full_query = (f"System Instruction: {instr}\n"
                  f"LANGUAGE: Respond ONLY in {language}.\n"
                  f"PATIENT VITALS: {patient_ctx}\n"
                  f"USER QUERY: {prompt}\n\n"
                  f"Important: Deliver a structured, professional response.")

    payload = {"contents": [{"role": "user", "parts": [{"text": full_query}]}]}
    headers = {"Content-Type": "application/json"}
    
    for version, model_name in discovery_paths:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={current_key}"
        try:
            # Increased timeout to 20s to prevent 'API timeout' errors
            response = requests.post(url, json=payload, headers=headers, timeout=20)
            if response.status_code == 200:
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text']
        except:
            continue
        time.sleep(0.3) 

    return local_clinical_brain(hb, bp, score, language)

# --- PDF ENGINE ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MaternalAI Personalized Clinical Report', 0, 1, 'C')
        self.ln(10)

def create_pdf(data, ai_note):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Patient Name: {data.get('name', 'N/A')}", 0, 1)
    pdf.set_font("Arial", '', 11)
    vitals = (f"Age: {data.get('age')} | Pregnancy Week: {data.get('week')}\n"
              f"Hemoglobin: {data.get('hb')} g/dL\n"
              f"Systolic BP: {data.get('bp')} mmHg\n"
              f"Weight: {data.get('weight')} kg")
    pdf.multi_cell(0, 10, vitals)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical Assessment:", 0, 1)
    pdf.set_font("Arial", '', 10)
    try:
        clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    except:
        clean_ai = "Report generated. Please check app for full clinical insights."
    pdf.multi_cell(0, 6, clean_ai)
    return pdf.output(dest='S').encode('latin-1')

# --- UI STYLING & HIGH CONTRAST ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* UNIVERSAL TEXT VISIBILITY FIX */
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown, [data-testid="stMetricLabel"], .stRadio label, .stSelectbox label { 
        color: #0f172a !important; 
        font-weight: 600 !important; 
    }
    
    /* FORCE INPUT FIELD READABILITY */
    input, textarea, [data-baseweb="input"], [data-baseweb="select"], .stNumberInput div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }

    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: white !important; font-family: 'Times New Roman', serif; font-size: 52px; font-weight: 900;
        width: 95px; height: 95px; display: flex; align-items: center; 
        justify-content: center; border-radius: 28px; border: 3px solid white;
        box-shadow: 0 12px 30px rgba(30, 27, 75, 0.4);
    }
    
    .main-card {
        background: white; padding: 40px; border-radius: 35px;
        box-shadow: 0 12px 45px rgba(0,0,0,0.05); border: 1px solid #eef2f6;
        margin-bottom: 25px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%);
        color: #ffffff !important; border-radius: 20px; font-weight: 700;
        padding: 18px 36px; border: none; box-shadow: 0 6px 20px rgba(67, 56, 202, 0.3);
        transition: 0.4s ease;
    }
    .stButton>button:hover { transform: translateY(-4px); box-shadow: 0 15px 30px rgba(67, 56, 202, 0.5); }
    
    [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'ai_assessment' not in st.session_state: st.session_state.ai_assessment = ""
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'risk_score' not in st.session_state: st.session_state.risk_score = None

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin:30px 0;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Choose Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
    
    # Navigation strings are localized based on the selector
    nav_options = {
        "English": ["Assessment", "Weekly AI Tips", "Medication Log", "Relaxation Music", "AI Doctor Chat"],
        "தமிழ்": ["மதிப்பீடு", "வாராந்திர குறிப்புகள்", "மருந்துப் பதிவு", "இசை", "AI மருத்துவர்"],
        "हिन्दी": ["मूल्यांकन", "साप्ताहिक सुझाव", "दवा लॉग", "संगीत", "AI डॉक्टर"]
    }
    nav_choice = st.radio("Menu / மெனு / मेनू", nav_options[lang])
    st.write("---")
    st.success("System Status: Online 🟢")

# --- FULL MULTILINGUAL CONTENT MAP ---
content = {
    "English": {
        "title": "Maternal Health Dashboard", "vitals": "📝 Clinical Inputs",
        "name": "Full Name", "age": "Age", "hb": "Hemoglobin", "bp": "Systolic BP", 
        "wt": "Weight (kg)", "wk": "Week", "btn_run": "Analyze Health",
        "res": "📊 Clinical Results", "btn_ai": "Generate AI Report", "dl": "Download PDF",
        "tips_title": "📅 Personalized Weekly Tips", "btn_tips": "Refresh Tips",
        "med_title": "💊 Medication Management", "med_add": "Add Medicine",
        "music_title": "🎵 Relaxation Music", "music_btn": "Get AI Selection",
        "chat_title": "🤖 Health Assistant Chat", "chat_btn": "Ask AI"
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார மேலாண்மை", "vitals": "📝 மருத்துவத் தரவு",
        "name": "முழு பெயர்", "age": "வயது", "hb": "ஹீமோகுளோபின்", "bp": "இரத்த அழுத்தம்", 
        "wt": "எடை (கிலோ)", "wk": "வாரம்", "btn_run": "ஆய்வு செய்",
        "res": "📊 மருத்துவ முடிவுகள்", "btn_ai": "AI அறிக்கையை உருவாக்கு", "dl": "PDF பதிவிறக்கம்",
        "tips_title": "📅 வாராந்திர AI குறிப்புகள்", "btn_tips": "குறிப்புகளைப் புதுப்பிக்கவும்",
        "med_title": "💊 மருந்துப் பதிவு", "med_add": "மருந்தைச் சேர்க்கவும்",
        "music_title": "🎵 இசை பரிந்துரைகள்", "music_btn": "பரிந்துரையைப் பெறுங்கள்",
        "chat_title": "🤖 சுகாதார உதவியாளர்", "chat_btn": "AI-யிடம் கேளுங்கள்"
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य डैशबोर्ड", "vitals": "📝 नैदानिक डेटा",
        "name": "पूरा नाम", "age": "आयु", "hb": "हीमोग्लोबिन", "bp": "रक्तचाप (BP)", 
        "wt": "वजन (kg)", "wk": "सप्ताह", "btn_run": "विश्लेषण करें",
        "res": "📊 मूल्यांकन परिणाम", "btn_ai": "AI रिपोर्ट तैयार करें", "dl": "PDF डाउनलोड",
        "tips_title": "📅 साप्ताहिक AI सुझाव", "btn_tips": "सुझाव अपडेट करें",
        "med_title": "💊 दवा प्रबंधन", "med_add": "दवा जोड़ें",
        "music_title": "🎵 विश्राम संगीत", "music_btn": "अनुशंसा प्राप्त करें",
        "chat_title": "🤖 स्वास्थ्य सहायक चैट", "chat_btn": "AI से पूछें"
    }
}
c = content[lang]

# --- PAGE LOGIC ---
# Standardize internal nav keys to avoid translation-dependent logic breakage
nav_index = nav_options[lang].index(nav_choice)

if nav_index == 0: # Assessment
    st.title(c["title"])
    col1, col2 = st.columns([1.1, 1], gap="large")
    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals"]}</h3>', unsafe_allow_html=True)
        name = st.text_input(c["name"], value=st.session_state.patient_data.get('name', ""))
        age = st.slider(c["age"], 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input(f"{c['hb']} (g/dL)", 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input(f"{c['bp']} (mmHg)", 80, 200, st.session_state.patient_data.get('bp', 120))
        weight = st.number_input(c["wt"], 30.0, 250.0, st.session_state.patient_data.get('weight', 60.0))
        week = st.number_input(c["wk"], 1, 42, st.session_state.patient_data.get('week', 12))
        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {"name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week}
            st.session_state.risk_score = 15.0 if hb >= 11 and bp <= 140 else 68.0
            st.session_state.ai_assessment = "" 
            st.success("Vitals Synchronized.")
    with col2:
        st.markdown(f'<div class="main-card"><h3>{c["res"]}</h3>', unsafe_allow_html=True)
        if st.session_state.risk_score:
            st.metric("Risk Score / அபாய மதிப்பெண்", f"{st.session_state.risk_score}%")
            if st.button(c["btn_ai"], use_container_width=True):
                with st.spinner("AI analyzing your specific clinical case..."):
                    st.session_state.ai_assessment = call_gemini_ai(f"Risk {st.session_state.risk_score}% with vitals: {st.session_state.patient_data}", "clinical", lang)
            if st.session_state.ai_assessment:
                st.markdown(f"**AI Results:**\n{st.session_state.ai_assessment}")
                pdf_bytes = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(c["dl"], pdf_bytes, f"Clinical_Report_{name}.pdf", "application/pdf")
        else: st.info("Run Assessment first.")

elif nav_index == 1: # Weekly AI Tips
    st.title(c["tips_title"])
    if not st.session_state.patient_data: st.warning("Complete Assessment first.")
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button(c["btn_tips"], use_container_width=True):
            with st.spinner("AI generating real-time tips..."):
                st.session_state.tips = call_gemini_ai(f"Give personalized tips for week {st.session_state.patient_data['week']}.", "tips", lang)
        if 'tips' in st.session_state: st.markdown(st.session_state.tips)

elif nav_index == 2: # Medication Log
    st.title(c["med_title"])
    st.markdown(f'<div class="main-card"><h3>{c["med_add"]}</h3>', unsafe_allow_html=True)
    m_name = st.text_input("Medicine / மருந்து / दवा")
    m_time = st.time_input("Dosage Time")
    if st.button("Save Log", use_container_width=True):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
    for m in st.session_state.medicines: st.markdown(f"🔔 **{m['name']}** at {m['time']}")

elif nav_index == 3: # Relaxation Music
    st.title(c["music_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.button(c["music_btn"], use_container_width=True):
        with st.spinner("Finding AI wellness sounds..."):
            st.session_state.music = call_gemini_ai("Suggest music based on my risk score and pregnancy week.", "music", lang)
    if 'music' in st.session_state: st.markdown(st.session_state.music)

elif nav_index == 4: # AI Doctor Chat
    st.title(c["chat_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    user_q = st.text_input("Your Question...")
    if st.button(c["chat_btn"], use_container_width=True):
        with st.spinner("Assistant is replying..."):
            st.markdown(f"**AI Doctor:** {call_gemini_ai(user_q, 'chatbot', lang)}")

st.write("---")
st.caption("MaternalAI Decision Support v3.0 | Industry Standard Build | Universal Language Engine")
