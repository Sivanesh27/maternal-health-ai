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
# Ensure GEMINI_API_KEY is saved in your Streamlit Cloud 'Secrets' dashboard.
apiKey = "" 

def get_api_key():
    try:
        if st.secrets and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except:
        pass
    return apiKey.strip()

def local_clinical_brain(hb, bp, risk_score, lang):
    """Rule-based clinical fallback if the API is completely unreachable."""
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
        if hb >= 11 and bp <= 140: advice += "Status: Vitals appear stable for your current progress."
        return advice + "\n\n(Note: This is a rule-based fallback generated locally due to API connection issues.)"

def call_gemini_ai(prompt, context_type="general", language="English"):
    """
    Hyper-Resilient Engine: Prioritizes v1 Stable endpoint with merged prompts.
    This bypasses 404/400 errors by using the most basic, compatible JSON structure.
    """
    current_key = get_api_key()
    patient_ctx = st.session_state.get('patient_data', {})
    hb = patient_ctx.get('hb', 11.0)
    bp = patient_ctx.get('bp', 120)
    score = st.session_state.get('risk_score', 0)

    if not current_key or len(current_key) < 10:
        return local_clinical_brain(hb, bp, score, language)

    # Order prioritized for production stability
    discovery_paths = [
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-flash"),
        ("v1", "gemini-1.5-pro"),
        ("v1beta", "gemini-1.5-flash-latest")
    ]
    
    prompts = {
        "clinical": "Senior Maternal Health Consultant. Provide a professional clinical risk assessment and roadmap.",
        "tips": "Provide 3 unique, medical-backed pregnancy tips for this specific week and vitals.",
        "music": "Recommend 3 specific relaxation soundscapes or meditation styles for this pregnancy stage.",
        "chatbot": "Supportive maternal health assistant. Answer queries with empathy and clarity."
    }
    
    instr = prompts.get(context_type, "Maternal health assistant.")
    # Merged prompt for maximum compatibility across all project tiers
    full_query = (f"System Role: {instr}\n"
                  f"LANGUAGE: Respond ONLY in {language}.\n"
                  f"PATIENT DATA: {patient_ctx}\n"
                  f"USER QUERY: {prompt}\n\n"
                  f"Instructions: Give a concise, professional answer. Do not use bold markdown.")

    payload = {"contents": [{"role": "user", "parts": [{"text": full_query}]}]}
    headers = {"Content-Type": "application/json"}
    
    for version, model_name in discovery_paths:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={current_key}"
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
        except:
            continue
        time.sleep(0.2) 

    return local_clinical_brain(hb, bp, score, language)

# --- PDF GENERATION ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MaternalAI Clinical Health Report', 0, 1, 'C')
        self.ln(10)

def create_pdf(data, ai_note):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Patient Name: {data.get('name', 'N/A')}", 0, 1)
    pdf.set_font("Arial", '', 11)
    vitals = (f"Age: {data.get('age')} | Pregnancy Week: {data.get('week')}\n"
              f"Hemoglobin: {data.get('hb')} g/dL | BP: {data.get('bp')} mmHg | Weight: {data.get('weight')} kg")
    pdf.multi_cell(0, 10, vitals)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical AI Insight:", 0, 1)
    pdf.set_font("Arial", '', 10)
    # Strip non-latin characters for PDF
    clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    pdf.multi_cell(0, 6, clean_ai)
    return pdf.output(dest='S').encode('latin-1')

# --- UI STYLING & TEXT VISIBILITY ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* UNIVERSAL TEXT VISIBILITY */
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown, [data-testid="stMetricLabel"], .stRadio label, .stSelectbox label { 
        color: #0f172a !important; 
        font-weight: 600 !important; 
    }
    
    /* INPUT FIELD READABILITY */
    input, textarea, [data-baseweb="input"], [data-baseweb="select"], .stNumberInput div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }

    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: white !important; font-family: 'Playfair Display', serif; font-size: 50px; font-weight: 900;
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
        padding: 16px 32px; border: none; box-shadow: 0 4px 15px rgba(67, 56, 202, 0.25);
        transition: 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-4px); box-shadow: 0 12px 25px rgba(67, 56, 202, 0.5); }
    
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
    
    # Navigation strings are localized based on the selection
    nav_options = {
        "English": ["Assessment", "Weekly AI Tips", "Medication Log", "Relaxation Music", "AI Health Chat"],
        "தமிழ்": ["மதிப்பீடு", "வாராந்திர குறிப்புகள்", "மருந்துப் பதிவு", "இசை", "AI மருத்துவர்"],
        "हिन्दी": ["मूल्यांकन", "साप्ताहिक सुझाव", "दवा लॉग", "संगीत", "AI डॉक्टर"]
    }
    nav_choice = st.radio("Menu / மெனு / मेनू", nav_options[lang])
    st.write("---")
    st.success("Personalization Engine: Online 🟢")

# --- MULTILINGUAL CONTENT ---
content = {
    "English": {
        "title": "Maternal Health Dashboard", "vitals": "📝 Clinical Inputs",
        "name": "Full Name", "age": "Age", "hb": "Hemoglobin", "bp": "Systolic BP", 
        "wt": "Weight", "wk": "Week", "btn_run": "Analyze Health",
        "res": "📊 Assessment Results", "btn_ai": "Generate AI Report", "dl": "Download PDF",
        "tips_title": "📅 Personalized Weekly Tips", "btn_tips": "Refresh Tips",
        "med_title": "💊 Medication Log", "med_add": "Add Medicine", "med_btn": "Save Log",
        "music_title": "🎵 Relaxation Music", "music_btn": "Get AI Recommendation",
        "chat_title": "🤖 Health Assistant Chat", "chat_btn": "Ask Assistant"
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார மேலாண்மை", "vitals": "📝 மருத்துவத் தரவு",
        "name": "முழு பெயர்", "age": "வயது", "hb": "ஹீமோகுளோபின்", "bp": "இரத்த அழுத்தம்", 
        "wt": "எடை (கிலோ)", "wk": "வாரம்", "btn_run": "ஆய்வு செய்",
        "res": "📊 மருத்துவ முடிவுகள்", "btn_ai": "AI அறிக்கையை உருவாக்கு", "dl": "PDF பதிவிறக்கம்",
        "tips_title": "📅 வாராந்திர AI வழிகாட்டி", "btn_tips": "AI குறிப்புகளைப் புதுப்பிக்கவும்",
        "med_title": "💊 மருந்துப் பதிவு", "med_add": "மருந்தைச் சேர்க்கவும்", "med_btn": "சேமிக்கவும்",
        "music_title": "🎵 இசை பரிந்துரைகள்", "music_btn": "பரிந்துரையைப் பெறுங்கள்",
        "chat_title": "🤖 சுகாதார AI உதவியாளர்", "chat_btn": "AI உதவியாளரிடம் கேளுங்கள்"
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य डैशबोर्ड", "vitals": "📝 नैदानिक डेटा",
        "name": "पूरा नाम", "age": "आयु", "hb": "हीमोग्लोबिन", "bp": "रक्तचाप (BP)", 
        "wt": "वजन (kg)", "wk": "गर्भावस्था सप्ताह", "btn_run": "स्वास्थ्य विश्लेषण करें",
        "res": "📊 मूल्यांकन परिणाम", "btn_ai": "AI रिपोर्ट तैयार करें", "dl": "PDF डाउनलोड करें",
        "tips_title": "📅 साप्ताहिक AI गाइड", "btn_tips": "सुझाव अपडेट करें",
        "med_title": "💊 दवा लॉग", "med_add": "दवा जोड़ें", "med_btn": "सहेजें",
        "music_title": "🎵 कल्याण संगीत अनुशंसाएं", "music_btn": "संगीत अनुशंसा प्राप्त करें",
        "chat_title": "🤖 स्वास्थ्य AI साथी", "chat_btn": "सहायक से पूछें"
    }
}
c = content[lang]

# Ensure navigation logic is language-independent by index
nav_idx = nav_options[lang].index(nav_choice)

# --- PAGE LOGIC ---
if nav_idx == 0: # Assessment
    st.title(c["title"])
    col1, col2 = st.columns([1.1, 1], gap="large")
    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals"]}</h3>', unsafe_allow_html=True)
        name = st.text_input(c["name"], value=st.session_state.patient_data.get('name', ""))
        age = st.slider(c["age"], 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input("Hemoglobin / ஹீமோகுளோபின்", 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input("Systolic BP / இரத்த அழுத்தம்", 80, 200, st.session_state.patient_data.get('bp', 120))
        weight = st.number_input("Weight / எடை", 30.0, 250.0, st.session_state.patient_data.get('weight', 60.0))
        week = st.number_input("Pregnancy Week / வாரம்", 1, 42, st.session_state.patient_data.get('week', 12))
        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {"name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week}
            st.session_state.risk_score = 15.0 if hb >= 11 and bp <= 140 else 68.0
            st.session_state.ai_assessment = "" 
            st.success("Vitals Synchronized.")
    with col2:
        st.markdown(f'<div class="main-card"><h3>{c["res"]}</h3>', unsafe_allow_html=True)
        if st.session_state.risk_score:
            st.metric("Risk Score", f"{st.session_state.risk_score}%")
            if st.button(c["btn_ai"], use_container_width=True):
                with st.spinner("AI analyzing your specific clinical case..."):
                    res = call_gemini_ai(f"Risk {st.session_state.risk_score}% with vitals {st.session_state.patient_data}", "clinical", lang)
                    st.session_state.ai_assessment = res
            if st.session_state.ai_assessment:
                st.markdown(f"**AI Insight:**\n{st.session_state.ai_assessment}")
                pdf_bytes = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(c["dl"], pdf_bytes, f"Report_{name}.pdf", "application/pdf")
        else: st.info("Run Assessment first.")

elif nav_idx == 1: # Weekly Tips
    st.title(c["tips_title"])
    if not st.session_state.patient_data: st.warning("Complete Assessment first.")
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button(c["btn_tips"], use_container_width=True):
            with st.spinner("AI Generating personalized tips..."):
                st.session_state.tips = call_gemini_ai(f"Generate 3 tips for week {st.session_state.patient_data['week']}.", "tips", lang)
        if 'tips' in st.session_state: st.markdown(st.session_state.tips)

elif nav_idx == 2: # Medication Log
    st.title(c["med_title"])
    st.markdown(f'<div class="main-card"><h3>{c["med_add"]}</h3>', unsafe_allow_html=True)
    m_name = st.text_input("Medicine Name")
    m_time = st.time_input("Reminder Time")
    if st.button(c["med_btn"], use_container_width=True):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
    for m in st.session_state.medicines: st.markdown(f"🔔 **{m['name']}** at {m['time']}")

elif nav_idx == 3: # Relaxation Music
    st.title(c["music_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.button(c["music_btn"], use_container_width=True):
        with st.spinner("Finding AI wellness sounds..."):
            st.session_state.music = call_gemini_ai("Suggest relaxation music based on current health.", "music", lang)
    if 'music' in st.session_state: st.markdown(st.session_state.music)

elif nav_idx == 4: # AI Chat
    st.title(c["chat_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    user_q = st.text_input("Question")
    if st.button(c["chat_btn"], use_container_width=True):
        st.markdown(f"**AI:** {call_gemini_ai(user_q, 'chatbot', lang)}")

st.write("---")
st.caption("MaternalAI Support v3.0 | Secure Handshake Engine | Universal Language Translation")
