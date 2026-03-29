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
# Ensure GEMINI_API_KEY is set in Streamlit Secrets.
apiKey = "" 

def get_api_key():
    try:
        if st.secrets and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except:
        pass
    return os.environ.get("GEMINI_API_KEY", apiKey)

def call_gemini_ai(prompt, context_type="general", language="English"):
    """
    AI Discovery Engine: Cycles through multiple endpoints to bypass 404/400 errors.
    Automatically personalizes responses based on stored patient data.
    """
    current_key = get_api_key()
    if not current_key or current_key == "":
        return "Error: API Key is missing. Please check your Streamlit Secrets."

    # Multi-path discovery list
    discovery_paths = [
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-flash-latest"),
        ("v1", "gemini-pro"),
        ("v1beta", "gemini-1.5-flash")
    ]
    
    prompts = {
        "clinical": "Professional maternal health consultant. Provide clinical risk assessment and next steps.",
        "tips": "Provide 3 medical-backed tips for this specific pregnancy week and vitals.",
        "music": "Recommend 3 music/meditation types for mental wellness based on current pregnancy stage.",
        "chatbot": "Friendly maternal health assistant. Answer with medical empathy."
    }
    
    instr = prompts.get(context_type, "Maternal health assistant.")
    patient_ctx = st.session_state.get('patient_data', 'No data entered yet.')
    
    # Combined prompt for maximum compatibility
    full_query = (f"System Instruction: {instr}\n"
                  f"LANGUAGE: Respond ONLY in {language}.\n"
                  f"PATIENT CONTEXT: {patient_ctx}\n"
                  f"USER REQUEST: {prompt}")

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": full_query}]
        }]
    }
    
    last_error = ""
    for version, model_name in discovery_paths:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={current_key}"
        try:
            response = requests.post(url, json=payload, timeout=20)
            if response.status_code == 200:
                res_json = response.json()
                return res_json['candidates'][0]['content']['parts'][0]['text']
            else:
                last_error = f"{model_name} ({version}) -> {response.status_code}"
        except Exception as e:
            last_error = str(e)
        time.sleep(0.5) 

    return f"AI Connection Failed. (Error: {last_error}). Please ensure 'Generative Language API' is enabled in your Google Cloud Project."

# --- PDF ENGINE ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MaternalAI - Personalized Clinical Report', 0, 1, 'C')
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
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical AI Insight:", 0, 1)
    pdf.set_font("Arial", '', 10)
    # Basic encoding fix for PDF
    clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    pdf.multi_cell(0, 6, clean_ai)
    return pdf.output(dest='S').encode('latin-1')

# --- UI STYLING & TEXT VISIBILITY ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* UNIVERSAL TEXT VISIBILITY - Forces all text to be dark and visible */
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown, [data-testid="stMetricLabel"], .stRadio label { 
        color: #0f172a !important; 
        font-weight: 600 !important; 
    }
    
    /* Input Field Fix */
    input, textarea, [data-baseweb="input"], [data-baseweb="select"], .stNumberInput div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }

    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: white !important; font-family: 'Georgia', serif; font-size: 50px; font-weight: 900;
        width: 90px; height: 90px; display: flex; align-items: center; 
        justify-content: center; border-radius: 26px; border: 3px solid white;
        box-shadow: 0 12px 30px rgba(30, 27, 75, 0.4);
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
        transition: 0.4s ease;
    }
    .stButton>button:hover { transform: translateY(-4px); box-shadow: 0 12px 25px rgba(67, 56, 202, 0.5); }
    
    [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'ai_assessment' not in st.session_state: st.session_state.ai_assessment = ""
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'risk_score' not in st.session_state: st.session_state.risk_score = None

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin:30px 0;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Choose Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
    nav = st.radio("Personalized Menu", ["Home & Assessment", "Weekly AI Tips", "Medication Log", "Personalized Music", "AI Doctor Chat"])
    st.write("---")
    st.success("Personalization Engine: Active 🟢")

# --- MULTILINGUAL CONTENT MAP ---
content = {
    "English": {
        "title": "Maternal Health Dashboard", "vitals": "📝 Clinical Inputs",
        "btn_run": "Analyze My Health", "res": "📊 Assessment Outcome",
        "btn_ai": "Generate AI Personalized Insight", "dl": "Download Medical PDF",
        "tips_title": "📅 Your Weekly AI Guide", "btn_tips": "Refresh Tips",
        "med_title": "💊 Daily Medication Log", "med_btn": "Add to Schedule",
        "music_title": "🎵 AI Relaxation Recommendations", "music_btn": "Get AI Recommendation",
        "chat_title": "🤖 Health Companion Chat"
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார மேலாண்மை", "vitals": "📝 மருத்துவத் தரவு",
        "btn_run": "ஆரோக்கியத்தை ஆய்வு செய்", "res": "📊 மருத்துவ முடிவு",
        "btn_ai": "தனிப்பயனாக்கப்பட்ட AI விளக்கத்தைப் பெறுங்கள்", "dl": "அறிக்கையைப் பதிவிறக்கவும்",
        "tips_title": "📅 வாராந்திர AI வழிகாட்டி", "btn_tips": "குறிப்புகளைப் புதுப்பிக்கவும்",
        "med_title": "💊 தினசரி மருந்துப் பதிவு", "med_btn": "அட்டவணையில் சேர்க்கவும்",
        "music_title": "🎵 இசை மற்றும் தியானப் பரிந்துரைகள்", "music_btn": "இசை பரிந்துரையைப் பெறுங்கள்",
        "chat_title": "🤖 சுகாதார AI உரையாடல்"
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य डैशबोर्ड", "vitals": "📝 नैदानिक डेटा",
        "btn_run": "स्वास्थ्य विश्लेषण करें", "res": "📊 मूल्यांकन परिणाम",
        "btn_ai": "व्यक्तिगत AI रिपोर्ट तैयार करें", "dl": "रिपोर्ट डाउनलोड करें",
        "tips_title": "📅 साप्ताहिक AI गाइड", "btn_tips": "सुझाव अपडेट करें",
        "med_title": "💊 दैनिक दवा लॉग", "med_btn": "लॉग में जोड़ें",
        "music_title": "🎵 कल्याण संगीत अनुशंसाएं", "music_btn": "संगीत अनुशंसा प्राप्त करें",
        "chat_title": "🤖 स्वास्थ्य AI साथी"
    }
}
c = content[lang]

# --- PAGE: ASSESSMENT ---
if nav == "Home & Assessment":
    st.title(c["title"])
    col1, col2 = st.columns([1.1, 1], gap="large")
    
    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals"]}</h3>', unsafe_allow_html=True)
        name = st.text_input("Full Name", value=st.session_state.patient_data.get('name', ""))
        age = st.slider("Age", 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input("Hemoglobin (g/dL)", 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input("Systolic BP (mmHg)", 80, 200, st.session_state.patient_data.get('bp', 120))
        weight = st.number_input("Weight (kg)", 30.0, 250.0, st.session_state.patient_data.get('weight', 60.0))
        week = st.number_input("Pregnancy Week", 1, 42, st.session_state.patient_data.get('week', 12))
        
        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {"name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week}
            # Heuristic Logic
            st.session_state.risk_score = 15.0 if hb >= 11 and bp <= 140 else 68.0
            st.session_state.ai_assessment = "" 
            st.success("Vitals saved.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="main-card"><h3>{c["res"]}</h3>', unsafe_allow_html=True)
        if st.session_state.risk_score:
            st.metric("Risk Score", f"{st.session_state.risk_score}%")
            if st.session_state.risk_score > 50: st.error("⚠️ STATUS: HIGH RISK")
            else: st.success("✅ STATUS: LOW RISK")
            
            st.write("---")
            if st.button(c["btn_ai"], use_container_width=True):
                with st.spinner("AI discovering best connection path..."):
                    res = call_gemini_ai(f"Evaluate risk of {st.session_state.risk_score}% with vitals: {st.session_state.patient_data}", "clinical", lang)
                    st.session_state.ai_assessment = res
            
            if st.session_state.ai_assessment:
                st.markdown(f"**AI Insight:**\n{st.session_state.ai_assessment}")
                pdf_bytes = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(c["dl"], pdf_bytes, f"Report_{name}.pdf", "application/pdf", use_container_width=True)
        else:
            st.info("Input vitals to begin analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: TIPS ---
elif nav == "Weekly AI Tips":
    st.title(c["tips_title"])
    if not st.session_state.patient_data:
        st.warning("Please complete Health Assessment first.")
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button(c["btn_tips"], use_container_width=True):
            with st.spinner("AI is generating real-time tips for you..."):
                st.session_state.tips = call_gemini_ai(f"Generate 3 personalized tips for week {st.session_state.patient_data['week']}.", "tips", lang)
        if 'tips' in st.session_state:
            st.markdown(st.session_state.tips)
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: MED LOG ---
elif nav == "Medication Log":
    st.title(c["med_title"])
    st.markdown(f'<div class="main-card">', unsafe_allow_html=True)
    m_name = st.text_input("Medicine Name")
    m_time = st.time_input("Daily Time")
    if st.button(c["med_btn"], use_container_width=True):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
        st.success(f"Log Updated: {m_name}")
    if st.session_state.medicines:
        st.write("---")
        for m in st.session_state.medicines:
            st.markdown(f"🔔 **{m['name']}** at {m['time']}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: MUSIC ---
elif nav == "Personalized Music":
    st.title(c["music_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.button(c["music_btn"], use_container_width=True):
        with st.spinner("AI finding peaceful sounds for your stage..."):
            st.session_state.music = call_gemini_ai("Suggest music based on my risk score and week.", "music", lang)
    if 'music' in st.session_state:
        st.markdown(st.session_state.music)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: CHAT ---
elif nav == "AI Doctor Chat":
    st.title(c["chat_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    user_q = st.text_input("Ask a clinical or lifestyle question...")
    if st.button("Ask AI Companion", use_container_width=True):
        with st.spinner("Thinking..."):
            st.markdown(f"**AI Doctor:** {call_gemini_ai(user_q, 'chatbot', lang)}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.caption("MaternalAI Decision Support v2.9 | Discovery Engine & High-Contrast Mode Enabled")
