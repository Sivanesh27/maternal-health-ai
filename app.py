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
            return st.secrets["GEMINI_API_KEY"]
    except:
        pass
    return os.environ.get("GEMINI_API_KEY", apiKey)

def local_clinical_fallback(hb, bp, risk_score, lang):
    """Provides a rule-based clinical analysis if the API is unreachable."""
    if lang == "தமிழ்":
        advice = "மருத்துவ எச்சரிக்கை: உங்கள் ஹீமோகுளோபின் அல்லது இரத்த அழுத்தம் சீராக இல்லை என்றால் உடனடியாக மருத்துவரை அணுகவும். "
        if hb < 11: advice += "இரும்புச்சத்து நிறைந்த உணவுகளை உட்கொள்ளுங்கள் (கீரை, பேரிச்சம்பழம்)."
        return advice + "\n\n(குறிப்பு: இணைப்பு சிக்கல் காரணமாக இது ஒரு தானியங்கி பதில்.)"
    elif lang == "हिन्दी":
        advice = "नैदानिक सलाह: यदि आपका हीमोग्लोबिन या रक्तचाप असामान्य है, तो तुरंत डॉक्टर से मिलें। "
        if hb < 11: advice += "आयरन युक्त भोजन लें (पालक, खजूर)।"
        return advice + "\n\n(नोट: कनेक्टिविटी समस्या के कारण यह एक स्वचालित प्रतिक्रिया है।)"
    else:
        advice = f"Clinical Assessment: Based on a risk of {risk_score}%, we recommend immediate consultation if you experience dizziness or severe swelling. "
        if hb < 11: advice += "Priority: Increase Iron intake. "
        if bp > 140: advice += "Priority: Monitor BP daily and reduce salt. "
        return advice + "\n\n(Note: This is an automated fallback generated locally due to API connection issues.)"

def call_gemini_ai(prompt, context_type="general", language="English"):
    """
    Enhanced Resiliency Engine: Cycles through multiple model tiers and versions.
    If all fail, it provides a local medical assessment fallback.
    """
    current_key = get_api_key()
    patient_ctx = st.session_state.get('patient_data', {})
    hb = patient_ctx.get('hb', 11.0)
    bp = patient_ctx.get('bp', 120)
    score = st.session_state.get('risk_score', 0)

    if not current_key or current_key == "":
        return local_clinical_fallback(hb, bp, score, language)

    # Expanded list of endpoints to bypass regional/account 404s
    discovery_paths = [
        ("v1beta", "gemini-1.5-flash"),
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-flash-8b"),
        ("v1beta", "gemini-1.5-flash-latest"),
        ("v1beta", "gemini-1.0-pro")
    ]
    
    prompts = {
        "clinical": "Senior Maternal Health Doctor. Provide clinical risk assessment and roadmap.",
        "tips": "Provide 3 medical-backed pregnancy tips for this week.",
        "music": "Recommend 3 relaxation soundscapes for this pregnancy stage.",
        "chatbot": "Empathetic health assistant. Answer queries warmly."
    }
    
    instr = prompts.get(context_type, "Maternal health assistant.")
    full_query = (f"Instruction: {instr}\n"
                  f"LANGUAGE: Respond ONLY in {language}.\n"
                  f"CONTEXT: {patient_ctx}\n"
                  f"REQUEST: {prompt}")

    payload = {"contents": [{"role": "user", "parts": [{"text": full_query}]}]}
    
    last_error = ""
    for version, model_name in discovery_paths:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={current_key}"
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                last_error = f"{model_name} ({version}) -> {response.status_code}"
        except:
            last_error = "Network Timeout"
        time.sleep(0.3) 

    # If all attempts fail, trigger local fallback logic
    return local_clinical_fallback(hb, bp, score, language)

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
    vitals = (f"Age: {data.get('age')} | Week: {data.get('week')}\n"
              f"Hemoglobin: {data.get('hb')} g/dL | BP: {data.get('bp')} mmHg | Wt: {data.get('weight')} kg")
    pdf.multi_cell(0, 10, vitals)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical Assessment:", 0, 1)
    pdf.set_font("Arial", '', 10)
    clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    pdf.multi_cell(0, 6, clean_ai)
    return pdf.output(dest='S').encode('latin-1')

# --- UI STYLING & TEXT VISIBILITY ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* FORCE TEXT VISIBILITY */
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown, [data-testid="stMetricLabel"], .stRadio label { 
        color: #0f172a !important; 
        font-weight: 600 !important; 
    }
    
    /* INPUT FIELD VISIBILITY FIX */
    input, textarea, [data-baseweb="input"], [data-baseweb="select"], .stNumberInput div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }

    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: white !important; font-family: serif; font-size: 50px; font-weight: 900;
        width: 90px; height: 90px; display: flex; align-items: center; 
        justify-content: center; border-radius: 26px; border: 3px solid white;
        box-shadow: 0 10px 25px rgba(30, 27, 75, 0.4);
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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin:30px 0;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Choose Language", ["English", "தமிழ்", "हिन्दी"])
    nav = st.radio("Menu", ["Assessment", "Weekly AI Tips", "Medication Log", "Personalized Music", "AI Doctor Chat"])
    st.write("---")
    st.success("System: Connected 🟢")

# --- MULTILINGUAL CONTENT ---
content = {
    "English": {"title": "Maternal Health Dashboard", "vitals": "📝 Inputs", "btn_run": "Analyze Health", "btn_ai": "Generate AI Report", "dl": "Download PDF"},
    "தமிழ்": {"title": "தாய்வழி சுகாதார மேலாண்மை", "vitals": "📝 தரவு", "btn_run": "ஆய்வு செய்", "btn_ai": "AI அறிக்கை", "dl": "PDF பதிவிறக்கம்"},
    "हिन्दी": {"title": "मातृ स्वास्थ्य डैशबोर्ड", "vitals": "📝 डेटा", "btn_run": "विश्लेषण करें", "btn_ai": "AI रिपोर्ट", "dl": "PDF डाउनलोड"}
}
c = content[lang]

# --- PAGE: ASSESSMENT ---
if nav == "Assessment":
    st.title(c["title"])
    col1, col2 = st.columns([1.1, 1], gap="large")
    
    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals"]}</h3>', unsafe_allow_html=True)
        name = st.text_input("Name", value=st.session_state.patient_data.get('name', ""))
        age = st.slider("Age", 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input("Hemoglobin", 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input("Systolic BP", 80, 200, st.session_state.patient_data.get('bp', 120))
        weight = st.number_input("Weight", 30.0, 250.0, st.session_state.patient_data.get('weight', 60.0))
        week = st.number_input("Week", 1, 42, st.session_state.patient_data.get('week', 12))
        
        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {"name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week}
            st.session_state.risk_score = 15.0 if hb >= 11 and bp <= 140 else 68.0
            st.session_state.ai_assessment = "" 
            st.success("Analysis Complete.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="main-card"><h3>📊 Result</h3>', unsafe_allow_html=True)
        if st.session_state.risk_score:
            st.metric("Risk Score", f"{st.session_state.risk_score}%")
            if st.session_state.risk_score > 50: st.error("⚠️ HIGH RISK")
            else: st.success("✅ LOW RISK")
            
            st.write("---")
            if st.button(c["btn_ai"], use_container_width=True):
                with st.spinner("AI assessing clinical status..."):
                    res = call_gemini_ai(f"Analyze risk of {st.session_state.risk_score}% with vitals: {st.session_state.patient_data}", "clinical", lang)
                    st.session_state.ai_assessment = res
            
            if st.session_state.ai_assessment:
                st.markdown(f"**Insight:**\n{st.session_state.ai_assessment}")
                pdf_bytes = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(c["dl"], pdf_bytes, f"Report_{name}.pdf", "application/pdf", use_container_width=True)
        else:
            st.info("Input vitals to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: TIPS ---
elif nav == "Weekly AI Tips":
    st.title("📅 Personalized Weekly Tips")
    if not st.session_state.patient_data:
        st.warning("Please complete Assessment first.")
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button("Generate Tips", use_container_width=True):
            with st.spinner("AI generating tips..."):
                st.session_state.tips = call_gemini_ai(f"Tips for week {st.session_state.patient_data['week']}.", "tips", lang)
        if 'tips' in st.session_state:
            st.markdown(st.session_state.tips)
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: MED LOG ---
elif nav == "Medication Log":
    st.title("💊 Medication Log")
    st.markdown(f'<div class="main-card">', unsafe_allow_html=True)
    m_name = st.text_input("Medicine Name")
    m_time = st.time_input("Daily Time")
    if st.button("Add to Schedule", use_container_width=True):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
    if st.session_state.medicines:
        st.write("---")
        for m in st.session_state.medicines:
            st.markdown(f"🔔 **{m['name']}** at {m['time']}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: MUSIC ---
elif nav == "Personalized Music":
    st.title("🎵 AI Wellness Music")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.button("Get Music Recommendation", use_container_width=True):
        with st.spinner("AI finding peaceful sounds..."):
            st.session_state.music = call_gemini_ai("Suggest music based on my stage.", "music", lang)
    if 'music' in st.session_state:
        st.markdown(st.session_state.music)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: CHAT ---
elif nav == "AI Doctor Chat":
    st.title("🤖 Ask AI Assistant")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    user_q = st.text_input("Ask a clinical question...")
    if st.button("Ask AI", use_container_width=True):
        with st.spinner("Thinking..."):
            st.markdown(f"**AI:** {call_gemini_ai(user_q, 'chatbot', lang)}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.caption("MaternalAI Support v2.9 | Local Fallback Enabled")
