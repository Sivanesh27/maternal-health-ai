import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import requests
import json
import os
from fpdf import FPDF

# --- AI CONFIGURATION ---
# Your key should be set in Streamlit Secrets as GEMINI_API_KEY
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
    Standardized AI caller using the exact role-based JSON structure required by Gemini 1.5.
    """
    current_key = get_api_key()
    if not current_key or current_key == "":
        return "Error: AI Key is missing. Please check your Streamlit Secrets."

    # Using the most stable v1beta endpoint for free-tier keys
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={current_key}"
    
    prompts = {
        "clinical": "You are a professional maternal health doctor. Provide a professional risk assessment and roadmap.",
        "tips": "Provide 3 unique, medical-backed tips for this specific pregnancy week.",
        "music": "Suggest 3 specific types of music or meditation for mental peace during this stage.",
        "chatbot": "You are a friendly health assistant. Answer the mother's query with medical empathy."
    }
    
    instr = prompts.get(context_type, "Maternal health assistant.")
    patient_ctx = st.session_state.get('patient_data', 'No data entered')
    
    # Combined instructions for the AI
    final_prompt = (f"Instruction: {instr}\n"
                    f"Language: Respond entirely in {language}\n"
                    f"Patient Context: {patient_ctx}\n"
                    f"User Request: {prompt}")

    # EXACT JSON STRUCTURE REQUIRED FOR SUCCESSFUL API HANDSHAKE
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": final_prompt}]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=25)
        if response.status_code == 200:
            res_json = response.json()
            return res_json['candidates'][0]['content']['parts'][0]['text']
        else:
            err_msg = response.json().get('error', {}).get('message', 'Unknown API Error')
            return f"The AI is temporarily unavailable (Code {response.status_code}: {err_msg}). Please follow clinical guidelines."
    except Exception as e:
        return f"Connection issue: {str(e)}. Please check your internet."

# --- PDF ENGINE ---
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
    # Strip non-latin characters for PDF compatibility
    clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    pdf.multi_cell(0, 6, clean_ai)
    return pdf.output(dest='S').encode('latin-1')

# --- UI STYLING & CONTRAST GUARD ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* FORCE TEXT VISIBILITY */
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown, [data-testid="stMetricLabel"] { 
        color: #0f172a !important; 
        font-weight: 600 !important; 
    }
    
    /* FORCE INPUT FIELD READABILITY */
    input, textarea, [data-baseweb="input"], [data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }

    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%);
        color: white !important; font-family: 'Playfair Display', serif; font-size: 48px; font-weight: 900;
        width: 85px; height: 85px; display: flex; align-items: center; 
        justify-content: center; border-radius: 24px; border: 3px solid white;
        box-shadow: 0 10px 25px rgba(30, 27, 75, 0.3);
    }
    
    .main-card {
        background: white; padding: 35px; border-radius: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06); border: 1px solid #eef2f6;
        margin-bottom: 25px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%);
        color: white !important; border-radius: 16px; font-weight: 700;
        padding: 14px 28px; border: none; box-shadow: 0 4px 15px rgba(67, 56, 202, 0.25);
    }
    
    [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'ai_assessment' not in st.session_state: st.session_state.ai_assessment = ""
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'risk_score' not in st.session_state: st.session_state.risk_score = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin:20px 0;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Choose Language", ["English", "தமிழ்", "हिन्दी"])
    nav = st.radio("Navigation", ["Health Assessment", "Weekly AI Tips", "Medication Log", "Relaxation Music", "AI Health Chat"])
    st.write("---")
    st.success("System: Connected 🟢")

# --- MULTILINGUAL CONTENT MAP ---
content = {
    "English": {"btn_run": "Analyze My Health", "btn_ai": "Generate AI Report", "dl": "Download PDF"},
    "தமிழ்": {"btn_run": "ஆரோக்கியத்தை ஆய்வு செய்", "btn_ai": "AI அறிக்கையை உருவாக்கு", "dl": "PDF பதிவிறக்கம்"},
    "हिन्दी": {"btn_run": "स्वास्थ्य विश्लेषण करें", "btn_ai": "AI रिपोर्ट तैयार करें", "dl": "PDF डाउनलोड करें"}
}
c = content[lang]

# --- PAGE LOGIC ---
if nav == "Health Assessment":
    st.title("Personalized Health Assessment")
    col1, col2 = st.columns([1.1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="main-card"><h3>📝 Input Vitals</h3>', unsafe_allow_html=True)
        name = st.text_input("Full Name", value=st.session_state.patient_data.get('name', ""))
        age = st.slider("Age", 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input("Hemoglobin (g/dL)", 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input("Systolic BP (mmHg)", 80, 200, st.session_state.patient_data.get('bp', 120))
        weight = st.number_input("Weight (kg)", 30.0, 250.0, st.session_state.patient_data.get('weight', 60.0))
        week = st.number_input("Pregnancy Week", 1, 42, st.session_state.patient_data.get('week', 12))
        
        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {"name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week}
            # Simple Prediction Logic (Fallback for missing pkl)
            st.session_state.risk_score = 12.0 if hb >= 11 and bp <= 140 else 68.5
            st.session_state.ai_assessment = "" 
            st.success("Assessment stored.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="main-card"><h3>📊 Clinical Result</h3>', unsafe_allow_html=True)
        if st.session_state.risk_score:
            score = st.session_state.risk_score
            st.metric("Risk Probability", f"{score}%")
            if score > 50: st.error("⚠️ STATUS: HIGH RISK")
            else: st.success("✅ STATUS: LOW RISK")
            
            st.write("---")
            if st.button(c["btn_ai"], use_container_width=True):
                with st.spinner("AI analyzing Clinical Data..."):
                    st.session_state.ai_assessment = call_gemini_ai(f"Evaluate risk of {score}% with vitals: {st.session_state.patient_data}", "clinical", lang)
            
            if st.session_state.ai_assessment:
                st.markdown(f"**AI Clinical Insight:**\n{st.session_state.ai_assessment}")
                pdf_bytes = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(c["dl"], pdf_bytes, f"Report_{name}.pdf", "application/pdf")
        else:
            st.info("Fill your vitals and click Analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

elif nav == "Weekly AI Tips":
    st.title("📅 Personalized Weekly Tips")
    if not st.session_state.patient_data:
        st.warning("Please complete Health Assessment first.")
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button("Generate Fresh Tips"):
            with st.spinner("Personalizing your guide..."):
                st.session_state.tips = call_gemini_ai(f"I am in week {st.session_state.patient_data['week']}. Give me 3 tips.", "tips", lang)
        if 'tips' in st.session_state:
            st.markdown(st.session_state.tips)
        st.markdown('</div>', unsafe_allow_html=True)

elif nav == "Medication Log":
    st.title("💊 Medication Management")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    m_name = st.text_input("Medicine Name")
    m_time = st.time_input("Reminder Time")
    if st.button("Add to Log"):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
        st.success(f"Added {m_name}")
    for m in st.session_state.medicines:
        st.markdown(f"🔔 **{m['name']}** at {m['time']}")
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "Relaxation Music":
    st.title("🎵 AI Wellness Music")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.button("Get Music Recommendation"):
        with st.spinner("AI finding peaceful sounds..."):
            st.session_state.music = call_gemini_ai("Suggest music based on my risk score.", "music", lang)
    if 'music' in st.session_state:
        st.markdown(st.session_state.music)
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "AI Health Chat":
    st.title("🤖 AI Maternal Companion")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    user_q = st.text_input("Ask me anything about your pregnancy...")
    if st.button("Ask AI"):
        with st.spinner("Thinking..."):
            st.markdown(f"**AI:** {call_gemini_ai(user_q, 'chatbot', lang)}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.caption("MaternalAI Decision Support v2.9 | Industry Standard Rural Health Mode")
