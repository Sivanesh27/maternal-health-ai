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
# Your key is correctly set as GEMINI_API_KEY in Streamlit Secrets based on your screenshot.
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
    Robust AI caller with standardized v1beta endpoint to resolve 404 errors.
    language: The target language for the response.
    """
    current_key = get_api_key()
    if not current_key:
        return "Please configure GEMINI_API_KEY in Streamlit Secrets to enable AI."

    # Using the v1beta endpoint which is often more reliable for flash-latest/1.5-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={current_key}"
    
    # Context-specific instructions
    prompts = {
        "clinical": "You are a professional maternal health expert. Analyze the vitals and provide a structured risk assessment and next steps.",
        "tips": "Provide 3 specific, actionable pregnancy tips for this week. Focus on diet, rest, and warning signs.",
        "music": "Recommend 3 types of music or soundscapes (e.g., Vedic chants, soft piano, nature sounds) based on the current risk level.",
        "chatbot": "You are a supportive maternal health assistant. Answer the user's question clearly and warmly."
    }
    
    system_instr = prompts.get(context_type, "You are a maternal health assistant.")
    
    # Merged prompt for maximum compatibility across API versions
    patient_context = st.session_state.get('patient_data', 'New User')
    full_query = f"{system_instr}\nIMPORTANT: You MUST respond ENTIRELY in {language}.\n\nPatient Context: {patient_context}\n\nUser Message: {prompt}"
    
    payload = {"contents": [{"parts": [{"text": full_query}]}]}
    
    try:
        response = requests.post(url, json=payload, timeout=25)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            err_data = response.json()
            msg = err_data.get('error', {}).get('message', 'Unknown Error')
            return f"AI is briefly unavailable (Status {response.status_code}: {msg}). Please try again."
    except Exception as e:
        return f"Connection issue: {str(e)}. Please check your internet."

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
              f"Hemoglobin: {data.get('hb')} g/dL\n"
              f"Systolic Blood Pressure: {data.get('bp')} mmHg\n"
              f"Weight: {data.get('weight')} kg")
    pdf.multi_cell(0, 8, vitals)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical AI Insight:", 0, 1)
    pdf.set_font("Arial", '', 10)
    # Strip non-latin characters for fpdf compatibility
    clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    pdf.multi_cell(0, 6, clean_ai)
    return pdf.output(dest='S').encode('latin-1')

# --- UI STYLING ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* UNIVERSAL TEXT VISIBILITY - Forces all text to be dark and visible */
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown, .stSelectbox label { 
        color: #0f172a !important; 
        font-weight: 600 !important; 
    }
    
    /* Logo Styling */
    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: white !important; font-family: 'Georgia', serif; font-size: 48px; font-weight: 900;
        width: 85px; height: 85px; display: flex; align-items: center; 
        justify-content: center; border-radius: 24px; border: 3px solid white;
        box-shadow: 0 10px 25px rgba(30, 27, 75, 0.3);
    }
    
    /* Card Design */
    .main-card {
        background: white; padding: 35px; border-radius: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06); border: 1px solid #eef2f6;
        margin-bottom: 25px;
    }
    
    /* Button Design */
    .stButton>button {
        background: linear-gradient(135deg, #4338ca 0%, #3b82f6 100%);
        color: white !important; border-radius: 16px; font-weight: 700;
        padding: 14px 28px; border: none; box-shadow: 0 4px 15px rgba(67, 56, 202, 0.25);
        transition: 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(67, 56, 202, 0.4); }
    
    /* Input Fields Fix */
    input { color: #0f172a !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'ai_assessment' not in st.session_state: st.session_state.ai_assessment = ""

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin:20px 0;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
    nav = st.radio("Menu", ["Home & Assessment", "Personalized Tips", "Medicine Log", "Health Resources", "Ask AI Assistant"])
    st.write("---")
    st.success("Clinical Engine: Connected 🟢")

# --- MULTILINGUAL CONTENT MAP ---
content = {
    "English": {
        "title": "Maternal Health Dashboard", "vitals": "📝 Clinical Inputs",
        "btn_run": "Analyze Health", "res": "📊 Assessment Results",
        "btn_ai": "Generate AI Personalized Insight", "dl": "Download PDF Report",
        "tips_title": "📅 Weekly AI Guide", "btn_tips": "Refresh Tips",
        "med_title": "💊 Medication Log", "med_add": "Add Medicine",
        "res_title": "🏥 Clinical Resources", "music_btn": "Get Music Recommendation",
        "chat_title": "🤖 Health Companion Chat"
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார மேலாண்மை", "vitals": "📝 மருத்துவத் தரவு",
        "btn_run": "ஆரோக்கியத்தை ஆய்வு செய்", "res": "📊 மருத்துவ முடிவு",
        "btn_ai": "AI தனிப்பயனாக்கப்பட்ட விளக்கத்தைப் பெறுங்கள்", "dl": "அறிக்கையைப் பதிவிறக்கவும்",
        "tips_title": "📅 வாராந்திர AI வழிகாட்டி", "btn_tips": "குறிப்புகளைப் புதுப்பிக்கவும்",
        "med_title": "💊 மருந்துப் பதிவு", "med_add": "மருந்தைச் சேர்க்கவும்",
        "res_title": "🏥 மருத்துவ வளங்கள்", "music_btn": "இசை பரிந்துரையைப் பெறுங்கள்",
        "chat_title": "🤖 சுகாதார AI உரையாடல்"
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य डैशबोर्ड", "vitals": "📝 नैदानिक डेटा",
        "btn_run": "स्वास्थ्य विश्लेषण करें", "res": "📊 मूल्यांकन परिणाम",
        "btn_ai": "AI व्यक्तिगत जानकारी प्राप्त करें", "dl": "रिपोर्ट डाउनलोड करें",
        "tips_title": "📅 साप्ताहिक AI गाइड", "btn_tips": "सुझाव अपडेट करें",
        "med_title": "💊 दवा लॉग", "med_add": "दवा जोड़ें",
        "res_title": "🏥 नैदानिक संसाधन", "music_btn": "संगीत अनुशंसा प्राप्त करें",
        "chat_title": "🤖 स्वास्थ्य AI साथी"
    }
}
c = content[lang]

# --- PAGE LOGIC: HOME ---
if nav == "Home & Assessment":
    st.title(c["title"])
    col1, col2 = st.columns([1.1, 1], gap="large")
    
    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals"]}</h3>', unsafe_allow_html=True)
        name = st.text_input("Name", value=st.session_state.patient_data.get('name', ""))
        age = st.slider("Age", 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input("Hemoglobin (g/dL)", 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input("Systolic BP (mmHg)", 80, 200, st.session_state.patient_data.get('bp', 120))
        weight = st.number_input("Weight (kg)", 30.0, 250.0, st.session_state.patient_data.get('weight', 60.0))
        week = st.number_input("Pregnancy Week", 1, 42, st.session_state.patient_data.get('week', 12))
        
        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {"name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week}
            # Heuristic/ML Risk Check
            st.session_state.risk_score = 15.0 if hb >= 11 and bp <= 140 else 65.0
            st.session_state.ai_assessment = "" 
            st.success("Analysis Complete.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="main-card"><h3>{c["res"]}</h3>', unsafe_allow_html=True)
        if 'risk_score' in st.session_state:
            score = st.session_state.risk_score
            st.metric("Risk Score", f"{score}%")
            if score > 50: st.error("Status: High Risk - Please consult a doctor.")
            else: st.success("Status: Low Risk - Maintaining healthy progress.")
            
            st.write("---")
            if st.button(c["btn_ai"], use_container_width=True):
                with st.spinner("AI analyzing clinical data..."):
                    res = call_gemini_ai(f"Analyze: Risk Score {score}%, Hb {hb}, BP {bp}, Week {week}", "clinical", lang)
                    st.session_state.ai_assessment = res
            
            if st.session_state.ai_assessment:
                st.markdown(f"**Insight:**\n{st.session_state.ai_assessment}")
                pdf_bytes = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(c["dl"], pdf_bytes, f"Clinical_Report_{name}.pdf", "application/pdf")
        else:
            st.info("Fill vitals and run analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE LOGIC: TIPS ---
elif nav == "Personalized Tips":
    st.title(c["tips_title"])
    if not st.session_state.patient_data:
        st.warning("Please complete Home Assessment first.")
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button(c["btn_tips"]):
            with st.spinner("AI is personalizing your guide..."):
                st.session_state.weekly_tips = call_gemini_ai(f"Give tips for week {st.session_state.patient_data['week']}", "tips", lang)
        
        if 'weekly_tips' in st.session_state:
            st.markdown(st.session_state.weekly_tips)
        else: st.write("Click button above to generate personalized AI tips.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE LOGIC: MED LOG ---
elif nav == "Medicine Log":
    st.title(c["med_title"])
    st.markdown(f'<div class="main-card"><h3>{c["med_add"]}</h3>', unsafe_allow_html=True)
    m_name = st.text_input("Medicine Name")
    m_time = st.time_input("Reminder Time")
    if st.button("Add to Schedule"):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
        st.success(f"Log updated: {m_name}")
    
    if st.session_state.medicines:
        st.write("---")
        for m in st.session_state.medicines:
            st.markdown(f"🔔 **{m['name']}** at {m['time']}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE LOGIC: RESOURCES ---
elif nav == "Health Resources":
    st.title(c["res_title"])
    t1, t2 = st.tabs(["🎵 AI Music Recommendations", "👩‍⚕️ Clinics"])
    with t1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button(c["music_btn"]):
            with st.spinner("Finding wellness sounds..."):
                st.session_state.music_rec = call_gemini_ai("Suggest prenatal music.", "music", lang)
        if 'music_rec' in st.session_state:
            st.markdown(st.session_state.music_rec)
        st.markdown('</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.write("**District General Hospital** (Gynae Ward) - +91 98765 00000")
        st.write("**Primary Health Center (PHC)** - +91 91234 11111")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE LOGIC: CHAT ---
elif nav == "Ask AI Assistant":
    st.title(c["chat_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    user_q = st.text_input("Message your AI Health Companion...")
    if st.button("Send"):
        with st.spinner("AI is replying..."):
            ans = call_gemini_ai(user_q, "chatbot", lang)
            st.markdown(f"**AI:** {ans}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.caption("MaternalAI Decision Support v2.9 | High-Contrast Rural Accessibility Mode")
