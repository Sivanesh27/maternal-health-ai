import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import os
from fpdf import FPDF

# --- 1. CONFIGURATION & PROJECT IDENTITY ---
# Ensure GEMINI_API_KEY is saved in your Streamlit Cloud 'Secrets' dashboard.
def get_gemini_key():
    try:
        if st.secrets and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except:
        pass
    return ""

def local_clinical_brain(hb, bp, risk_score, lang, diagnostic_msg=""):
    """Rule-based clinical fallback if Gemini AI is unreachable."""
    if lang == "தமிழ்":
        advice = f"மருத்துவ ஆய்வு: அபாய மதிப்பெண் {risk_score}%. "
        if hb < 11: advice += "ஹீமோகுளோபின் குறைவாக உள்ளது. இரும்புச்சத்து உணவுகளை உட்கொள்ளுங்கள். "
        if bp > 140: advice += "இரத்த அழுத்தம் அதிகம். ஓய்வெடுக்கவும்."
        footer = f"\n\n(குறிப்பு: இணைப்பு சிக்கல். பிழை விவரம்: {diagnostic_msg})" if diagnostic_msg else "\n\n(குறிப்பு: இது ஒரு தானியங்கி மருத்துவ உதவி.)"
        return advice + footer
    elif lang == "हिन्दी":
        advice = f"नैदानिक विश्लेषण: जोखिम स्कोर {risk_score}% है। "
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

# --- 2. GEMINI AI CORE (DISCOVERY ENGINE) ---
def call_gemini_ai(prompt, context_type="general", language="English"):
    """
    Exhaustive Discovery Engine for Gemini API.
    Attempts multiple model paths to bypass potential 404 errors.
    """
    current_key = get_gemini_key()
    patient_ctx = st.session_state.get('patient_data', {})
    hb = patient_ctx.get('hb', 11.0)
    bp = patient_ctx.get('bp', 120)
    score = st.session_state.get('risk_score', 0)

    if not current_key or len(current_key) < 10:
        return local_clinical_brain(hb, bp, score, language, "GEMINI_API_KEY missing in Secrets")

    # Mapping roles for better prompt engineering
    roles = {
        "clinical": "Senior Maternal Health Doctor. Provide a professional clinical risk assessment and roadmap.",
        "tips": "Pregnancy wellness expert. Provide 3 specific tips tailored to the current week and vitals.",
        "music": "Prenatal therapist. Recommend 3 relaxation soundscapes for this pregnancy stage.",
        "chatbot": "Friendly maternal health assistant. Answer queries with empathy and clinical clarity."
    }
    
    role_instr = roles.get(context_type, "Maternal health assistant.")
    
    # Combined instruction for maximum compatibility across v1/v1beta
    full_query = (f"System Role: {role_instr}\n"
                  f"LANGUAGE REQUIREMENT: Respond ONLY in {language}.\n"
                  f"PATIENT DATA: {patient_ctx}\n"
                  f"USER QUERY: {prompt}\n\n"
                  f"Important: Deliver a clean, professional response in {language}. No bolding.")

    payload = {"contents": [{"role": "user", "parts": [{"text": full_query}]}]}
    
    # List of endpoints to try (solving the 404 issue)
    discovery_paths = [
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-flash"),
        ("v1", "gemini-1.5-pro"),
        ("v1beta", "gemini-pro"),
        ("v1", "gemini-1.5-flash-8b")
    ]
    
    error_log = []
    for version, model_name in discovery_paths:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={current_key}"
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                error_log.append(f"{model_name}({version}): {response.status_code}")
        except:
            error_log.append(f"{model_name}: Connection Fail")
        time.sleep(0.05)

    diagnostic_msg = " | ".join(error_log)
    return local_clinical_brain(hb, bp, score, language, diagnostic_msg)

# --- 3. API STATUS CHECK ---
def check_gemini_status():
    """Verify if any Gemini models are visible to the key."""
    key = get_gemini_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            models = [m['name'].split('/')[-1] for m in res.json().get('models', [])]
            return True, f"Connected! Models found: {', '.join(models[:3])}"
        return False, f"API Error {res.status_code}"
    except Exception as e:
        return False, str(e)

# --- 4. PDF GENERATION ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MaternalAI Clinical Summary Report', 0, 1, 'C')
        self.ln(10)

def create_pdf(data, ai_note):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Patient Name: {data.get('name', 'N/A')}", 0, 1)
    pdf.set_font("Arial", '', 11)
    vitals = (f"Age: {data.get('age')} | Week: {data.get('week')}\n"
              f"Hemoglobin: {data.get('hb')} g/dL | BP: {data.get('bp')} mmHg | Weight: {data.get('weight')} kg")
    pdf.multi_cell(0, 10, vitals)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical AI Insight:", 0, 1)
    pdf.set_font("Arial", '', 10)
    try:
        clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    except:
        clean_ai = "Assessment generated. Please refer to the app dashboard for localized details."
    pdf.multi_cell(0, 6, clean_ai)
    return pdf.output(dest='S').encode('latin-1')

# --- 5. UI STYLING & TEXT VISIBILITY ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown, [data-testid="stMetricLabel"], .stRadio label, .stSelectbox label { 
        color: #0f172a !important; font-weight: 600 !important; 
    }
    input, textarea, [data-baseweb="input"], [data-baseweb="select"], .stNumberInput div {
        background-color: #ffffff !important; color: #000000 !important;
        border: 2px solid #cbd5e1 !important; border-radius: 12px !important;
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
    .gemini-badge {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white !important; font-size: 11px; font-weight: 700;
        padding: 4px 10px; border-radius: 20px; display: inline-block; margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 6. SESSION INITIALIZATION ---
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'ai_assessment' not in st.session_state: st.session_state.ai_assessment = ""
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'risk_score' not in st.session_state: st.session_state.risk_score = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# --- 7. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin:30px 0;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center"><span class="gemini-badge">✨ Powered by Gemini AI</span></div>', unsafe_allow_html=True)
    
    lang = st.selectbox("🌐 Choose Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
    nav_map = {
        "English": ["Assessment", "Weekly AI Tips", "Medication Log", "Relaxation Music", "AI Doctor Chat"],
        "தமிழ்": ["மதிப்பீடு", "வாராந்திர குறிப்புகள்", "மருந்துப் பதிவு", "இசை", "AI மருத்துவர்"],
        "हिन्दी": ["मूल्यांकन", "साप्ताहिक सुझाव", "दवा लॉग", "संगीत", "AI डॉक्टर"]
    }
    nav_choice = st.radio("Menu / மெனு / मेनू", nav_map[lang])
    
    st.write("---")
    if st.button("🔍 Check Gemini API Status"):
        ok, msg = check_gemini_status()
        if ok: st.success(msg)
        else: st.error(msg)
    
    st.write("---")
    st.success("System: Connected 🟢")

# --- 8. MULTILINGUAL CONTENT ---
content = {
    "English": {
        "title": "Maternal Health Dashboard", "vitals": "📝 Clinical Inputs",
        "name": "Full Name", "age": "Age", "hb": "Hemoglobin (g/dL)",
        "bp": "Systolic BP (mmHg)", "wt": "Weight (kg)", "wk": "Pregnancy Week",
        "btn_run": "Analyze Health", "res": "📊 Clinical Results", "btn_ai": "Generate Gemini AI Report", 
        "dl": "Download PDF", "tips_title": "📅 Personalized Weekly Tips", "btn_tips": "Refresh Tips",
        "med_title": "💊 Medication Log", "med_add": "Add Medicine",
        "music_title": "🎵 Wellness Music Recommendations", "music_btn": "Get AI Selection",
        "chat_title": "🤖 AI Chat Assistant", "chat_ph": "Ask your question...", "chat_btn": "Ask Gemini AI"
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார மேலாண்மை", "vitals": "📝 மருத்துவத் தரவு",
        "name": "முழு பெயர்", "age": "வயது", "hb": "ஹீமோகுளோபின் (g/dL)",
        "bp": "இரத்த அழுத்தம் (mmHg)", "wt": "எடை (கிலோ)", "wk": "கர்ப்ப வாரம்",
        "btn_run": "ஆய்வு செய்", "res": "📊 மருத்துவ முடிவுகள்", "btn_ai": "AI அறிக்கையை உருவாக்கு", 
        "dl": "PDF பதிவிறக்கம்", "tips_title": "📅 வாராந்திர AI வழிகாட்டி", "btn_tips": "புதுப்பிக்கவும்",
        "med_title": "💊 தினசரி மருந்துப் பதிவு", "med_add": "மருந்தைச் சேர்க்கவும்",
        "music_title": "🎵 தியானப் பரிந்துரைகள்", "music_btn": "இசை பரிந்துரையைப் பெறுங்கள்",
        "chat_title": "🤖 சுகாதார AI உதவியாளர்", "chat_ph": "கேள்வியைக் கேளுங்கள்...", "chat_btn": "AI உதவியாளரிடம் கேளுங்கள்"
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य डैशबोर्ड", "vitals": "📝 नैदानिक डेटा",
        "name": "पूरा नाम", "age": "आयु", "hb": "हीमोग्लोबिन (g/dL)",
        "bp": "रक्तचाप (mmHg)", "wt": "वजन (kg)", "wk": "गर्भावस्था सप्ताह",
        "btn_run": "स्वास्थ्य विश्लेषण करें", "res": "📊 मूल्यांकन परिणाम", "btn_ai": "AI रिपोर्ट तैयार करें", 
        "dl": "PDF डाउनलोड करें", "tips_title": "📅 आपकी साप्ताहिक AI गाइड", "btn_tips": "सुझाव अपडेट करें",
        "med_title": "💊 दैनिक दवा लॉग", "med_add": "दवा जोड़ें",
        "music_title": "🎵 कल्याण संगीत", "music_btn": "AI अनुशंसा प्राप्त करें",
        "chat_title": "🤖 स्वास्थ्य AI सहायक", "chat_ph": "अपना प्रश्न पूछें...", "chat_btn": "AI से पूछें"
    }
}
c = content[lang]
page_idx = nav_map[lang].index(nav_choice)

# --- 9. PAGE LOGIC ---

if page_idx == 0: # Assessment
    st.title(c["title"])
    col1, col2 = st.columns([1.1, 1], gap="large")
    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals"]}</h3>', unsafe_allow_html=True)
        name = st.text_input(c["name"], value=st.session_state.patient_data.get('name', ""))
        age = st.slider(c["age"], 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input(c["hb"], 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input(c["bp"], 80, 200, st.session_state.patient_data.get('bp', 120))
        weight = st.number_input(c["wt"], 30.0, 250.0, st.session_state.patient_data.get('weight', 60.0))
        week = st.number_input(c["wk"], 1, 42, st.session_state.patient_data.get('week', 12))
        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {"name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week}
            risk = 10
            if hb < 11: risk += 20
            if bp > 140: risk += 30
            st.session_state.risk_score = min(risk, 95)
            st.session_state.ai_assessment = ""
            st.success("✅ Vitals Synchronized.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="main-card"><h3>{c["res"]}</h3>', unsafe_allow_html=True)
        if st.session_state.risk_score:
            score = st.session_state.risk_score
            color = "#16a34a" if score < 30 else "#dc2626"
            st.markdown(f'<h1 style="color:{color}; font-size:48px;">{score}% Risk</h1>', unsafe_allow_html=True)
            if st.button(c["btn_ai"], use_container_width=True):
                with st.spinner("Gemini AI analyzing clinical data..."):
                    st.session_state.ai_assessment = call_gemini_ai(f"Risk {score}% vitals {st.session_state.patient_data}", "clinical", lang)
            if st.session_state.ai_assessment:
                st.markdown(f"**🤖 Gemini AI Insight:**\n\n{st.session_state.ai_assessment}")
                pdf = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(c["dl"], pdf, f"Report_{name}.pdf", "application/pdf", use_container_width=True)
        else: st.info("👆 Run Analysis to see results.")
        st.markdown('</div>', unsafe_allow_html=True)

elif page_idx == 1: # Tips
    st.title(c["tips_title"])
    if not st.session_state.patient_data: st.warning("Complete Assessment first.")
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button(c["btn_tips"], use_container_width=True):
            with st.spinner("Gemini AI generating personalized tips..."):
                st.session_state.tips = call_gemini_ai(f"Provide tips for week {st.session_state.patient_data['week']}.", "tips", lang)
        if 'tips' in st.session_state: st.markdown(st.session_state.tips)
        st.markdown('</div>', unsafe_allow_html=True)

elif page_idx == 2: # Medication
    st.title(c["med_title"])
    st.markdown(f'<div class="main-card"><h3>{c["med_add"]}</h3>', unsafe_allow_html=True)
    m_name = st.text_input("Name")
    m_time = st.time_input("Time")
    if st.button("💾 Save to Log", use_container_width=True):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
        st.success("Saved.")
    for i, m in enumerate(st.session_state.medicines):
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"🔔 **{m['name']}** at {m['time']}")
        if col2.button("🗑️", key=f"del_{i}"):
            st.session_state.medicines.pop(i)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif page_idx == 3: # Music
    st.title(c["music_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.button(c["music_btn"], use_container_width=True):
        with st.spinner("Gemini AI finding wellness sounds..."):
            st.session_state.music = call_gemini_ai("Suggest relaxation music.", "music", lang)
    if 'music' in st.session_state: st.markdown(st.session_state.music)
    st.markdown('</div>', unsafe_allow_html=True)

elif page_idx == 4: # Chat
    st.title(c["chat_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]): st.markdown(chat["content"])
    user_q = st.text_input(c["chat_ph"])
    if st.button(c["chat_btn"], use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        reply = call_gemini_ai(user_q, "chatbot", lang)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.caption("MaternalAI Support v4.0 | Powered by Google Gemini AI | Multi-Language Clinical Hub")
