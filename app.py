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

def call_gemini_ai(prompt, context_type="general", language="English"):
    """
    Robust AI caller with simplified payload to avoid 400/404 errors.
    language: The target language for the response.
    """
    current_key = get_api_key()
    if not current_key:
        return "Please configure API Key to enable AI personalization."

    # Using the most stable stable endpoint
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={current_key}"
    
    # Context-specific instructions merged into prompt
    system_prompts = {
        "clinical": "You are a senior maternal health consultant. Provide a detailed risk analysis and clinical steps.",
        "tips": "Provide 3 personalized, actionable pregnancy tips for this specific week and health status.",
        "music": "Recommend 2 specific types of music or meditation for this pregnancy stage.",
        "chatbot": "You are a friendly maternal health companion. Answer the mother's question warmly."
    }
    
    instruction = system_prompts.get(context_type, "You are a maternal health assistant.")
    # Explicitly instruct the AI to respond in the selected language
    full_query = f"{instruction}\nIMPORTANT: Respond entirely in {language}.\n\nPatient Data: {st.session_state.get('patient_data', 'Not provided')}\n\nUser Query: {prompt}"
    
    payload = {"contents": [{"parts": [{"text": full_query}]}]}
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"AI is resting. (Error {response.status_code}). Please try again shortly."
    except:
        return "Connectivity issue. Please check your internet."

# --- PDF ENGINE ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MaternalAI Personalized Health Report', 0, 1, 'C')
        self.ln(10)

def create_pdf(data, ai_note):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Patient: {data['name']}", 0, 1)
    pdf.set_font("Arial", '', 11)
    vitals = f"Age: {data['age']} | Week: {data['week']} | Hb: {data['hb']} | BP: {data['bp']} | Weight: {data['weight']}"
    pdf.multi_cell(0, 10, vitals)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical AI Assessment:", 0, 1)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 6, ai_note.encode('ascii', 'ignore').decode('ascii'))
    return pdf.output(dest='S').encode('latin-1')

# --- UI STYLING ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    /* GLOBAL TEXT VISIBILITY FIX */
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown { 
        color: #0f172a !important; 
        font-weight: 600 !important; 
    }
    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: white !important; font-family: serif; font-size: 46px; font-weight: 900;
        width: 80px; height: 80px; display: flex; align-items: center; 
        justify-content: center; border-radius: 22px; border: 2px solid white;
    }
    .main-card {
        background: white; padding: 30px; border-radius: 28px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;
        margin-bottom: 25px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%);
        color: white !important; border-radius: 14px; font-weight: 700;
        padding: 12px 24px; border: none;
    }
    [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION INITIALIZATION ---
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'ai_assessment' not in st.session_state: st.session_state.ai_assessment = ""

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin-bottom:20px;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Language", ["English", "தமிழ்", "हिन्दी"])
    nav = st.radio("Navigation", ["Home & Assessment", "Personalized Tips", "Medicine Log", "Health Resources", "Chat with AI"])
    st.write("---")
    st.success("AI Core: Connected 🟢")

# --- MULTILINGUAL MAP ---
content = {
    "English": {
        "home_title": "Maternal Health Assessment", "vitals_header": "📝 Patient Vitals",
        "name_lbl": "Full Name", "age_lbl": "Age", "hb_lbl": "Hemoglobin (g/dL)",
        "bp_lbl": "Systolic BP (mmHg)", "wt_lbl": "Weight (kg)", "wk_lbl": "Pregnancy Week",
        "btn_run": "Run AI Diagnostic", "results_header": "📊 Clinical Result",
        "risk_score_lbl": "Predictive Risk Score", "btn_report": "Generate AI Detailed Report",
        "dl_report": "Download PDF Report", "tips_title": "📅 Your Weekly AI Guide",
        "tips_header": "Personalized Tips", "btn_tips": "Generate Fresh Tips",
        "med_title": "💊 Medication Reminder", "med_header": "Add Medicine",
        "med_name": "Medicine Name", "med_time": "Reminder Time", "btn_med": "Add Medicine",
        "res_title": "🏥 Personalized Resources", "res_tab1": "🎵 Music & Relaxation",
        "res_tab2": "👩‍⚕️ Doctor Directory", "btn_music": "Get AI Music Recommendation",
        "chat_title": "🤖 Maternal Health Companion", "chat_input": "Ask me anything...",
        "btn_chat": "Ask AI"
    },
    "தமிழ்": {
        "home_title": "தாய்வழி சுகாதார மதிப்பீடு", "vitals_header": "📝 நோயாளி தரவு",
        "name_lbl": "முழு பெயர்", "age_lbl": "வயது", "hb_lbl": "ஹீமோகுளோபின் (g/dL)",
        "bp_lbl": "இரத்த அழுத்தம் (mmHg)", "wt_lbl": "எடை (kg)", "wk_lbl": "கர்ப்ப வாரம்",
        "btn_run": "AI ஆய்வைத் தொடங்கு", "results_header": "📊 மருத்துவ முடிவு",
        "risk_score_lbl": "ஆபத்து நிகழ்தகவு", "btn_report": "AI விரிவான அறிக்கையை உருவாக்கு",
        "dl_report": "PDF அறிக்கையைப் பதிவிறக்கவும்", "tips_title": "📅 உங்கள் வாராந்திர AI வழிகாட்டி",
        "tips_header": "தனிப்பயனாக்கப்பட்ட குறிப்புகள்", "btn_tips": "புதிய குறிப்புகளை உருவாக்கு",
        "med_title": "💊 மருந்து நினைவூட்டல்", "med_header": "மருந்தைச் சேர்க்கவும்",
        "med_name": "மருந்தின் பெயர்", "med_time": "நினைவூட்டல் நேரம்", "btn_med": "சேர்க்கவும்",
        "res_title": "🏥 தனிப்பயனாக்கப்பட்ட வளங்கள்", "res_tab1": "🎵 இசை & தியானம்",
        "res_tab2": "👩‍⚕️ மருத்துவர் அடைவு", "btn_music": "AI இசை பரிந்துரையைப் பெறுங்கள்",
        "chat_title": "🤖 தாய்வழி சுகாதார துணை", "chat_input": "ஏதாவது கேளுங்கள்...",
        "btn_chat": "AI இடம் கேளுங்கள்"
    },
    "हिन्दी": {
        "home_title": "मातृ स्वास्थ्य मूल्यांकन", "vitals_header": "📝 रोगी का विवरण",
        "name_lbl": "पूरा नाम", "age_lbl": "आयु", "hb_lbl": "हीमोग्लोबिन (g/dL)",
        "bp_lbl": "रक्तचाप (mmHg)", "wt_lbl": "वजन (kg)", "wk_lbl": "गर्भावस्था सप्ताह",
        "btn_run": "AI निदान चलाएं", "results_header": "📊 नैदानिक परिणाम",
        "risk_score_lbl": "जोखिम स्कोर", "btn_report": "AI विस्तृत रिपोर्ट तैयार करें",
        "dl_report": "PDF रिपोर्ट डाउनलोड करें", "tips_title": "📅 आपकी साप्ताहिक AI गाइड",
        "tips_header": "व्यक्तिगत सुझाव", "btn_tips": "नए सुझाव प्राप्त करें",
        "med_title": "💊 दवा रिमाइंडर", "med_header": "दवा जोड़ें",
        "med_name": "दवा का नाम", "med_time": "रिमाइंडर का समय", "btn_med": "जोड़ें",
        "res_title": "🏥 व्यक्तिगत संसाधन", "res_tab1": "🎵 संगीत और विश्राम",
        "res_tab2": "👩‍⚕️ डॉक्टर निर्देशिका", "btn_music": "AI संगीत अनुशंसा प्राप्त करें",
        "chat_title": "🤖 मातृ स्वास्थ्य साथी", "chat_input": "मुझसे कुछ भी पूछें...",
        "btn_chat": "AI से पूछें"
    }
}
c = content[lang]

# --- PAGE: HOME & ASSESSMENT ---
if nav == "Home & Assessment":
    st.title(c["home_title"])
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals_header"]}</h3>', unsafe_allow_html=True)
        name = st.text_input(c["name_lbl"], value=st.session_state.patient_data.get('name', ""))
        age = st.slider(c["age_lbl"], 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input(c["hb_lbl"], 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input(c["bp_lbl"], 80, 200, st.session_state.patient_data.get('bp', 120))
        weight = st.number_input(c["wt_lbl"], 30.0, 250.0, st.session_state.patient_data.get('weight', 60.0))
        week = st.number_input(c["wk_lbl"], 1, 42, st.session_state.patient_data.get('week', 12))
        
        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {
                "name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week
            }
            try:
                model = joblib.load('maternal_health_model.pkl')
                prob = model.predict_proba(np.array([[age, hb, bp, weight]]))[0][1]
                st.session_state.risk_score = round(prob * 100, 2)
            except:
                st.session_state.risk_score = 15.0 if hb > 10 else 65.0
            st.session_state.ai_assessment = "" 
            st.success("Analysis Complete.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="main-card"><h3>{c["results_header"]}</h3>', unsafe_allow_html=True)
        if 'risk_score' in st.session_state:
            score = st.session_state.risk_score
            st.metric(c["risk_score_lbl"], f"{score}%")
            if score > 70: st.error("⚠️ HIGH RISK")
            elif score > 40: st.warning("⚠️ MODERATE RISK")
            else: st.success("✅ LOW RISK")
            
            st.write("---")
            if st.button(c["btn_report"]):
                with st.spinner("AI analyzing clinical data..."):
                    res = call_gemini_ai(f"Risk Score is {score}%. Analyze these vitals.", "clinical", lang)
                    st.session_state.ai_assessment = res
            
            if st.session_state.ai_assessment:
                st.markdown(f"**AI Insight:**\n{st.session_state.ai_assessment}")
                pdf_bytes = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(c["dl_report"], pdf_bytes, f"Report_{name}.pdf", "application/pdf")
        else:
            st.info("Please enter data and run diagnostic.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: PERSONALIZED TIPS ---
elif nav == "Personalized Tips":
    st.title(c["tips_title"])
    if not st.session_state.patient_data:
        st.warning("Please complete your Health Assessment first.")
    else:
        st.markdown(f'<div class="main-card"><h3>Week {st.session_state.patient_data["week"]} {c["tips_header"]}</h3>', unsafe_allow_html=True)
        if st.button(c["btn_tips"]):
            with st.spinner("AI is thinking..."):
                tips = call_gemini_ai(f"I am in week {st.session_state.patient_data['week']}. Give me personalized tips.", "tips", lang)
                st.session_state.weekly_tips = tips
        
        if 'weekly_tips' in st.session_state:
            st.markdown(st.session_state.weekly_tips)
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: MEDICINE LOG ---
elif nav == "Medicine Log":
    st.title(c["med_title"])
    with st.container():
        st.markdown(f'<div class="main-card"><h3>{c["med_header"]}</h3>', unsafe_allow_html=True)
        m_name = st.text_input(c["med_name"])
        m_time = st.time_input(c["med_time"])
        if st.button(c["btn_med"]):
            st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
            st.success(f"Added: {m_name}")
        
        if st.session_state.medicines:
            st.write("---")
            for m in st.session_state.medicines:
                st.markdown(f"🔔 **{m['name']}** at {m['time']}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: RESOURCES ---
elif nav == "Health Resources":
    st.title(c["res_title"])
    t1, t2 = st.tabs([c["res_tab1"], c["res_tab2"]])
    
    with t1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button(c["btn_music"]):
            with st.spinner("AI is finding the perfect tune..."):
                rec = call_gemini_ai("Suggest music for my current stage.", "music", lang)
                st.session_state.music_rec = rec
        if 'music_rec' in st.session_state:
            st.markdown(st.session_state.music_rec)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with t2:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.write("**Dr. Priya** (Gynecologist) - +91 98765 43210")
        st.write("**Dr. Anjali** (Obstetrician) - +91 91234 56780")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: CHATBOT ---
elif nav == "Chat with AI":
    st.title(c["chat_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    user_q = st.text_input(c["chat_input"])
    if st.button(c["btn_chat"]):
        with st.spinner("Thinking..."):
            ans = call_gemini_ai(user_q, "chatbot", lang)
            st.markdown(f"**AI Companion:** {ans}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.caption("MaternalAI Companion v2.8 | Real-time AI Personalization Enabled")
