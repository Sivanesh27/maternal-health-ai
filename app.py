import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import os
from fpdf import FPDF

# --- 1. CONFIGURATION & PROJECT IDENTITY ---
def get_secure_key():
    try:
        if st.secrets and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except:
        pass
    return ""

def get_local_fallback(hb, bp, risk_score, lang):
    """Rule-based medical logic used if the cloud AI is unreachable."""
    if lang == "தமிழ்":
        advice = f"மருத்துவ ஆய்வு: உங்கள் அபாய மதிப்பெண் {risk_score}%. "
        if hb < 11: advice += "ஹீமோகுளோபின் குறைவாக உள்ளது. இரும்புச்சத்து உணவுகளை உட்கொள்ளுங்கள். "
        if bp > 140: advice += "இரத்த அழுத்தம் அதிகம். ஓய்வெடுக்கவும்."
        return advice + "\n\n(குறிப்பு: இது ஒரு தானியங்கி மருத்துவ உதவி.)"
    elif lang == "हिन्दी":
        advice = f"नैदानिक विश्लेषण: आपका जोखिम स्कोर {risk_score}% है। "
        if hb < 11: advice += "हीमोग्लोबिन कम है। आयरन युक्त भोजन लें। "
        if bp > 140: advice += "रक्तचाप अधिक है। आराम करें।"
        return advice + "\n\n(नोट: यह एक स्वचालित सहायता है।)"
    else:
        advice = f"Clinical Assessment: Your predictive risk is {risk_score}%. "
        if hb < 11: advice += "Action: Increase Iron intake. "
        if bp > 140: advice += "Action: Monitor BP daily and reduce salt. "
        return advice + "\n\n(Note: Rule-based fallback generated locally.)"

# --- 2. MULTILINGUAL CONTENT MAP (TOTAL COVERAGE) ---
content = {
    "English": {
        "title": "MaternalAI Health Dashboard",
        "nav_header": "Main Menu",
        "nav_options": ["Assessment", "Weekly Tips", "Medication Log", "Relaxation Music", "AI Doctor Chat"],
        "vitals_header": "📝 Clinical Input Parameters",
        "name_lbl": "Full Name",
        "age_lbl": "Age",
        "hb_lbl": "Hemoglobin (g/dL)",
        "bp_lbl": "Systolic BP (mmHg)",
        "wt_lbl": "Weight (kg)",
        "wk_lbl": "Pregnancy Week",
        "btn_analyze": "Analyze Health Data",
        "res_header": "📊 Clinical Results",
        "risk_lbl": "Risk Score",
        "btn_report": "Generate AI Clinical Report",
        "dl_btn": "Download Medical PDF",
        "tips_header": "📅 Personalized Weekly Guidance",
        "btn_tips": "Generate New Advice",
        "med_header": "💊 Daily Medication Log",
        "med_name": "Medicine Name",
        "med_time": "Dosage Time",
        "med_save": "Save to Log",
        "med_curr": "📋 Current Schedule",
        "music_header": "🎵 Wellness Soundscapes",
        "music_btn": "Get AI Music Suggestion",
        "chat_header": "🤖 Health Assistant Chat",
        "chat_ph": "Type your health query...",
        "chat_btn": "Ask Assistant",
        "status_connected": "System: Online",
        "sync_success": "✅ Vitals Synchronized.",
        "complete_first": "⚠️ Please complete the Assessment first.",
        "low_risk": "✅ Low Risk — Healthy profile",
        "mod_risk": "⚠️ Moderate Risk — Monitor closely",
        "high_risk": "🚨 High Risk — Consult doctor",
        "ai_insight_header": "📄 Detailed Clinical Report",
        "powered_by": "Powered by MaternalAI Support",
        "clear_chat": "🗑️ Clear Chat History",
        "save_success": "✅ Saved successfully.",
        "delete_btn": "Delete"
    },
    "தமிழ்": {
        "title": "MaternalAI சுகாதார மேலாண்மை",
        "nav_header": "முதன்மை மெனு",
        "nav_options": ["மதிப்பீடு", "வாராந்திர குறிப்புகள்", "மருந்துப் பதிவு", "இசை", "AI மருத்துவர்"],
        "vitals_header": "📝 மருத்துவத் தரவு உள்ளீடு",
        "name_lbl": "முழு பெயர்",
        "age_lbl": "வயது",
        "hb_lbl": "ஹீமோகுளோபின் (g/dL)",
        "bp_lbl": "இரத்த அழுத்தம் (mmHg)",
        "wt_lbl": "எடை (கிலோ)",
        "wk_lbl": "கர்ப்ப வாரம்",
        "btn_analyze": "ஆரோக்கியத்தை ஆய்வு செய்",
        "res_header": "📊 மருத்துவ முடிவுகள்",
        "risk_lbl": "அபாய மதிப்பெண்",
        "btn_report": "AI மருத்துவ அறிக்கையை உருவாக்கு",
        "dl_btn": "மருத்துவ PDF பதிவிறக்கம்",
        "tips_header": "📅 வாராந்திர தனிப்பயனாக்கப்பட்ட வழிகாட்டி",
        "btn_tips": "புதிய ஆலோசனையைப் பெறு",
        "med_header": "💊 தினசரி மருந்துப் பதிவு",
        "med_name": "மருந்து பெயர்",
        "med_time": "மருந்து நேரம்",
        "med_save": "பதிவேட்டில் சேமிக்கவும்",
        "med_curr": "📋 தற்போதைய அட்டவணை",
        "music_header": "🎵 ஆரோக்கிய இசை பரிந்துரைகள்",
        "music_btn": "இசை பரிந்துரையைப் பெறுங்கள்",
        "chat_header": "🤖 சுகாதார உதவியாளர் அரட்டை",
        "chat_ph": "உங்கள் கேள்வியைத் தட்டச்சு செய்க...",
        "chat_btn": "உதவியாளரிடம் கேளுங்கள்",
        "status_connected": "கணினி: இணைக்கப்பட்டுள்ளது",
        "sync_success": "✅ தரவுகள் ஒத்திசைக்கப்பட்டன.",
        "complete_first": "⚠️ தயவுசெய்து முதலில் மதிப்பீட்டை முடிக்கவும்.",
        "low_risk": "✅ குறைந்த ஆபத்து - ஆரோக்கியமான நிலை",
        "mod_risk": "⚠️ நடுத்தர ஆபத்து - கண்காணிக்கவும்",
        "high_risk": "🚨 அதிக ஆபத்து - மருத்துவரை அணுகவும்",
        "ai_insight_header": "📄 விரிவான மருத்துவ அறிக்கை",
        "powered_by": "MaternalAI ஆதரவு மூலம் வழங்கப்படுகிறது",
        "clear_chat": "🗑️ உரையாடலை அழி",
        "save_success": "✅ வெற்றிகரமாக சேமிக்கப்பட்டது.",
        "delete_btn": "நீக்கு"
    },
    "हिन्दी": {
        "title": "MaternalAI स्वास्थ्य डैशबोर्ड",
        "nav_header": "मुख्य मेनू",
        "nav_options": ["मूल्यांकन", "साप्ताहिक सुझाव", "दवा लॉग", "संगीत", "AI डॉक्टर"],
        "vitals_header": "📝 नैदानिक डेटा इनपुट",
        "name_lbl": "पूरा नाम",
        "age_lbl": "आयु",
        "hb_lbl": "हीमोग्लोबिन (g/dL)",
        "bp_lbl": "रक्तचाप (mmHg)",
        "wt_lbl": "वजन (kg)",
        "wk_lbl": "गर्भावस्था सप्ताह",
        "btn_analyze": "स्वास्थ्य विश्लेषण करें",
        "res_header": "📊 मूल्यांकन परिणाम",
        "risk_lbl": "जोखिम स्कोर",
        "btn_report": "AI मेडिकल रिपोर्ट तैयार करें",
        "dl_btn": "PDF डाउनलोड करें",
        "tips_header": "📅 साप्ताहिक व्यक्तिगत सुझाव",
        "btn_tips": "नए सुझाव प्राप्त करें",
        "med_header": "💊 दैनिक दवा लॉग",
        "med_name": "दवा का नाम",
        "med_time": "दवा का समय",
        "med_save": "लॉग में सहेजें",
        "med_curr": "📋 वर्तमान समय सारिणी",
        "music_header": "🎵 कल्याण संगीत अनुशंसा",
        "music_btn": "AI संगीत सुझाव प्राप्त करें",
        "chat_header": "🤖 स्वास्थ्य सहायक चैट",
        "chat_ph": "अपना प्रश्न टाइप करें...",
        "chat_btn": "सहायक से पूछें",
        "status_connected": "सिस्टम: सक्रिय",
        "sync_success": "✅ महत्वपूर्ण डेटा सिंक किया गया।",
        "complete_first": "⚠️ कृपया पहले मूल्यांकन पूरा करें।",
        "low_risk": "✅ कम जोखिम - स्वस्थ स्थिति",
        "mod_risk": "⚠️ मध्यम जोखिम - बारीकी से निगरानी करें",
        "high_risk": "🚨 उच्च जोखिम - तुरंत डॉक्टर से मिलें",
        "ai_insight_header": "📄 विस्तृत नैदानिक रिपोर्ट",
        "powered_by": "MaternalAI सपोर्ट द्वारा संचालित",
        "clear_chat": "🗑️ चैट इतिहास साफ़ करें",
        "save_success": "✅ सफलतापूर्वक सहेजा गया।",
        "delete_btn": "हटाएं"
    }
}

# --- 3. UI STYLING & CONTRAST FIXES ---
st.set_page_config(page_title="MaternalAI", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* LIGHT TEXT FOR DARK SIDEBAR NAVIGATION */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebarNavItems"] span {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    
    /* BUTTON TEXT COLOR FIX (BRIGHT WHITE ON DARK BG) */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%) !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        border: none !important;
    }
    
    /* ENSURE ALL BUTTON TEXTS INCLUDING DOWNLOAD BUTTON ARE WHITE */
    .stButton>button p, .stButton>button span, .stButton>button div,
    .stDownloadButton>button p, .stDownloadButton>button span { 
        color: #ffffff !important; 
    }

    /* DASHBOARD ELEMENTS */
    p, span, label, h1, h2, h3, h4, .stMarkdown { color: #0f172a; }
    
    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: white !important; font-family: serif; font-size: 50px; font-weight: 900;
        width: 90px; height: 90px; display: flex; align-items: center; 
        justify-content: center; border-radius: 26px; border: 3px solid white;
        box-shadow: 0 10px 25px rgba(30, 27, 75, 0.3);
    }
    .main-card {
        background: white; padding: 35px; border-radius: 28px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.06); border: 1px solid #eef2f6;
        margin-bottom: 25px;
    }
    .report-view {
        background: #fdfdfd;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        margin-top: 15px;
        color: #1e293b;
        font-family: sans-serif;
    }
    [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. AI ENGINE (ANONYMIZED) ---
def call_maternal_ai(prompt, context_type="general", language="English"):
    current_key = get_secure_key()
    patient_ctx = st.session_state.get('patient_data', {})
    score = st.session_state.get('risk_score', 0)
    hb = patient_ctx.get('hb', 11.0)
    bp = patient_ctx.get('bp', 120)

    if not current_key or len(current_key) < 10:
        return get_local_fallback(hb, bp, score, language)

    roles = {
        "clinical": "Senior Maternal Health Doctor. Provide a clinical assessment and roadmap.",
        "tips": "Pregnancy wellness expert. Provide exactly 3 evidence-based tips.",
        "music": "Wellness therapist. Recommend 3 relaxation soundscapes.",
        "chatbot": "Empathetic maternal health assistant. Answer queries clearly."
    }
    
    full_query = (f"System Role: {roles.get(context_type)}\n"
                  f"STRICT REQUIREMENT: Respond ONLY in {language}.\n"
                  f"DO NOT MENTION AI MODEL NAMES (like Gemini or Google).\n"
                  f"PATIENT DATA: {patient_ctx}, Risk: {score}%\n"
                  f"USER QUERY: {prompt}")

    payload = {"contents": [{"role": "user", "parts": [{"text": full_query}]}]}
    # Discovery paths based on verified list
    discovery_paths = [("v1beta", "gemini-2.5-flash"), ("v1beta", "gemini-2.0-flash"), ("v1", "gemini-1.5-flash")]
    
    for version, model_name in discovery_paths:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={current_key}"
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
        except: continue
    return get_local_fallback(hb, bp, score, language)

# --- 5. PDF ENGINE ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MaternalAI Clinical Report', 0, 1, 'C')

def create_pdf(data, ai_note, lang_name):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Patient: {data.get('name', 'N/A')} | Language: {lang_name}", 0, 1)
    pdf.set_font("Arial", '', 11)
    vitals = (f"Age: {data.get('age')} | Week: {data.get('week')}\n"
              f"Hb: {data.get('hb')} | BP: {data.get('bp')} | Weight: {data.get('weight')}")
    pdf.multi_cell(0, 10, vitals)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "Clinical Assessment:", 0, 1)
    pdf.set_font("Arial", '', 10)
    try:
        clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    except:
        clean_ai = "Assessment contains specialized characters. Please see digital dashboard for full details."
    pdf.multi_cell(0, 6, clean_ai)
    return pdf.output(dest='S').encode('latin-1')

# --- 6. SESSION & SIDEBAR ---
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'ai_assessment' not in st.session_state: st.session_state.ai_assessment = ""
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'risk_score' not in st.session_state: st.session_state.risk_score = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin:30px 0;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Choose Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
    c = content[lang]
    
    nav_choice = st.radio(c["nav_header"], c["nav_options"])
    page_idx = c["nav_options"].index(nav_choice)
    st.write("---")
    st.success(f"{c['status_connected']} 🟢")

# --- 7. PAGE LOGIC ---
if page_idx == 0: # Assessment
    st.title(c["title"])
    col1, col2 = st.columns([1, 1.2], gap="large")
    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals_header"]}</h3>', unsafe_allow_html=True)
        name = st.text_input(c["name_lbl"], value=st.session_state.patient_data.get('name', ""))
        age = st.slider(c["age_lbl"], 15, 55, st.session_state.patient_data.get('age', 25))
        hb = st.number_input(c["hb_lbl"], 5.0, 16.0, st.session_state.patient_data.get('hb', 11.0))
        bp = st.number_input(c["bp_lbl"], 80, 200, st.session_state.patient_data.get('bp', 120))
        wt = st.number_input(c["wt_lbl"], 30.0, 200.0, st.session_state.patient_data.get('weight', 60.0))
        wk = st.number_input(c["wk_lbl"], 1, 42, st.session_state.patient_data.get('week', 12))
        
        if st.button(c["btn_analyze"], use_container_width=True):
            st.session_state.patient_data = {"name": name, "age": age, "hb": hb, "bp": bp, "weight": wt, "week": wk}
            risk = 10
            if hb < 11: risk += 30
            if bp > 140: risk += 40
            st.session_state.risk_score = min(risk, 95)
            st.session_state.ai_assessment = ""
            st.success(c["sync_success"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'<div class="main-card"><h3>{c["res_header"]}</h3>', unsafe_allow_html=True)
        if st.session_state.risk_score:
            score = st.session_state.risk_score
            color = "#16a34a" if score < 30 else "#dc2626"
            st.markdown(f'<h1 style="color:{color}; font-size:48px;">{score}% {c["risk_lbl"]}</h1>', unsafe_allow_html=True)
            if score < 30: st.success(c["low_risk"])
            elif score < 60: st.warning(c["mod_risk"])
            else: st.error(c["high_risk"])

            if st.button(c["btn_report"], use_container_width=True):
                with st.spinner("..."):
                    st.session_state.ai_assessment = call_maternal_ai(f"Assessment for {st.session_state.patient_data}", "clinical", lang)
            
            if st.session_state.ai_assessment:
                st.markdown(f"#### {c['ai_insight_header']}")
                # Inline report viewer
                st.markdown(f'<div class="report-view">{st.session_state.ai_assessment}</div>', unsafe_allow_html=True)
                
                # Download option
                st.write("")
                pdf = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment, lang)
                st.download_button(c["dl_btn"], pdf, f"Report_{name}.pdf", "application/pdf", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page_idx == 1: # Tips
    st.title(c["tips_header"])
    if not st.session_state.patient_data: st.warning(c["complete_first"])
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        if st.button(c["btn_tips"], use_container_width=True):
            with st.spinner("..."):
                st.session_state.tips = call_maternal_ai(f"Week {st.session_state.patient_data['week']} wellness tips.", "tips", lang)
        if 'tips' in st.session_state: st.markdown(st.session_state.tips)
        st.markdown('</div>', unsafe_allow_html=True)

elif page_idx == 2: # Med Log
    st.title(c["med_header"])
    st.markdown(f'<div class="main-card"><h3>{c["med_header"]}</h3>', unsafe_allow_html=True)
    m_name = st.text_input(c["med_name"])
    m_time = st.time_input(c["med_time"])
    if st.button(c["med_save"], use_container_width=True):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
        st.success(c["save_success"])
    st.markdown(f"#### {c['med_curr']}")
    for i, m in enumerate(st.session_state.medicines):
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"🔔 **{m['name']}** - {m['time']}")
        if col2.button(c["delete_btn"], key=f"del_{i}"):
            st.session_state.medicines.pop(i)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif page_idx == 3: # Music
    st.title(c["music_header"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.button(c["music_btn"], use_container_width=True):
        with st.spinner("..."):
            st.session_state.music = call_maternal_ai("Wellness music.", "music", lang)
    if 'music' in st.session_state: st.markdown(st.session_state.music)
    st.markdown('</div>', unsafe_allow_html=True)

elif page_idx == 4: # Chat
    st.title(c["chat_header"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]): st.markdown(chat["content"])
    user_q = st.text_input(c["chat_ph"])
    if st.button(c["chat_btn"], use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        reply = call_maternal_ai(user_q, "chatbot", lang)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
    if st.session_state.chat_history: st.button(c["clear_chat"], on_click=lambda: st.session_state.update({"chat_history": []}))
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.caption(f"MaternalAI | {c['powered_by']}")
