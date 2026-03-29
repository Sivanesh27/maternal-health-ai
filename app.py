import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import os
from fpdf import FPDF

# --- 1. CONFIGURATION ---
def get_anthropic_key():
    try:
        if st.secrets and "ANTHROPIC_API_KEY" in st.secrets:
            return str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except:
        pass
    return ""

def local_clinical_brain(hb, bp, risk_score, lang, diagnostic_msg=""):
    """Rule-based clinical fallback if AI is unreachable."""
    if lang == "தமிழ்":
        advice = f"மருத்துவ ஆய்வு: அபாய மதிப்பெண் {risk_score}%. "
        if hb < 11: advice += "ஹீமோகுளோபின் குறைவாக உள்ளது. இரும்புச்சத்து உணவுகளை உட்கொள்ளுங்கள். "
        if bp > 140: advice += "இரத்த அழுத்தம் அதிகம். ஓய்வெடுக்கவும்."
        footer = f"\n\n(குறிப்பு: இணைப்பு சிக்கல். பிழை: {diagnostic_msg})" if diagnostic_msg else "\n\n(குறிப்பு: தானியங்கி மருத்துவ உதவி.)"
        return advice + footer
    elif lang == "हिन्दी":
        advice = f"नैदानिक विश्लेषण: जोखिम स्कोर {risk_score}% है। "
        if hb < 11: advice += "हीमोग्लोबिन कम है। आयरन युक्त भोजन लें। "
        if bp > 140: advice += "रक्तचाप अधिक है। आराम करें।"
        footer = f"\n\n(त्रुटि: {diagnostic_msg})" if diagnostic_msg else "\n\n(नोट: स्वचालित सहायता।)"
        return advice + footer
    else:
        advice = f"Clinical Assessment: Predictive risk is {risk_score}%. "
        if hb < 11: advice += "Action: Increase Iron intake. "
        if bp > 140: advice += "Action: Monitor BP daily and reduce salt. "
        footer = f"\n\n(Diagnostic Details: {diagnostic_msg})" if diagnostic_msg else "\n\n(Note: Rule-based fallback.)"
        return advice + footer

# --- 2. CLAUDE AI CORE ---
def call_claude_ai(prompt, context_type="general", language="English"):
    """Claude API - Primary AI engine for MaternalAI."""
    
    current_key = get_anthropic_key()
    patient_ctx = st.session_state.get('patient_data', {})
    hb = patient_ctx.get('hb', 11.0)
    bp = patient_ctx.get('bp', 120)
    score = st.session_state.get('risk_score', 0)

    if not current_key or len(current_key) < 10:
        return local_clinical_brain(hb, bp, score, language, "ANTHROPIC_API_KEY missing in Streamlit Secrets")

    # Role instructions per context type
    prompts = {
        "clinical": (
            "You are a Senior Maternal Health Doctor with 20 years of experience. "
            "Provide a professional, structured clinical risk assessment with: "
            "1) Risk Summary, 2) Key Concerns, 3) Immediate Action Plan, 4) Follow-up Recommendations."
        ),
        "tips": (
            "You are a certified pregnancy wellness expert. "
            "Provide exactly 3 specific, evidence-based pregnancy tips tailored to the patient's current week and vitals. "
            "Format as numbered list with brief explanations."
        ),
        "music": (
            "You are a prenatal wellness therapist. "
            "Recommend exactly 3 specific relaxation soundscapes, music genres, or meditation styles "
            "suited to this stage of pregnancy. Include why each helps."
        ),
        "chatbot": (
            "You are a warm, empathetic maternal health assistant. "
            "Answer the patient's query with clinical accuracy and emotional sensitivity. "
            "Keep response concise and reassuring."
        )
    }

    system_instruction = prompts.get(context_type, "You are a helpful maternal health assistant.")
    system_instruction += f" Always respond ONLY in {language}. Be clear, concise, and professional."

    user_message = (
        f"Patient Clinical Data: {patient_ctx}\n\n"
        f"Query: {prompt}\n\n"
        f"Important: Your entire response must be in {language} only."
    )

    headers = {
        "Content-Type": "application/json",
        "x-api-key": current_key,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "system": system_instruction,
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data["content"][0]["text"]
        else:
            error_msg = f"Claude API Error {response.status_code}: {response.text[:200]}"
            return local_clinical_brain(hb, bp, score, language, error_msg)

    except requests.exceptions.Timeout:
        return local_clinical_brain(hb, bp, score, language, "Request timed out after 30s")
    except Exception as e:
        return local_clinical_brain(hb, bp, score, language, str(e))

# --- 3. API STATUS CHECK ---
def check_claude_status():
    """Check if Claude API key is valid and working."""
    key = get_anthropic_key()
    if not key or len(key) < 10:
        return False, "No API key found in secrets"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}]
    }
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, timeout=10
        )
        if response.status_code == 200:
            return True, "Claude API connected ✅"
        elif response.status_code == 401:
            return False, "Invalid API Key (401)"
        elif response.status_code == 429:
            return False, "Rate limit hit (429)"
        else:
            return False, f"Error {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

# --- 4. PDF GENERATION ---
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
              f"Hemoglobin: {data.get('hb')} g/dL | BP: {data.get('bp')} mmHg | Weight: {data.get('weight')} kg")
    pdf.multi_cell(0, 10, vitals)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Claude AI Clinical Assessment:", 0, 1)
    pdf.set_font("Arial", '', 10)
    try:
        clean_ai = ai_note.encode('ascii', 'ignore').decode('ascii')
    except:
        clean_ai = "Assessment complete. See application dashboard for localized details."
    pdf.multi_cell(0, 6, clean_ai)
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Generated by MaternalAI powered by Claude AI (Anthropic)", 0, 1, 'C')
    return pdf.output(dest='S').encode('latin-1')

# --- 5. UI STYLING ---
st.set_page_config(page_title="MaternalAI Companion", layout="wide", page_icon="🤰")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    p, span, label, li, div, h1, h2, h3, h4, h5, h6, .stMarkdown,
    [data-testid="stMetricLabel"], .stRadio label, .stSelectbox label { 
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
    .stButton>button:hover { 
        transform: translateY(-4px); 
        box-shadow: 0 12px 25px rgba(67, 56, 202, 0.5); 
    }
    [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 800 !important; }
    .claude-badge {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white !important; font-size: 11px; font-weight: 700;
        padding: 4px 10px; border-radius: 20px; display: inline-block;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 6. SESSION INITIALIZATION ---
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'ai_assessment' not in st.session_state: st.session_state.ai_assessment = ""
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'risk_score' not in st.session_state: st.session_state.risk_score = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div style="display:flex; justify-content:center; margin:30px 0;"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center"><span class="claude-badge">⚡ Powered by Claude AI</span></div>', unsafe_allow_html=True)
    st.write("")
    
    lang = st.selectbox("🌐 Choose Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
    
    nav_map = {
        "English": ["Assessment", "Weekly AI Tips", "Medication Log", "Relaxation Music", "AI Doctor Chat"],
        "தமிழ்": ["மதிப்பீடு", "வாராந்திர குறிப்புகள்", "மருந்துப் பதிவு", "இசை", "AI மருத்துவர்"],
        "हिन्दी": ["मूल्यांकन", "साप्ताहिक सुझाव", "दवा लॉग", "संगीत", "AI डॉक्टर"]
    }
    nav_choice = st.radio("Menu / மெனு / मेनू", nav_map[lang])
    
    st.write("---")
    if st.button("🔍 Check Claude API Status"):
        with st.spinner("Testing connection..."):
            ok, msg = check_claude_status()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    st.write("---")
    st.success("System: Connected 🟢")

# --- MULTILINGUAL CONTENT ---
content = {
    "English": {
        "title": "Maternal Health Dashboard", "vitals": "📝 Clinical Inputs",
        "name": "Full Name", "age": "Age", "hb": "Hemoglobin (g/dL)",
        "bp": "Systolic BP (mmHg)", "wt": "Weight (kg)", "wk": "Pregnancy Week",
        "btn_run": "Analyze Health",
        "res": "📊 Clinical Results", "btn_ai": "Generate Claude AI Report", "dl": "Download PDF",
        "tips_title": "📅 Personalized Weekly Tips", "btn_tips": "Refresh Tips",
        "med_title": "💊 Medication Log", "med_add": "Add Medicine",
        "music_title": "🎵 Wellness Music Recommendations", "music_btn": "Get AI Selection",
        "chat_title": "🤖 AI Doctor Chat", "chat_ph": "Type your health question here...",
        "chat_btn": "Ask Claude AI"
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார மேலாண்மை", "vitals": "📝 மருத்துவத் தரவு",
        "name": "முழு பெயர்", "age": "வயது", "hb": "ஹீமோகுளோபின் (g/dL)",
        "bp": "இரத்த அழுத்தம் (mmHg)", "wt": "எடை (கிலோ)", "wk": "கர்ப்ப வாரம்",
        "btn_run": "ஆய்வு செய்",
        "res": "📊 மருத்துவ முடிவுகள்", "btn_ai": "AI அறிக்கையை உருவாக்கு", "dl": "PDF பதிவிறக்கம்",
        "tips_title": "📅 வாராந்திர AI வழிகாட்டி", "btn_tips": "குறிப்புகளைப் புதுப்பிக்கவும்",
        "med_title": "💊 தினசரி மருந்துப் பதிவு", "med_add": "மருந்தைச் சேர்க்கவும்",
        "music_title": "🎵 தியானப் பரிந்துரைகள்", "music_btn": "இசை பரிந்துரையைப் பெறுங்கள்",
        "chat_title": "🤖 சுகாதார AI உதவியாளர்", "chat_ph": "உங்கள் கேள்வியை இங்கே தட்டச்சு செய்யுங்கள்...",
        "chat_btn": "AI உதவியாளரிடம் கேளுங்கள்"
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य डैशबोर्ड", "vitals": "📝 नैदानिक डेटा",
        "name": "पूरा नाम", "age": "आयु", "hb": "हीमोग्लोबिन (g/dL)",
        "bp": "रक्तचाप (mmHg)", "wt": "वजन (kg)", "wk": "गर्भावस्था सप्ताह",
        "btn_run": "स्वास्थ्य विश्लेषण करें",
        "res": "📊 मूल्यांकन परिणाम", "btn_ai": "AI रिपोर्ट तैयार करें", "dl": "PDF डाउनलोड करें",
        "tips_title": "📅 साप्ताहिक AI गाइड", "btn_tips": "सुझाव अपडेट करें",
        "med_title": "💊 दैनिक दवा लॉग", "med_add": "दवा जोड़ें",
        "music_title": "🎵 कल्याण संगीत", "music_btn": "AI अनुशंसा प्राप्त करें",
        "chat_title": "🤖 स्वास्थ्य AI सहायक", "chat_ph": "अपना स्वास्थ्य प्रश्न यहाँ टाइप करें...",
        "chat_btn": "AI से पूछें"
    }
}
c = content[lang]
page_idx = nav_map[lang].index(nav_choice)

# --- 7. PAGE LOGIC ---

# PAGE 0: Assessment
if page_idx == 0:
    st.title(c["title"])
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown(f'<div class="main-card"><h3>{c["vitals"]}</h3>', unsafe_allow_html=True)
        name   = st.text_input(c["name"],  value=st.session_state.patient_data.get('name', ""))
        age    = st.slider(c["age"], 15, 55, st.session_state.patient_data.get('age', 25))
        hb     = st.number_input(c["hb"],  5.0, 16.0, st.session_state.patient_data.get('hb', 11.0), step=0.1)
        bp     = st.number_input(c["bp"],  80,  200,  st.session_state.patient_data.get('bp', 120))
        weight = st.number_input(c["wt"],  30.0, 250.0, st.session_state.patient_data.get('weight', 60.0), step=0.5)
        week   = st.number_input(c["wk"],  1, 42, st.session_state.patient_data.get('week', 12))

        if st.button(c["btn_run"], use_container_width=True):
            st.session_state.patient_data = {
                "name": name, "age": age, "hb": hb,
                "bp": bp, "weight": weight, "week": week
            }
            # Risk calculation
            risk = 0
            if hb < 8:   risk += 40
            elif hb < 11: risk += 20
            if bp > 160:  risk += 40
            elif bp > 140: risk += 20
            if age > 40 or age < 18: risk += 10
            if week < 12 or week > 36: risk += 10
            st.session_state.risk_score = min(risk, 95) if risk > 0 else 10
            st.session_state.ai_assessment = ""
            st.success("✅ Vitals Synchronized.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="main-card"><h3>{c["res"]}</h3>', unsafe_allow_html=True)
        if st.session_state.risk_score is not None:
            score = st.session_state.risk_score
            color = "#16a34a" if score < 30 else "#d97706" if score < 60 else "#dc2626"
            st.markdown(f'<h1 style="color:{color}; font-size:48px;">{score}% Risk</h1>', unsafe_allow_html=True)
            
            if score < 30:
                st.success("✅ Low Risk — Healthy pregnancy profile")
            elif score < 60:
                st.warning("⚠️ Moderate Risk — Monitor closely")
            else:
                st.error("🚨 High Risk — Consult doctor immediately")

            if st.button(c["btn_ai"], use_container_width=True):
                with st.spinner("Claude AI is analyzing patient data..."):
                    res = call_claude_ai(
                        f"Patient risk score is {score}%. Vitals: {st.session_state.patient_data}",
                        "clinical", lang
                    )
                    st.session_state.ai_assessment = res

            if st.session_state.ai_assessment:
                st.markdown("---")
                st.markdown(f"**🤖 Claude AI Assessment:**\n\n{st.session_state.ai_assessment}")
                st.markdown("---")
                pdf_bytes = create_pdf(st.session_state.patient_data, st.session_state.ai_assessment)
                st.download_button(
                    c["dl"], pdf_bytes,
                    file_name=f"MaternalAI_Report_{name}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("👆 Enter vitals and click Analyze Health to see results.")
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 1: Weekly AI Tips
elif page_idx == 1:
    st.title(c["tips_title"])
    if not st.session_state.patient_data:
        st.warning("⚠️ Please complete the Assessment first.")
    else:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        week_val = st.session_state.patient_data.get('week', 'unknown')
        st.info(f"📅 Generating personalized tips for Week {week_val} of pregnancy.")
        if st.button(c["btn_tips"], use_container_width=True):
            with st.spinner("Claude AI is personalizing your advice..."):
                st.session_state.tips = call_claude_ai(
                    f"Provide tailored wellness guidance for pregnancy week {week_val} "
                    f"with vitals: Hb={st.session_state.patient_data.get('hb')}, "
                    f"BP={st.session_state.patient_data.get('bp')}.",
                    "tips", lang
                )
        if 'tips' in st.session_state:
            st.markdown(st.session_state.tips)
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 2: Medication Log
elif page_idx == 2:
    st.title(c["med_title"])
    st.markdown(f'<div class="main-card"><h3>{c["med_add"]}</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        m_name = st.text_input("Medicine Name / மருந்து / दवा का नाम")
    with col2:
        m_dose = st.text_input("Dosage / அளவு / खुराक (e.g. 1 tablet)")
    m_time = st.time_input("Time / நேரம் / समय")
    m_notes = st.text_input("Notes / குறிப்புகள் / नोट्स (optional)")
    
    if st.button("💾 Save to Log", use_container_width=True):
        if m_name:
            st.session_state.medicines.append({
                "name": m_name, "dose": m_dose,
                "time": str(m_time), "notes": m_notes
            })
            st.success(f"✅ {m_name} saved!")
        else:
            st.warning("Please enter a medicine name.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.medicines:
        st.markdown("### 📋 Current Medication Log")
        for i, m in enumerate(st.session_state.medicines):
            col1, col2 = st.columns([4, 1])
            with col1:
                notes_str = f"| _{m['notes']}_" if m.get('notes') else ''
                st.markdown(f"🔔 **{m['name']}** — {m.get('dose','')} at **{m['time']}** {notes_str}")
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.medicines.pop(i)
                    st.rerun()

# PAGE 3: Relaxation Music
elif page_idx == 3:
    st.title(c["music_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.session_state.patient_data:
        week_val = st.session_state.patient_data.get('week', 'unknown')
        st.info(f"🎵 AI will suggest music for Week {week_val} of your pregnancy.")
    if st.button(c["music_btn"], use_container_width=True):
        with st.spinner("Claude AI is finding your perfect wellness sounds..."):
            patient_info = st.session_state.patient_data if st.session_state.patient_data else "General pregnancy"
            st.session_state.music = call_claude_ai(
                f"Suggest relaxation and wellness music for this patient: {patient_info}",
                "music", lang
            )
    if 'music' in st.session_state:
        st.markdown(st.session_state.music)
    st.markdown('</div>', unsafe_allow_html=True)

# PAGE 4: AI Doctor Chat
elif page_idx == 4:
    st.title(c["chat_title"])
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
    
    # Input
    user_q = st.text_input(c["chat_ph"], key="chat_input")
    if st.button(c["chat_btn"], use_container_width=True):
        if user_q.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.spinner("Claude AI is responding..."):
                reply = call_claude_ai(user_q, "chatbot", lang)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()
        else:
            st.warning("Please type a question first.")
    
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.write("---")
st.caption("MaternalAI v4.0 | Powered by Claude AI (Anthropic) | Trilingual Maternal Health Support")
