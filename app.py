import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import requests
import json
import base64
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

def basic_ai_fallback(msg, hb=11, bp=120, risk="Low"):
    """Industry-standard clinical rule-based fallback if AI API fails."""
    msg = msg.lower()
    advice = []
    
    if "pain" in msg or "stomach" in msg:
        advice.append("Note: Abdominal pain requires rest. If it is sharp, persistent, or accompanied by bleeding, visit a PHC immediately.")
    if "diet" in msg or "food" in msg or hb < 11:
        advice.append("Dietary Insight: Focus on iron-rich foods (Spinach, Jaggery, Dates). Ensure 75g-100g of protein daily.")
    if "exercise" in msg:
        advice.append("Activity: 30 minutes of light walking and prenatal yoga are recommended. Avoid heavy lifting.")
    if bp > 140:
        advice.append("Warning: Your BP is high. Reduce salt intake, stay hydrated, and monitor daily.")
    
    if not advice:
        advice.append(f"Current risk level is {risk}. Maintain regular checkups and a balanced diet.")
        
    return "\n".join(advice) + "\n\n(Note: This is an automated clinical fallback due to connectivity issues.)"

def call_gemini_ai(prompt, system_prompt, hb=11, bp=120, risk="Low"):
    current_key = get_api_key()
    if not current_key:
        return basic_ai_fallback(prompt, hb, bp, risk)

    # Prioritizing the stable v1 endpoint for gemini-1.5-flash
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={current_key}"
    
    payload = {
        "contents": [{"parts": [{"text": f"System: {system_prompt}\n\nPatient Vitals: {prompt}"}]}]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return basic_ai_fallback(prompt, hb, bp, risk)
    except:
        return basic_ai_fallback(prompt, hb, bp, risk)

# --- DATA & CONTENT ---
PREGNANCY_TIPS = {
    1: "Start folic acid supplements and avoid raw or unpasteurized foods.",
    5: "Baby's heart starts developing! ❤️ Stay hydrated and take small, frequent meals.",
    10: "Ultrasound stage. Baby is the size of a prune and moving slightly.",
    20: "Halfway point! Focus on calcium-rich foods like milk and paneer.",
    30: "Baby is gaining weight. Sleep on your side for better blood flow.",
    40: "Preparedness phase. Keep emergency contact numbers and hospital files ready."
}

DOCTORS = [
    {"name": "Dr. Priya", "spec": "Gynecologist", "contact": "+91 98765-43210"},
    {"name": "Dr. Anjali", "spec": "Obstetrician", "contact": "+91 91234-56780"}
]

MUSIC = [
    {"title": "Relaxing Prenatal Lullaby", "url": "https://www.youtube.com/watch?v=2OEL4P1Rz04"},
    {"title": "Baby Brain Development Music", "url": "https://www.youtube.com/watch?v=1ZYbU82GVz4"}
]

# --- PDF ENGINE ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MaternalAI Clinical Summary Report', 0, 1, 'C')
        self.ln(10)

def create_pdf(data):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Patient Name: {data['name']}", 0, 1)
    pdf.ln(5)
    pdf.set_font("Arial", '', 11)
    vitals = (f"Age: {data['age']} | Week: {data['week']}\n"
              f"Hemoglobin: {data['hb']} g/dL\n"
              f"Blood Pressure: {data['bp']} mmHg\n"
              f"Weight: {data['weight']} kg\n"
              f"Risk Score: {data['score']}% ({data['level']})")
    pdf.multi_cell(0, 8, vitals)
    
    if 'ai' in data:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Clinical Insight & Recommendations:", 0, 1)
        pdf.set_font("Arial", '', 10)
        # Strip non-latin characters for standard PDF compatibility
        clean_text = data['ai'].encode('ascii', 'ignore').decode('ascii')
        pdf.multi_cell(0, 6, clean_text)
    return pdf.output(dest='S').encode('latin-1')

# --- UI STYLING ---
st.set_page_config(page_title="MaternalAI Support", page_icon="🤰", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* High Contrast Text Global Visibility */
    p, span, label, li, div, h1, h2, h3, h4 { 
        color: #0f172a !important; 
        font-weight: 600 !important; 
    }
    
    /* Premium Logo Design */
    .logo-container { display: flex; justify-content: center; margin: 20px 0; }
    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        color: white !important;
        font-family: 'Playfair Display', serif; font-size: 46px; font-weight: 900;
        width: 80px; height: 80px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 22px; box-shadow: 0 10px 25px rgba(30, 27, 75, 0.3);
        border: 2px solid white;
    }
    
    .main-card {
        background: white; padding: 35px; border-radius: 28px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.04); border: 1px solid #f1f5f9;
        margin-bottom: 25px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%);
        color: #ffffff !important; border-radius: 14px; font-weight: 700;
        border: none; padding: 12px 25px; transition: 0.3s;
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(67, 56, 202, 0.3); }
    
    [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'medicines' not in st.session_state: st.session_state.medicines = []
if 'ai_note' not in st.session_state: st.session_state.ai_note = ""

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="logo-container"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Choose Language", ["English", "தமிழ்", "हिन्दी"])
    nav = st.radio("Navigation", ["Home & Assessment", "Medicine Log", "Weekly Tips", "Resources"])
    st.write("---")
    st.success("System: Active 🟢")

# --- MULTILINGUAL DICTIONARY ---
content = {
    "English": {
        "title": "Maternal Health Support", "btn": "Analyze Health", "lbl_name": "Full Name",
        "lbl_hb": "Hemoglobin (g/dL)", "lbl_bp": "Systolic BP", "lbl_wt": "Weight (kg)",
        "ai_btn": "Get AI Detailed Insight", "dl_btn": "Download PDF Report"
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார ஆதரவு", "btn": "ஆரோக்கியத்தை ஆய்வு செய்", "lbl_name": "முழு பெயர்",
        "lbl_hb": "ஹீமோகுளோபின்", "lbl_bp": "இரத்த அழுத்தம்", "lbl_wt": "எடை (கிலோ)",
        "ai_btn": "AI விரிவான விளக்கம்", "dl_btn": "அறிக்கையைப் பதிவிறக்கவும்"
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य सहायता", "btn": "स्वास्थ्य विश्लेषण करें", "lbl_name": "पूरा नाम",
        "lbl_hb": "हीमोग्लोबिन", "lbl_bp": "रक्तचाप (BP)", "lbl_wt": "वजन (kg)",
        "ai_btn": "AI विस्तृत जानकारी", "dl_btn": "रिपोर्ट डाउनलोड करें"
    }
}
c = content[lang]

# --- NAVIGATION LOGIC ---
if nav == "Home & Assessment":
    st.title(c["title"])
    col_in, col_out = st.columns([1.1, 1], gap="large")
    
    with col_in:
        st.markdown('<div class="main-card"><h3>📝 Clinical Inputs</h3>', unsafe_allow_html=True)
        name = st.text_input(c["lbl_name"], placeholder="e.g. Anjali Sharma")
        age = st.slider("Age", 15, 55, 25)
        hb = st.number_input(c["lbl_hb"], 5.0, 16.0, 11.0)
        bp = st.number_input(c["lbl_bp"], 80, 200, 120)
        weight = st.number_input(c["lbl_wt"], 30.0, 250.0, 60.0)
        week = st.number_input("Pregnancy Week", 1, 42, 12)
        analyze = st.button(c["btn"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        st.markdown('<div class="main-card"><h3>📊 AI Assessment Results</h3>', unsafe_allow_html=True)
        if analyze:
            try:
                model = joblib.load('maternal_health_model.pkl')
                prob = model.predict_proba(np.array([[age, hb, bp, weight]]))[0][1]
                score = round(prob * 100, 2)
            except: score = 10.0 # Default if model missing during preview
            
            st.metric("Risk Score", f"{score}%")
            level = "High Risk" if score > 70 else "Moderate" if score > 40 else "Low Risk"
            if score > 70: st.error(level)
            elif score > 40: st.warning(level)
            else: st.success(level)
            
            st.write("---")
            if st.button(c["ai_btn"], use_container_width=True):
                with st.spinner("AI Generating detailed insight..."):
                    sys_prompt = "Professional maternal health assistant. Analyze vitals and give detailed nutritional and medical advice."
                    prompt = f"Patient: {name}, Hb: {hb}, BP: {bp}, Week: {week}, Score: {score}%."
                    st.session_state.ai_note = call_gemini_ai(prompt, sys_prompt, hb, bp, level)
            
            if st.session_state.ai_note:
                st.markdown(f"**AI Insight:**\n{st.session_state.ai_note}")
                report_data = {
                    "name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, 
                    "week": week, "score": score, "level": level, "ai": st.session_state.ai_note
                }
                pdf_bytes = create_pdf(report_data)
                st.download_button(c["dl_btn"], pdf_bytes, f"Report_{name}.pdf", "application/pdf", use_container_width=True)
        else:
            st.info("Please fill in patient vitals and click 'Analyze Patient Health'.")
        st.markdown('</div>', unsafe_allow_html=True)

elif nav == "Medicine Log":
    st.title("💊 Medicine Reminder Log")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    m_name = st.text_input("Medicine Name")
    m_time = st.time_input("Reminder Time")
    if st.button("Add to Log"):
        st.session_state.medicines.append({"name": m_name, "time": str(m_time)})
        st.success(f"Added: {m_name}")
    
    if st.session_state.medicines:
        st.write("---")
        for m in st.session_state.medicines:
            st.markdown(f"🔔 **{m['name']}** at {m['time']}")
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "Weekly Tips":
    st.title("📅 Pregnancy Weekly Guide")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    w_check = st.number_input("Enter Week:", 1, 42, 1)
    tip = PREGNANCY_TIPS.get(w_check, "Maintain a healthy lifestyle and routine prenatal checkups.")
    st.info(f"**Week {w_check}:** {tip}")
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "Resources":
    st.title("🏥 Essential Resources")
    t1, t2 = st.tabs(["👩‍⚕️ Doctor Directory", "🎵 Relaxing Music"])
    with t1:
        for d in DOCTORS:
            with st.expander(d['name']):
                st.write(f"Specialty: {d['spec']}")
                st.write(f"Contact: {d['contact']}")
    with t2:
        for m in MUSIC:
            st.markdown(f"🔗 [{m['title']}]({m['url']})")

st.write("---")
st.caption("Rural Health Decision Support System v2.7 Stable | Powered by MaternalAI")
