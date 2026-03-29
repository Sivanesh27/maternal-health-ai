import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import requests
import json
import base64
from fpdf import FPDF

# --- AI CONFIGURATION (Gemini API) ---
apiKey = "" 

def call_gemini_ai(prompt, system_prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    for delay in [1, 2, 4, 8, 16]:
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response generated.")
        except:
            time.sleep(delay)
    return "Error: Unable to connect to AI Assistant. Check network or API configuration."

# --- PDF GENERATION ENGINE ---
class PDFReport(FPDF):
    def header(self):
        # Arial Bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'MaternalAI Clinical Report', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 250)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, body)
        self.ln()

def create_pdf(data_dict, ai_text=""):
    pdf = PDFReport()
    pdf.add_page()
    
    # Patient Data Section
    pdf.chapter_title("Patient Information")
    info = f"Name: {data_dict['name']}\nAge: {data_dict['age']} years\nPregnancy Week: {data_dict['week']}"
    pdf.chapter_body(info)
    
    # Clinical Vitals
    pdf.chapter_title("Clinical Vitals")
    vitals = f"Hemoglobin: {data_dict['hb']} g/dL\nSystolic BP: {data_dict['bp']} mmHg\nWeight: {data_dict['weight']} kg"
    pdf.chapter_body(vitals)
    
    # Risk Assessment
    pdf.chapter_title("Assessment Results")
    results = f"AI Risk Score: {data_dict['score']}%\nRisk Level: {data_dict['level']}"
    pdf.chapter_body(results)
    
    # AI Explanation
    if ai_text:
        pdf.chapter_title("AI Clinical Insight")
        # Handle unicode/non-latin characters for PDF compatibility (basic fpdf Limitation)
        clean_text = ai_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.chapter_body(clean_text)
        
    return pdf.output(dest='S').encode('latin-1')

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="MaternalAI Support", page_icon="🤰", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* Stylish Modern M Logo */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 25px;
        margin-top: 10px;
    }
    .logo-m {
        background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
        color: white !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 38px;
        font-weight: 800;
        width: 65px;
        height: 65px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 18px;
        box-shadow: 0 10px 15px -3px rgba(67, 56, 202, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-card {
        background: #ffffff;
        padding: 30px;
        border-radius: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
        margin-bottom: 20px;
    }
    
    h1, h2, h3 { color: #1e293b !important; font-weight: 700 !important; }
    p, label, span, li { color: #475569 !important; font-weight: 500; }

    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white !important;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 10px 15px rgba(59, 130, 246, 0.4); 
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
try:
    model = joblib.load('maternal_health_model.pkl')
except:
    st.error("Model file not found. Ensure the training script was run.")

# --- MULTILINGUAL DATABASE ---
content = {
    "English": {
        "title": "Maternal Health Support",
        "lbl_name": "Full Name", "lbl_age": "Age", "lbl_hb": "Hemoglobin (g/dL)", "lbl_bp": "BP (Systolic)", "lbl_wt": "Weight (kg)", "lbl_wk": "Week",
        "btn_calc": "Analyze Health",
        "ai_btn": "Generate AI Detailed Explanation",
        "download_btn": "Download PDF Report",
        "risk_score": "Risk Score",
        "ai_sys_prompt": "You are a professional maternal health expert. Provide a detailed, clinical analysis based on the vitals provided. Focus on risk factors and prevention."
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார ஆதரவு",
        "lbl_name": "முழு பெயர்", "lbl_age": "வயது", "lbl_hb": "ஹீமோகுளோபின்", "lbl_bp": "இரத்த அழுத்தம்", "lbl_wt": "எடை", "lbl_wk": "வாரம்",
        "btn_calc": "பகுப்பாய்வு செய்",
        "ai_btn": "AI விளக்கத்தைப் பெறுங்கள்",
        "download_btn": "PDF அறிக்கையைப் பதிவிறக்கவும்",
        "risk_score": "ஆபத்து மதிப்பெண்",
        "ai_sys_prompt": "நீங்கள் ஒரு தாய்வழி சுகாதார நிபுணர். வழங்கப்பட்ட தரவுகளின் அடிப்படையில் விரிவான மருத்துவ பகுப்பாய்வை வழங்கவும்."
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य सहायता",
        "lbl_name": "पूरा नाम", "lbl_age": "आयु", "lbl_hb": "हीमोग्लोबिन", "lbl_bp": "रक्तचाप", "lbl_wt": "वजन", "lbl_wk": "सप्ताह",
        "btn_calc": "स्वास्थ्य विश्लेषण",
        "ai_btn": "एआई स्पष्टीकरण",
        "download_btn": "PDF रिपोर्ट डाउनलोड करें",
        "risk_score": "जोखिम स्कोर",
        "ai_sys_prompt": "आप एक मातृ स्वास्थ्य विशेषज्ञ हैं। दिए गए डेटा के आधार पर विस्तृत नैदानिक विश्लेषण प्रदान करें।"
    }
}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="logo-container"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("Language Selection", ["English", "தமிழ்", "हिन्दी"])
    c = content[lang]
    st.write("---")
    st.info("System Online 🟢")

# --- MAIN UI ---
st.markdown(f"<h1>{c['title']}</h1>", unsafe_allow_html=True)

col_input, col_output = st.columns([1.1, 1], gap="large")

with col_input:
    st.markdown(f"<div class='main-card'><h3>📝 Patient Vitals</h3>", unsafe_allow_html=True)
    name = st.text_input(c['lbl_name'])
    age = st.slider(c['lbl_age'], 15, 55, 25)
    hb = st.number_input(c['lbl_hb'], 5.0, 16.0, 11.0)
    bp = st.number_input(c['lbl_bp'], 80, 200, 120)
    weight = st.number_input(c['lbl_wt'], 30.0, 250.0, 60.0)
    week = st.number_input(c['lbl_wk'], 1, 42, 12)
    
    st.write("")
    analyze_now = st.button(c['btn_calc'], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.markdown(f"<div class='main-card'><h3>📊 Health Assessment</h3>", unsafe_allow_html=True)
    
    if analyze_now:
        # Prediction
        features = np.array([[age, hb, bp, weight]])
        prob = model.predict_proba(features)[0][1]
        risk_score = round(prob * 100, 2)
        
        # Save to session state to persist through re-renders
        st.session_state['risk_score'] = risk_score
        st.session_state['assessment_done'] = True
        
    if st.session_state.get('assessment_done'):
        score = st.session_state['risk_score']
        st.metric(label=c["risk_score"], value=f"{score}%")
        
        level = "High Risk" if score > 70 else "Moderate Risk" if score > 40 else "Low Risk"
        if score > 70: st.error(level)
        elif score > 40: st.warning(level)
        else: st.success(level)
        
        # AI Section
        st.write("---")
        if st.button(c['ai_btn']):
            with st.spinner("AI Generating detailed insight..."):
                prompt = f"Patient: {name}, Age: {age}, Hb: {hb}, BP: {bp}, Weight: {weight}, Week: {week}. Risk: {score}%. Language: {lang}."
                ai_text = call_gemini_ai(prompt, c['ai_sys_prompt'])
                st.session_state['ai_text'] = ai_text
        
        ai_val = st.session_state.get('ai_text', "")
        if ai_val:
            st.markdown(f"**AI Insight:**\n\n{ai_val}")
            
        # PDF Generation - Always available once assessment is done
        report_data = {
            "name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week,
            "score": score, "level": level
        }
        
        try:
            pdf_bytes = create_pdf(report_data, ai_val)
            st.download_button(
                label=c['download_btn'],
                data=pdf_bytes,
                file_name=f"Report_{name}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            
    else:
        st.info("Input details and click 'Analyze Health' to generate the assessment.")
    st.markdown("</div>", unsafe_allow_html=True)
