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
# For external hosting (Streamlit Cloud), add your key to 'Secrets' as GEMINI_API_KEY.
apiKey = "" 

def get_api_key():
    try:
        if st.secrets and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except:
        pass
    return os.environ.get("GEMINI_API_KEY", apiKey)

def call_gemini_ai(prompt, system_prompt):
    current_key = get_api_key()
    if not current_key:
        return "Error: API Key is missing. Please add GEMINI_API_KEY to Streamlit Secrets."

    # Robust model list. 'gemini-1.5-flash' is the modern standard.
    # Trying v1 first as it is more stable than v1beta for many users.
    models_to_try = [
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-flash"),
        ("v1", "gemini-pro"),
        ("v1beta", "gemini-pro")
    ]
    
    last_error = ""
    
    for version, model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={current_key}"
        
        # We use the most basic payload structure to avoid '400 Bad Request' errors.
        # We merge the system instruction into the user prompt for maximum compatibility.
        combined_prompt = f"{system_prompt}\n\nPatient Vitals & Data: {prompt}\n\nPlease provide a detailed clinical explanation."
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": combined_prompt}]
                }
            ]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=20)
            if response.status_code == 200:
                result = response.json()
                # Extract text safely
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                last_error = f"Model {model_name} ({version}) failed with Status {response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)
            
        time.sleep(1) 
        
    return f"AI Connection Failed. Please check your API key status in Google AI Studio dashboard.\nDetails: {last_error}"

# --- PDF GENERATION ENGINE ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'MaternalAI - Clinical Assessment Report', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 255)
        self.cell(0, 10, f"  {title}", 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

def create_pdf(data_dict, ai_text=""):
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.chapter_title("Patient Identification")
    info = f"Patient Name: {data_dict['name']}\nAge: {data_dict['age']} years\nPregnancy Progress: Week {data_dict['week']}"
    pdf.chapter_body(info)
    
    pdf.chapter_title("Clinical Vitals Recorded")
    vitals = (f"Hemoglobin Level: {data_dict['hb']} g/dL\n"
              f"Systolic Blood Pressure: {data_dict['bp']} mmHg\n"
              f"Body Weight: {data_dict['weight']} kg")
    pdf.chapter_body(vitals)
    
    pdf.chapter_title("Risk Assessment Result")
    results = f"Calculated AI Risk Score: {data_dict['score']}%\nClassification: {data_dict['level']}"
    pdf.chapter_body(results)
    
    if ai_text:
        pdf.chapter_title("AI Clinical Insight & Recommendations")
        # Strip non-Latin characters for basic PDF compatibility
        clean_text = ai_text.encode('ascii', 'ignore').decode('ascii')
        pdf.chapter_body(clean_text)
        
    return pdf.output(dest='S').encode('latin-1')

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="MaternalAI Support", page_icon="🤰", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #fcfcfd; }
    
    /* Stylish Modern M Logo with Animation */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 25px;
        margin-top: 10px;
    }
    .logo-m {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%);
        color: white !important;
        font-family: 'Times New Roman', serif;
        font-size: 42px;
        font-weight: 900;
        width: 70px;
        height: 70px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 20px;
        box-shadow: 0 12px 20px -5px rgba(67, 56, 202, 0.4);
        border: 2px solid #fff;
        transition: transform 0.3s ease;
    }
    .logo-m:hover { transform: rotate(5deg) scale(1.05); }
    
    /* High Contrast Text Visibility */
    .main-card {
        background: #ffffff;
        padding: 35px;
        border-radius: 28px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.04);
        border: 1px solid #edf2f7;
        margin-bottom: 25px;
    }
    
    h1, h2, h3 { color: #0f172a !important; font-weight: 800 !important; }
    p, label, span, li, div { color: #334155 !important; font-weight: 500 !important; }

    .stButton>button {
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
        color: white !important;
        border-radius: 14px;
        border: none;
        padding: 0.85rem 1.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: translateY(-3px); 
        box-shadow: 0 12px 20px rgba(37, 99, 235, 0.3); 
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
        "title": "Maternal Health Decision Support",
        "lbl_name": "Full Name", "lbl_age": "Age", "lbl_hb": "Hemoglobin (g/dL)", "lbl_bp": "BP (Systolic)", "lbl_wt": "Weight (kg)", "lbl_wk": "Week",
        "btn_calc": "Analyze Patient Health",
        "ai_btn": "Generate Detailed AI Analysis",
        "download_btn": "Download Clinical PDF Report",
        "risk_score": "Predictive Risk Score",
        "ai_sys_prompt": "You are a professional maternal health expert. Provide a detailed, clinical analysis based on the vitals provided. Focus on risk factors, nutritional advice, and emergency prevention."
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார ஆதரவு",
        "lbl_name": "முழு பெயர்", "lbl_age": "வயது", "lbl_hb": "ஹீமோகுளோபின்", "lbl_bp": "இரத்த அழுத்தம்", "lbl_wt": "எடை", "lbl_wk": "வாரம்",
        "btn_calc": "பகுப்பாய்வு செய்",
        "ai_btn": "AI விரிவான விளக்கம்",
        "download_btn": "PDF அறிக்கையைப் பதிவிறக்கவும்",
        "risk_score": "ஆபத்து மதிப்பெண்",
        "ai_sys_prompt": "நீங்கள் ஒரு தாய்வழி சுகாதார நிபுணர். வழங்கப்பட்ட தரவுகளின் அடிப்படையில் விரிவான மருத்துவ பகுப்பாய்வை வழங்கவும்."
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य सहायता",
        "lbl_name": "पूरा नाम", "lbl_age": "आयु", "lbl_hb": "हीमोग्लोबिन", "lbl_bp": "रक्तचाप", "lbl_wt": "वजन", "lbl_wk": "सप्ताह",
        "btn_calc": "स्वास्थ्य विश्लेषण",
        "ai_btn": "एआई विस्तृत विश्लेषण",
        "download_btn": "PDF रिपोर्ट डाउनलोड करें",
        "risk_score": "जोखिम स्कोर",
        "ai_sys_prompt": "आप एक मातृ स्वास्थ्य विशेषज्ञ हैं। दिए गए डेटा के आधार पर विस्तृत नैदानिक विश्लेषण प्रदान करें।"
    }
}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="logo-container"><div class="logo-m">M</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
    c = content[lang]
    st.write("---")
    st.success("AI Core: Connected 🟢")

# --- MAIN UI ---
st.markdown(f"<h1>{c['title']}</h1>", unsafe_allow_html=True)

col_input, col_output = st.columns([1.1, 1], gap="large")

with col_input:
    st.markdown(f"<div class='main-card'><h3>📝 Input Patient Vitals</h3>", unsafe_allow_html=True)
    name = st.text_input(c['lbl_name'], placeholder="Enter patient name...")
    age = st.slider(c['lbl_age'], 15, 55, 25)
    hb = st.number_input(c['lbl_hb'], 5.0, 16.0, 11.0, help="Clinical Hemoglobin level")
    bp = st.number_input(c['lbl_bp'], 80, 200, 120, help="Systolic Blood Pressure")
    weight = st.number_input(c['lbl_wt'], 30.0, 250.0, 60.0)
    week = st.number_input(c['lbl_wk'], 1, 42, 12)
    
    st.write("")
    analyze_now = st.button(c['btn_calc'], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.markdown(f"<div class='main-card'><h3>📊 AI Assessment Results</h3>", unsafe_allow_html=True)
    
    if analyze_now:
        features = np.array([[age, hb, bp, weight]])
        prob = model.predict_proba(features)[0][1]
        risk_score = round(prob * 100, 2)
        
        st.session_state['risk_score'] = risk_score
        st.session_state['assessment_done'] = True
        
    if st.session_state.get('assessment_done'):
        score = st.session_state['risk_score']
        st.metric(label=c["risk_score"], value=f"{score}%")
        
        level = "High Risk Pregnancy" if score > 70 else "Moderate Risk" if score > 40 else "Low Risk"
        if score > 70: st.error(f"⚠️ {level}")
        elif score > 40: st.warning(f"⚠️ {level}")
        else: st.success(f"✅ {level}")
        
        st.write("---")
        if st.button(c['ai_btn'], use_container_width=True):
            with st.spinner("AI is analyzing clinical data..."):
                vitals_str = f"Age: {age}, Hb: {hb}, BP: {bp}, Weight: {weight}, Week: {week}"
                ai_text = call_gemini_ai(vitals_str, c['ai_sys_prompt'])
                st.session_state['ai_text'] = ai_text
        
        ai_val = st.session_state.get('ai_text', "")
        if ai_val:
            st.markdown(f"#### 🤖 AI Clinical Insight\n{ai_val}")
            
            # PDF Generation
            report_data = {
                "name": name, "age": age, "hb": hb, "bp": bp, "weight": weight, "week": week,
                "score": score, "level": level
            }
            
            try:
                pdf_bytes = create_pdf(report_data, ai_val)
                st.download_button(
                    label=c['download_btn'],
                    data=pdf_bytes,
                    file_name=f"Clinical_Report_{name}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF Error: {e}")
            
    else:
        st.info("Please fill in patient vitals and click 'Analyze Patient Health'.")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Healthcare Decision Support System v2.5 | For professional clinical use only.</p>", unsafe_allow_html=True)
