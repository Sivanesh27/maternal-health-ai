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

    # Comprehensive list of model identifiers and versions to handle regional 404 errors.
    # We prioritize 1.5-flash as it is the current industry standard for speed/cost.
    models_to_try = [
        ("v1beta", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-flash-latest"),
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-pro"),
        ("v1beta", "gemini-1.0-pro")
    ]
    
    last_error = ""
    
    for version, model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={current_key}"
        
        # Simplified payload: Merging instructions into the user prompt for maximum compatibility.
        combined_prompt = f"{system_prompt}\n\nPatient Vitals & Clinical Data for Analysis:\n{prompt}\n\nPlease provide a comprehensive medical explanation and next steps."
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": combined_prompt}]
                }
            ]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=25)
            if response.status_code == 200:
                result = response.json()
                # Safely extract the generated text
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                last_error = f"Model {model_name} ({version}) returned {response.status_code}: {response.text}"
        except Exception as e:
            last_error = f"Model {model_name} failed: {str(e)}"
            
        time.sleep(1) # Short pause between retries to avoid rate limiting
        
    return f"AI Connection Failed. Please ensure your API key is active and the Generative Language API is enabled in your Google Cloud Project.\nDetails: {last_error}"

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
        # Standard FPDF handles Latin-1 only; stripping non-compatible characters
        clean_text = ai_text.encode('ascii', 'ignore').decode('ascii')
        pdf.chapter_body(clean_text)
        
    return pdf.output(dest='S').encode('latin-1')

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="MaternalAI Support", page_icon="🤰", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #fcfcfd; }
    
    /* Stylish Premium M Logo */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 25px;
        margin-top: 15px;
    }
    .logo-m {
        background: linear-gradient(135deg, #0f172a 0%, #312e81 100%);
        color: white !important;
        font-family: 'Playfair Display', 'Times New Roman', serif;
        font-size: 44px;
        font-weight: 900;
        width: 75px;
        height: 75px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 22px;
        box-shadow: 0 15px 25px -5px rgba(49, 46, 129, 0.4);
        border: 2px solid #ffffff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .logo-m:hover { 
        transform: scale(1.1) rotate(-3deg); 
        box-shadow: 0 20px 30px -5px rgba(49, 46, 129, 0.5);
    }
    
    /* Enhanced Contrast for Text Visibility */
    .main-card {
        background: #ffffff;
        padding: 35px;
        border-radius: 28px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.04);
        border: 1px solid #f1f5f9;
        margin-bottom: 25px;
    }
    
    h1, h2, h3 { color: #1e1b4b !important; font-weight: 800 !important; }
    p, label, span, li, div { color: #334155 !important; font-weight: 600 !important; }

    .stButton>button {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%);
        color: white !important;
        border-radius: 14px;
        border: none;
        padding: 0.9rem 2rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover { 
        transform: translateY(-4px); 
        box-shadow: 0 15px 25px rgba(67, 56, 202, 0.35); 
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
try:
    model = joblib.load('maternal_health_model.pkl')
except:
    st.error("AI Core file missing. Ensure model training is complete.")

# --- MULTILINGUAL DATABASE ---
content = {
    "English": {
        "title": "Maternal Health Decision Support",
        "lbl_name": "Full Name", "lbl_age": "Age", "lbl_hb": "Hemoglobin (g/dL)", "lbl_bp": "BP (Systolic)", "lbl_wt": "Weight (kg)", "lbl_wk": "Week",
        "btn_calc": "Analyze Patient Health",
        "ai_btn": "Generate Detailed AI Clinical Insight",
        "download_btn": "Download Clinical PDF Report",
        "risk_score": "Predictive Risk Score",
        "ai_sys_prompt": "You are a professional maternal health expert. Provide a detailed, clinical analysis based on the vitals provided. Focus on risk factors, specific nutritional advice, and emergency prevention."
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
    lang = st.selectbox("Select Language", ["English", "தமிழ்", "हिन्दी"])
    c = content[lang]
    st.write("---")
    st.success("Clinical Engine: Connected 🟢")
    st.caption("v2.6 Stable Release")

# --- MAIN UI ---
st.markdown(f"<h1>{c['title']}</h1>", unsafe_allow_html=True)

col_input, col_output = st.columns([1.1, 1], gap="large")

with col_input:
    st.markdown(f"<div class='main-card'><h3>📝 Input Patient Vitals</h3>", unsafe_allow_html=True)
    name = st.text_input(c['lbl_name'], placeholder="e.g. Mrs. Jane Doe")
    age = st.slider(c['lbl_age'], 15, 55, 25)
    hb = st.number_input(c['lbl_hb'], 5.0, 16.0, 11.0, help="Clinical Hemoglobin level (g/dL)")
    bp = st.number_input(c['lbl_bp'], 80, 200, 120, help="Systolic Blood Pressure (mmHg)")
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
            with st.spinner("Clinical AI is generating deep insight..."):
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
                    file_name=f"Clinical_Assessment_{name}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF Generation Issue: {e}")
            
    else:
        st.info("Please fill in the patient vitals and click 'Analyze Patient Health' to start.")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.95rem; font-weight: 400;'>MaternalAI Decision Support v2.6 | Designed for Clinical Professional Use.</p>", unsafe_allow_html=True)
