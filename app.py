import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import requests
import json

# --- AI CONFIGURATION (Gemini API) ---
# The environment provides the API key automatically at runtime.
apiKey = "" 

def call_gemini_ai(prompt, system_prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    
    # Exponential backoff retry logic (Industry Standard)
    for delay in [1, 2, 4, 8, 16]:
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response generated.")
        except:
            time.sleep(delay)
    return "Error: Unable to connect to AI Assistant. Please check your connection."

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="MaternalAI Support", page_icon="🤰", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .stApp, .stMarkdown, p, span, label, li { color: #2c3e50 !important; }
    .main-card {
        background: #ffffff;
        padding: 30px;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        animation: fadeIn 0.8s ease-in-out;
    }
    .main-card h1, .main-card h2, .main-card h3, .main-card p, .main-card label { color: #1e293b !important; }
    h1 { color: #1e1b4b !important; font-weight: 800 !important; }
    h2, h3 { color: #312e81 !important; font-weight: 700 !important; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white !important;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 15px rgba(59, 130, 246, 0.4); }
    [data-testid="stMetricValue"] { color: #4338ca !important; font-weight: 800 !important; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
try:
    model = joblib.load('maternal_health_model.pkl')
except:
    st.error("Model file not found. Please run the training shell first.")

# --- MULTILINGUAL DATABASE ---
content = {
    "English": {
        "title": "Maternal Health Decision Support",
        "tagline": "AI-powered clinical guidance for rural wellness",
        "p_info": "Patient Particulars",
        "lbl_name": "Full Name", "lbl_age": "Age", "lbl_hb": "Hemoglobin", "lbl_bp": "BP (Systolic)", "lbl_wt": "Weight", "lbl_wk": "Week",
        "btn_calc": "Analyze Health",
        "results_hdr": "Health Assessment",
        "risk_score": "Risk Score",
        "ai_btn": "Get AI Detailed Explanation",
        "download_btn": "Download Full Report",
        "ai_loading": "AI is generating a detailed clinical report...",
        "risk_data_hdr": "Clinical Risk Breakdown",
        "nut_hdr": "Nutritional Roadmap",
        "emg_hdr": "Emergency Protocol",
        "ai_sys_prompt": "You are a professional maternal health assistant. Explain the risks and give detailed advice based on the provided clinical data."
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார AI ஆதரவு",
        "tagline": "கிராமப்புற நலனுக்கான AI மருத்துவ வழிகாட்டுதல்",
        "p_info": "நோயாளி விவரங்கள்",
        "lbl_name": "பெயர்", "lbl_age": "வயது", "lbl_hb": "ஹீமோகுளோபின்", "lbl_bp": "இரத்த அழுத்தம்", "lbl_wt": "எடை", "lbl_wk": "வாரம்",
        "btn_calc": "ஆரோக்கியத்தை ஆய்வு செய்",
        "results_hdr": "சுகாதார மதிப்பீடு",
        "risk_score": "ஆபத்து மதிப்பெண்",
        "ai_btn": "AI விரிவான விளக்கத்தைப் பெறுங்கள்",
        "download_btn": "முழு அறிக்கையைப் பதிவிறக்கவும்",
        "ai_loading": "AI விரிவான மருத்துவ அறிக்கையை உருவாக்குகிறது...",
        "risk_data_hdr": "மருத்துவ இடர் முறிவு",
        "nut_hdr": "ஊட்டச்சத்து வழிகாட்டி",
        "emg_hdr": "அவசரகால நெறிமுறை",
        "ai_sys_prompt": "நீங்கள் ஒரு தொழில்முறை தாய்வழி சுகாதார உதவியாளர். வழங்கப்பட்ட மருத்துவத் தரவுகளின் அடிப்படையில் அபாயங்களை விளக்கி விரிவான ஆலோசனைகளை வழங்கவும்."
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य एआई सहायता",
        "tagline": "ग्रामीण स्वास्थ्य के लिए एआई-आधारित मार्गदर्शन",
        "p_info": "रोगी का विवरण",
        "lbl_name": "नाम", "lbl_age": "आयु", "lbl_hb": "हीमोग्लोबिन", "lbl_bp": "रक्तचाप", "lbl_wt": "वजन", "lbl_wk": "सप्ताह",
        "btn_calc": "स्वास्थ्य विश्लेषण करें",
        "results_hdr": "स्वास्थ्य मूल्यांकन",
        "risk_score": "जोखिम स्कोर",
        "ai_btn": "एआई विस्तृत विवरण प्राप्त करें",
        "download_btn": "पूरी रिपोर्ट डाउनलोड करें",
        "ai_loading": "एआई विस्तृत नैदानिक रिपोर्ट तैयार कर रहा है...",
        "risk_data_hdr": "नैदानिक जोखिम विवरण",
        "nut_hdr": "पोषण मार्गदर्शिका",
        "emg_hdr": "आपातकालीन प्रोटोकॉल",
        "ai_sys_prompt": "आप एक पेशेवर मातृ स्वास्थ्य सहायक हैं। दिए गए नैदानिक डेटा के आधार पर जोखिमों की व्याख्या करें और विस्तृत सलाह दें।"
    }
}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"<h2 style='text-align: center;'>🤰 MaternalAI</h2>", unsafe_allow_html=True)
    lang = st.selectbox("Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
    c = content[lang]
    st.write("---")
    st.caption("AI Assistant: Enabled 🤖")

# --- MAIN UI ---
st.markdown(f"<h1>{c['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #475569; font-size: 1.1rem;'>{c['tagline']}</p>", unsafe_allow_html=True)

col_input, col_output = st.columns([1.1, 1], gap="large")

with col_input:
    st.markdown(f"<div class='main-card'><h3>📝 {c['p_info']}</h3>", unsafe_allow_html=True)
    name = st.text_input(c['lbl_name'])
    age = st.slider(c['lbl_age'], 15, 55, 25)
    hb = st.number_input(c['lbl_hb'], 5.0, 16.0, 11.0)
    bp = st.number_input(c['lbl_bp'], 80, 200, 120)
    weight = st.number_input(c['lbl_wt'], 30.0, 250.0, 60.0)
    week = st.number_input(c['lbl_wk'], 1, 42, 12)
    btn_analyze = st.button(c['btn_calc'], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.markdown(f"<div class='main-card'><h3>📊 {c['results_hdr']}</h3>", unsafe_allow_html=True)
    if btn_analyze:
        # Standard Prediction
        features = np.array([[age, hb, bp, weight]])
        prob = model.predict_proba(features)[0][1]
        score = round(prob * 100, 2)
        st.metric(label=c["risk_score"], value=f"{score}%")
        
        # Risk Logic Strings
        risk_desc = "High" if score > 70 else "Moderate" if score > 40 else "Low"
        
        # AI Insight Section
        st.write("---")
        if st.button(c['ai_btn']):
            with st.spinner(c['ai_loading']):
                prompt = f"Patient: {name}, Age: {age}, Hb: {hb}, BP: {bp}, Weight: {weight}, Week: {week}. Risk Score: {score}%. Provide a detailed clinical explanation in {lang}."
                ai_response = call_gemini_ai(prompt, c['ai_sys_prompt'])
                st.markdown(f"#### 🤖 AI Deep Insight\n{ai_response}")
                st.session_state['ai_report'] = ai_response

        # Download Report Functionality
        if 'ai_report' in st.session_state:
            full_report = f"""
            MATERNAL HEALTH REPORT
            ----------------------
            Patient: {name}
            Age: {age} | Weight: {weight}kg
            Hb: {hb} g/dL | BP: {bp} mmHg | Week: {week}
            
            AI Risk Score: {score}%
            Assessment: {risk_desc}
            
            AI DETAILED EXPLANATION:
            {st.session_state['ai_report']}
            
            Generated by MaternalAI Decision Support System.
            """
            st.download_button(label=c['download_btn'], data=full_report, file_name=f"Report_{name}.txt", mime="text/plain")
    else:
        st.info("Complete the form to see analysis.")
    st.markdown("</div>", unsafe_allow_html=True)
