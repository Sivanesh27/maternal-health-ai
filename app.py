import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="MaternalAI Support", page_icon="🤰", layout="wide")

# Modern Aesthetic CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Card Container */
    .main-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        animation: fadeIn 0.8s ease-in-out;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Buttons */
    .stButton>button {
        background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(37, 117, 252, 0.4);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4a148c;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Poppins', sans-serif;
    }
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
        "lbl_name": "Full Name",
        "lbl_age": "Age (Years)",
        "lbl_hb": "Hemoglobin (Hb) g/dL",
        "lbl_bp": "Systolic Blood Pressure (mmHg)",
        "lbl_weight": "Weight (kg)",
        "lbl_week": "Pregnancy Week",
        "btn_calc": "Generate Clinical Analysis",
        "results_hdr": "Health Assessment",
        "risk_score": "Risk Probability Score",
        "high": "🔴 HIGH RISK: Critical Attention Needed",
        "med": "🟡 MODERATE RISK: Monitor Closely",
        "low": "🟢 LOW RISK: Maintaining Healthy Progress",
        "nut_hdr": "Nutritional Roadmap",
        "emg_hdr": "Emergency Protocol",
        "emg_msg": "⚠️ IMMEDIATE ACTION: Transport to the nearest Medical Center immediately.",
        "safe_msg": "✅ Stable: Continue prescribed prenatal vitamins and routine visits.",
        "anemia_s": "Severe Anemia: Immediate Iron infusions or high-dose supplements + Liver/Beetroot/Spinach.",
        "anemia_m": "Mild Anemia: Increase Green Leafy Vegetables, Jaggery, and Vitamin C.",
        "diet_ok": "Balanced Nutrition: Your diet is supporting healthy fetal development.",
        "analysis_msg": "Enter data to initiate AI diagnostic..."
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார AI ஆதரவு",
        "tagline": "கிராமப்புற நலனுக்கான AI-ஆல் இயக்கப்படும் மருத்துவ வழிகாட்டுதல்",
        "p_info": "நோயாளி விவரங்கள்",
        "lbl_name": "முழு பெயர்",
        "lbl_age": "வயது",
        "lbl_hb": "ஹீமோகுளோபின் (Hb) g/dL",
        "lbl_bp": "இரத்த அழுத்தம் (Systolic)",
        "lbl_weight": "எடை (கிலோ)",
        "lbl_week": "கர்ப்ப வாரம்",
        "btn_calc": "மருத்துவ பகுப்பாய்வை உருவாக்கு",
        "results_hdr": "சுகாதார மதிப்பீடு",
        "risk_score": "ஆபத்து நிகழ்தகவு",
        "high": "🔴 அதிக ஆபத்து: உடனடி கவனம் தேவை",
        "med": "🟡 மிதமான ஆபத்து: உன்னிப்பாக கண்காணிக்கவும்",
        "low": "🟢 குறைந்த ஆபத்து: ஆரோக்கியமான முன்னேற்றம்",
        "nut_hdr": "ஊட்டச்சத்து வழிகாட்டி",
        "emg_hdr": "அவசரகால நெறிமுறை",
        "emg_msg": "⚠️ உடனடி நடவடிக்கை: உடனே மருத்துவமனைக்குச் செல்லுங்கள்.",
        "safe_msg": "✅ சீராக உள்ளது: வழக்கமான பரிசோதனைகளைத் தொடரவும்.",
        "anemia_s": "கடுமையான ரத்தச்சோகை: இரும்புச்சத்து நிறைந்த உணவு மற்றும் மாத்திரைகள் அவசியம்.",
        "anemia_m": "லேசான ரத்தச்சோகை: பச்சை காய்கறிகள் மற்றும் வைட்டமின் C சேர்க்கவும்.",
        "diet_ok": "சீரான உணவு: உங்கள் உணவு ஆரோக்கியமான வளர்ச்சிக்கு உதவுகிறது.",
        "analysis_msg": "தகவலை உள்ளிட்டு பகுப்பாய்வைத் தொடங்கவும்..."
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य एआई सहायता",
        "tagline": "ग्रामीण स्वास्थ्य के लिए एआई-आधारित नैदानिक मार्गदर्शन",
        "p_info": "रोगी का विवरण",
        "lbl_name": "पूरा नाम",
        "lbl_age": "आयु (वर्ष)",
        "lbl_hb": "हीमोग्लोबिन (Hb) g/dL",
        "lbl_bp": "सिस्टोलिक रक्तचाप (mmHg)",
        "lbl_weight": "वजन (किलोग्राम)",
        "lbl_week": "गर्भावस्था का सप्ताह",
        "btn_calc": "नैदानिक रिपोर्ट तैयार करें",
        "results_hdr": "स्वास्थ्य मूल्यांकन",
        "risk_score": "जोखिम संभावना स्कोर",
        "high": "🔴 उच्च जोखिम: तत्काल ध्यान देने की आवश्यकता है",
        "med": "🟡 मध्यम जोखिम: बारीकी से निगरानी करें",
        "low": "🟢 कम जोखिम: स्वस्थ प्रगति बनी हुई है",
        "nut_hdr": "पोषण संबंधी मार्गदर्शिका",
        "emg_hdr": "आपातकालीन प्रोटोकॉल",
        "emg_msg": "⚠️ तत्काल कार्रवाई: तुरंत नजदीकी अस्पताल ले जाएं।",
        "safe_msg": "✅ स्थिर: नियमित जांच और विटामिन जारी रखें।",
        "anemia_s": "गंभीर एनीमिया: आयरन युक्त भोजन (पालक, गुड़) और सप्लीमेंट तुरंत शुरू करें।",
        "anemia_m": "हल्का एनीमिया: हरी सब्जियां और खट्टे फल (विटामिन C) बढ़ाएं।",
        "diet_ok": "संतुलित आहार: आपका पोषण स्वस्थ विकास में सहायक है।",
        "analysis_msg": "एआई निदान शुरू करने के लिए विवरण दर्ज करें..."
    }
}

# --- SIDEBAR ---
st.sidebar.markdown(f"<h2 style='text-align: center;'>🤰 MaternalAI</h2>", unsafe_allow_html=True)
lang = st.sidebar.selectbox("🌐 Choose Language / மொழி / भाषा", ["English", "தமிழ்", "हिन्दी"])
c = content[lang]
st.sidebar.write("---")
st.sidebar.caption("System Status: Online 🟢")
st.sidebar.caption("Version: 2.0 (Aesthetic Update)")

# --- MAIN UI ---
st.markdown(f"<h1>{c['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #7f8c8d; font-size: 1.1rem;'>{c['tagline']}</p>", unsafe_allow_html=True)

# Layout Container
with st.container():
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown(f"<div class='main-card'><h3>📝 {c['p_info']}</h3>", unsafe_allow_html=True)
        name = st.text_input(c['lbl_name'])
        age = st.slider(c['lbl_age'], 15, 50, 25)
        hb = st.number_input(c['lbl_hb'], 5.0, 16.0, 11.0, step=0.1)
        bp = st.number_input(c['lbl_bp'], 80, 200, 120)
        weight = st.number_input(c['lbl_weight'], 35, 120, 60)
        week = st.number_input(c['lbl_week'], 1, 42, 12)
        
        btn_pressed = st.button(c['btn_calc'], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        st.markdown(f"<div class='main-card'><h3>📊 {c['results_hdr']}</h3>", unsafe_allow_html=True)
        if btn_pressed:
            # Animation effect
            with st.spinner('Analyzing clinical markers...'):
                time.sleep(1)
            
            # Predict
            features = np.array([[age, hb, bp, weight]])
            prob = model.predict_proba(features)[0][1]
            score = round(prob * 100, 2)
            
            # Show Score
            st.metric(label=c["risk_score"], value=f"{score}%")
            
            # Visual Feedback
            if score > 75:
                st.error(c["high"])
                st.balloons() if score < 10 else None # Don't balloon on high risk
            elif score > 40:
                st.warning(c["med"])
            else:
                st.success(c["low"])
                st.balloons()
            
            st.write("---")
            
            # Nutrition & Emergency
            st.markdown(f"#### 🍏 {c['nut_hdr']}")
            if hb < 9: st.info(c["anemia_s"])
            elif hb < 11: st.info(c["anemia_m"])
            else: st.success(c["diet_ok"])
            
            st.markdown(f"#### 🆘 {c['emg_hdr']}")
            if hb < 8 or bp > 155:
                st.error(c["emg_msg"])
            else:
                st.write(c["safe_msg"])
        else:
            st.info(c["analysis_msg"])
        st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.caption("© 2024 AI Maternal Health Support. Designed for rural accessibility and impact.")
