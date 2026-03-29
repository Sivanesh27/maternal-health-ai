import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Model
model = joblib.load('maternal_health_model.pkl')

# Multi-lingual Content
content = {
    "English": {
        "title": "Maternal Health AI Support",
        "sidebar": "Navigation & Settings",
        "p_info": "Patient Information",
        "results": "Clinical Assessment",
        "risk_lvl": "Risk Level",
        "score": "Predictive Risk Score",
        "high": "🔴 HIGH RISK", "med": "🟡 MODERATE RISK", "low": "🟢 LOW RISK",
        "advice_header": "AI Recommendations",
        "emergency_msg": "🚨 EMERGENCY: Immediate medical intervention required!",
        "safe_msg": "✅ Stable condition. Follow routine prenatal care.",
        "anemia_s": "Severe Anemia: Iron rich diet (Spinach, Meat, Dates) + Supplements.",
        "anemia_m": "Mild Anemia: Increase green vegetables and Vitamin C (Citrus fruits).",
        "diet_ok": "Balanced Diet: Maintain current healthy nutrition habits."
    },
    "தமிழ்": {
        "title": "தாய்வழி சுகாதார AI ஆதரவு",
        "sidebar": "அமைப்புகள்",
        "p_info": "நோயாளி தகவல்",
        "results": "மருத்துவ மதிப்பீடு",
        "risk_lvl": "ஆபத்து நிலை",
        "score": "ஆபத்து மதிப்பெண்",
        "high": "🔴 அதிக ஆபத்து", "med": "🟡 மிதமான ஆபத்து", "low": "🟢 குறைந்த ஆபத்து",
        "advice_header": "AI பரிந்துரைகள்",
        "emergency_msg": "🚨 அவசரம்: உடனடியாக மருத்துவமனைக்குச் செல்லுங்கள்!",
        "safe_msg": "✅ நிலைமை சீராக உள்ளது. வழக்கமான பரிசோதனைகளைத் தொடரவும்.",
        "anemia_s": "கடுமையான ரத்தச்சோகை: இரும்புச்சத்து நிறைந்த உணவு + மாத்திரைகள் அவசியம்.",
        "anemia_m": "லேசான ரத்தச்சோகை: பச்சை காய்கறிகள் மற்றும் வைட்டமின் C உணவுகளை அதிகரிக்கவும்.",
        "diet_ok": "சீரான உணவு: தற்போதைய ஆரோக்கியமான உணவு முறையைத் தொடரவும்."
    },
    "हिन्दी": {
        "title": "मातृ स्वास्थ्य एआई सहायता",
        "sidebar": "नेविगेशन और सेटिंग्स",
        "p_info": "रोगी की जानकारी",
        "results": "नैदानिक मूल्यांकन",
        "risk_lvl": "जोखिम स्तर",
        "score": "जोखिम स्कोर",
        "high": "🔴 उच्च जोखिम", "med": "🟡 मध्यम जोखिम", "low": "🟢 कम जोखिम",
        "advice_header": "एआई सिफारिशें",
        "emergency_msg": "🚨 आपातकाल: तत्काल चिकित्सा हस्तक्षेप की आवश्यकता है!",
        "safe_msg": "✅ स्थिति स्थिर है। नियमित देखभाल जारी रखें।",
        "anemia_s": "गंभीर एनीमिया: आयरन युक्त आहार + सप्लीमेंट्स लें।",
        "anemia_m": "हल्का एनीमिया: हरी सब्जियां और विटामिन C बढ़ाएं।",
        "diet_ok": "संतुलित आहार: स्वस्थ पोषण बनाए रखें।"
    }
}

st.set_page_config(page_title="Maternal AI", page_icon="🤰", layout="wide")

# Sidebar
st.sidebar.title("🤰 MaternalAI")
lang = st.sidebar.selectbox("Select Language / மொழியைத் தேர்ந்தெடுக்கவும்", ["English", "தமிழ்", "हिन्दी"])
c = content[lang]

# Main UI
st.title(c["title"])
st.write("---")

col_inp, col_res = st.columns([1, 1], gap="large")

with col_inp:
    st.subheader(f"📋 {c['p_info']}")
    name = st.text_input("Name / பெயர்")
    age = st.slider("Age / வயது", 15, 50, 25)
    hb = st.number_input("Hemoglobin (Hb) g/dL", 5.0, 16.0, 11.0, step=0.1)
    bp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
    weight = st.number_input("Weight (kg)", 35, 120, 60)
    
    predict_clicked = st.button("Generate Health Report", use_container_width=True)

with col_res:
    st.subheader(f"📊 {c['results']}")
    if predict_clicked:
        # Prediction
        features = np.array([[age, hb, bp, weight]])
        prob = model.predict_proba(features)[0][1]
        score = round(prob * 100, 2)
        
        # Display Metrics
        st.metric(label=c["score"], value=f"{score}%")
        st.progress(score / 100)
        
        if score > 75:
            st.error(c["high"])
        elif score > 40:
            st.warning(c["med"])
        else:
            st.success(c["low"])
            
        st.markdown(f"### 💡 {c['advice_header']}")
        
        # Clinical Logic for advice
        if hb < 9: st.info(c["anemia_s"])
        elif hb < 11: st.info(c["anemia_m"])
        else: st.success(c["diet_ok"])
        
        if hb < 8 or bp > 155:
            st.error(c["emergency_msg"])
        else:
            st.write(c["safe_msg"])
    else:
        st.info("Enter details and click 'Generate Health Report' to see AI analysis.")

st.sidebar.markdown("---")
st.sidebar.caption("Rural Health Decision Support v1.0")
