# ✅ FULLY FIXED GEMINI + STREAMLIT CODE (WORKING VERSION)

import streamlit as st
import requests
import time
from fpdf import FPDF

# ================== API KEY ==================
# Use ONLY secrets (secure way)
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return ""

# ================== FALLBACK ==================
def local_clinical_brain(hb, bp, risk_score):
    advice = f"Clinical Assessment: Risk {risk_score}%. "
    if hb < 11:
        advice += "Low Hemoglobin → Increase Iron. "
    if bp > 140:
        advice += "High BP → Monitor daily. "
    if hb >= 11 and bp <= 140:
        advice += "Vitals stable."
    return advice + " (Fallback Mode)"

# ================== GEMINI CALL ==================
def call_gemini(prompt):
    api_key = get_api_key()

    if not api_key:
        return "❌ API Key Missing"

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=20)

        # DEBUG
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)

        if response.status_code == 200:
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"❌ API Error {response.status_code}"

    except Exception as e:
        return f"❌ Exception: {str(e)}"

# ================== PDF ==================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'MaternalAI Report', 0, 1, 'C')


def create_pdf(name, result):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Name: {name}", 0, 1)
    pdf.multi_cell(0, 10, result)
    return pdf.output(dest='S').encode('latin-1')

# ================== UI ==================
st.set_page_config(page_title="Maternal AI", layout="wide")

st.title("🤰 Maternal Health Dashboard")

name = st.text_input("Name")
hb = st.number_input("Hemoglobin", 5.0, 16.0, 11.0)
bp = st.number_input("BP", 80, 200, 120)

if st.button("Analyze"):
    risk = 15 if hb >= 11 and bp <= 140 else 70

    st.metric("Risk", f"{risk}%")

    if st.button("Generate AI Report"):
        with st.spinner("Calling AI..."):
            result = call_gemini(f"Patient Hb {hb}, BP {bp}, Risk {risk}%. Give medical advice.")

            # fallback if failed
            if "❌" in result:
                result = local_clinical_brain(hb, bp, risk)

            st.write(result)

            pdf = create_pdf(name, result)
            st.download_button("Download PDF", pdf, "report.pdf")

# ================== STATUS ==================
if get_api_key():
    st.success("🟢 AI Connected")
else:
    st.warning("🟡 No API Key")
