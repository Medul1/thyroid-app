import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
from datetime import datetime
from fpdf import FPDF
from sklearn.metrics import accuracy_score

# ===============================
# PAGE CONFIG (UI)
# ===============================
st.set_page_config(
    page_title="ThyroPredict AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# LOAD MODEL + DATA
# ===============================
@st.cache_resource
def load_all():
    model = joblib.load("thyroid_model.pkl")
    explainer = shap.TreeExplainer(model)

    df = pd.read_csv("cleaned_dataset_Thyroid1.csv")
    return model, explainer, df

model, explainer, df = load_all()

# ===============================
# LOAD USERS
# ===============================
with open("users.json") as f:
    users = json.load(f)

# ===============================
# LOGIN SYSTEM
# ===============================
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 AI Medical Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and str(users[u]) == str(p):
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid Login")

if not st.session_state.login:
    login()
    st.stop()

# ===============================
# REAL ACCURACY CALCULATION
# ===============================
# Model required feature list
required_features = list(model.feature_names_in_)

# Missing feature automatically add
for col in required_features:
    if col not in df.columns:
        df[col] = 0

# Extra feature remove + correct order maintain
X = df[required_features]
y = df['target']

pred_all = model.predict(X)
real_acc = accuracy_score(y, pred_all) * 100

# ===============================
# SIDEBAR (PRO UI)
# ===============================
with st.sidebar:
    st.title("🧠 ThyroPredict")
    st.markdown("AI Clinical Decision Support")

    st.markdown("### 📊 Model Performance")
    st.metric("Real Accuracy", f"{real_acc:.2f}%")

    st.markdown("### 📌 Normal Range")
    st.info("""
    **TSH:** 0.4 – 4.0  
    **FTI:** 60 – 160  
    **Ratio:** 0.003 – 0.067
    """)

# ===============================
# MAIN TITLE
# ===============================
st.title("🧠 Thyroid Disease AI System")
st.markdown("---")

# ===============================
# INPUT SECTION
# ===============================
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name", "Patient_01")
    age = st.slider("Age", 1, 100, 30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex_val = 1 if sex == "Male" else 0

with col2:
    tsh = st.number_input("TSH", value=6.0)
    fti = st.number_input("FTI", value=50.0)

ratio = tsh / (fti + 0.001)

st.markdown(f"### 🔢 TSH/FTI Ratio: `{ratio:.4f}`")

# ===============================
# PREDICTION
# ===============================
if st.button("🚀 Run Diagnosis", use_container_width=True):

    input_df = pd.DataFrame([{
        'age': age, 'sex': sex_val, 'on thyroxine': 0,
        'query on thyroxine': 0, 'on antithyroid medication': 0,
        'sick': 0, 'pregnant': 0, 'thyroid surgery': 0,
        'I131 treatment': 0, 'query hypothyroid': 0,
        'query hyperthyroid': 0, 'lithium': 0,
        'goitre': 0, 'tumor': 0, 'hypopituitary': 0,
        'psych': 0, 'TSH measured': 1,
        'TSH': tsh, 'T3 measured': 0,
        'TT4 measured': 0, 'TT4': 0,
        'T4U measured': 0, 'T4U': 0,
        'FTI measured': 1, 'FTI': fti,
        'TSH_FTI_Ratio': ratio,
        'Age_Group': 0,
        'Symptom_Score': 0
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    verdict = "Positive" if pred == 1 else "Negative"

    st.markdown("---")

    # ===========================
    # RESULT UI
    # ===========================
    if pred == 1:
        st.error(f"🚨 Disease Detected (Positive)")
        confidence = prob
    else:
        st.success(f"✅ Healthy (Negative)")
        confidence = 100 - prob

    st.metric("Confidence", f"{confidence:.2f}%")

    # ===========================
    # SMART EXPLANATION (NEW 🔥)
    # ===========================
    st.markdown("### 🧠 AI Explanation")

    explanation = []

    if tsh > 4:
        explanation.append("TSH বেশি → Hypothyroid signal")

    if fti < 60:
        explanation.append("FTI কম → Hormone কম")

    if ratio > 0.07:
        explanation.append("Ratio বেশি → Strong imbalance")

    if confidence < 70:
        explanation.append("⚠️ Confidence কম কারণ data borderline")

    if explanation:
        for e in explanation:
            st.write("•", e)
    else:
        st.write("Normal pattern detected")

    # ===========================
    # SHAP GRAPH (BIG SIZE 🔥)
    # ===========================
    st.markdown("### 📊 Explainable AI Graph")

    shap_vals = explainer(input_df)

    fig, ax = plt.subplots(figsize=(12,6))
    shap.plots.waterfall(shap_vals[0], show=False)
    st.pyplot(fig)

    # ===========================
    # PDF REPORT
    # ===========================
    def pdf_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, "Thyroid AI Report", ln=True, align='C')
        pdf.ln(10)

        pdf.cell(200, 10, f"Name: {name}", ln=True)
        pdf.cell(200, 10, f"Age: {age}", ln=True)
        pdf.cell(200, 10, f"TSH: {tsh}", ln=True)
        pdf.cell(200, 10, f"FTI: {fti}", ln=True)

        pdf.cell(200, 10, f"Result: {verdict}", ln=True)
        pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)

        return pdf.output(dest='S').encode('latin-1')

    st.download_button(
        "📄 Download Report",
        data=pdf_report(),
        file_name="report.pdf"
    )
