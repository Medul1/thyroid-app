import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
import os
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
# LOAD MODEL + DATA (With Safety)
# ===============================
@st.cache_resource
def load_all():
    if not os.path.exists("thyroid_model.pkl") or not os.path.exists("cleaned_dataset_Thyroid1.csv"):
        return None, None, None
    
    model = joblib.load("thyroid_model.pkl")
    explainer = shap.TreeExplainer(model)
    df = pd.read_csv("cleaned_dataset_Thyroid1.csv")
    return model, explainer, df

model, explainer, df = load_all()

# File check
if model is None:
    st.error("Error: 'thyroid_model.pkl' অথবা 'cleaned_dataset_Thyroid1.csv' ফাইলটি খুঁজে পাওয়া যাচ্ছে না।")
    st.stop()

# ===============================
# LOAD USERS
# ===============================
try:
    with open("users.json") as f:
        users = json.load(f)
except Exception:
    st.error("Error: 'users.json' ফাইলটি পাওয়া যায়নি।")
    st.stop()

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
# FIXED: REAL ACCURACY CALCULATION
# ===============================
try:
    # মডেল যে কলামগুলো চায় সেগুলো CSV-তে আছে কি না চেক করা
    features = list(model.feature_names_in_)
    if all(col in df.columns for col in features) and 'target' in df.columns:
        X = df[features]
        y = df['target']
        pred_all = model.predict(X)
        real_acc = accuracy_score(y, pred_all) * 100
    else:
        # যদি কলামের নামে অমিল থাকে
        missing = [c for c in features if c not in df.columns]
        st.warning(f"CSV ফাইলে কিছু কলাম পাওয়া যায়নি: {missing}")
        real_acc = 0
except Exception as e:
    real_acc = 0
    st.sidebar.error(f"Accuracy Error: {e}")

# ===============================
# SIDEBAR (PRO UI)
# ===============================
with st.sidebar:
    st.title("🧠 ThyroPredict")
    st.markdown("AI Clinical Decision Support")
    st.markdown("### 📊 Model Performance")
    st.metric("Real Accuracy", f"{real_acc:.2f}%")
    st.markdown("### 📌 Normal Range")
    st.info("**TSH:** 0.4 – 4.0\n\n**FTI:** 60 – 160")

# ===============================
# MAIN UI
# ===============================
st.title("🧠 Thyroid Disease AI System")
st.markdown("---")

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
# PREDICTION (Fixed Feature Alignment)
# ===============================
if st.button("🚀 Run Diagnosis", use_container_width=True):
    # সব ফিচার মেইনটেইন করে ইনপুট তৈরি করা
    input_data = {col: 0 for col in model.feature_names_in_} # ডিফল্ট সব 0
    
    # ইউজার ইনপুট বসানো
    input_data.update({
        'age': age,
        'sex': sex_val,
        'TSH': tsh,
        'FTI': fti,
        'TSH_FTI_Ratio': ratio,
        'TSH measured': 1,
        'FTI measured': 1
    })
    
    input_df = pd.DataFrame([input_data])[model.feature_names_in_]

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100
    verdict = "Positive" if pred == 1 else "Negative"

    st.markdown("---")
    if pred == 1:
        st.error(f"🚨 Disease Detected (Positive)")
        confidence = prob
    else:
        st.success(f"✅ Healthy (Negative)")
        confidence = 100 - prob

    st.metric("Confidence", f"{confidence:.2f}%")

    # Explanation and SHAP
    st.markdown("### 🧠 AI Explanation")
    if tsh > 4: st.write("• TSH High: Hypothyroid signal detected.")
    if fti < 60: st.write("• FTI Low: Hormone levels below normal.")
    
    st.markdown("### 📊 Explainable AI Graph")
    shap_vals = explainer(input_df)
    fig, ax = plt.subplots(figsize=(12,6))
    shap.plots.waterfall(shap_vals[0], show=False)
    st.pyplot(fig)

    # PDF Report (Removed Bengali to avoid encoding error)
    def pdf_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Thyroid AI Diagnosis Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, f"Name: {name}", ln=True)
        pdf.cell(200, 10, f"Age: {age}", ln=True)
        pdf.cell(200, 10, f"Result: {verdict}", ln=True)
        pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
        return pdf.output(dest='S').encode('latin-1')

    st.download_button("📄 Download Report", data=pdf_report(), file_name="report.pdf")
