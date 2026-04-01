import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import shap
import matplotlib.pyplot as plt

# ---------------------------
# Load Model & Explainer
# ---------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("thyroid_model.pkl")
    explainer = shap.TreeExplainer(model)
    return model, explainer

# ---------------------------
# Load Users
# ---------------------------
try:
    with open("users.json") as f:
        users = json.load(f)
except FileNotFoundError:
    st.error("users.json file not found. Please upload it to GitHub.")
    st.stop()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Thyroid AI", layout="centered")

# ---------------------------
# Login System
# ---------------------------
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    st.title("🔐 Login System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid Login")

if not st.session_state.login:
    login()
    st.stop()

# ---------------------------
# Main App UI
# ---------------------------
st.title("🧠 Thyroid Disease Prediction System")
st.markdown("### AI-based Smart Diagnosis Tool")

# ---------------------------
# Input Section
# ---------------------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 1, 100, 30)
    tsh = st.number_input("TSH Level", value=6.0)
with col2:
    fti = st.number_input("FTI Level", value=50.0)
    # Ratio calculation
    tsh_fti_ratio = tsh / (fti + 0.001)

# ---------------------------
# Prediction & Explanation
# ---------------------------
model, explainer = load_resources()

if st.button("Predict"):
    # মডেলের রিকোয়ারমেন্ট অনুযায়ী ২৮টি কলামের ডাটাফ্রেম তৈরি
    data = {
        'age': [age], 'sex': [0], 'on thyroxine': [0], 'query on thyroxine': [0],
        'on antithyroid medication': [0], 'sick': [0], 'pregnant': [0], 
        'thyroid surgery': [0], 'I131 treatment': [0], 'query hypothyroid': [0],
        'query hyperthyroid': [0], 'lithium': [0], 'goitre': [0], 'tumor': [0],
        'hypopituitary': [0], 'psych': [0], 'TSH measured': [1], 'TSH': [tsh],
        'T3 measured': [0], 'TT4 measured': [0], 'TT4': [0], 'T4U measured': [0],
        'T4U': [0], 'FTI measured': [1], 'FTI': [fti],
        'TSH_FTI_Ratio': [tsh_fti_ratio], 'Age_Group': [0], 'Symptom_Score': [0]
    }
    features = pd.DataFrame(data)

    # প্রেডিকশন
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1] * 100

    st.markdown("---")
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Thyroid Disease Detected ({prob:.2f}%)")
    else:
        st.success(f"✅ Healthy ({100-prob:.2f}%)")

    # ---------------------------
    # Explainable AI (SHAP)
    # ---------------------------
    st.markdown("---")
    st.subheader("🔍 Model Explanation (SHAP)")
    st.write("This chart explains how the AI analyzed the inputs.")
    
    shap_values = explainer(features)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False) # কোলাব নোটবুকের মতো গ্রাফ
    st.pyplot(fig)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Developed for Thesis Project | AI in Healthcare")
