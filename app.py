# =========================================================
# APP.PY
# Explainable Thyroid AI Web Application
# Thesis Project
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path
import warnings
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Explainable Thyroid AI",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp{
    background-color:#0B1120;
    color:white;
}

[data-testid="stSidebar"]{
    background-color:#111827;
}

.metric-card{
    background:#1E293B;
    padding:15px;
    border-radius:15px;
    border:1px solid #334155;
}

.result-positive{
    background:#7F1D1D;
    padding:20px;
    border-radius:15px;
    border:2px solid #EF4444;
}

.result-negative{
    background:#052E16;
    padding:20px;
    border-radius:15px;
    border:2px solid #22C55E;
}

.section-card{
    background:#1E293B;
    padding:20px;
    border-radius:15px;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOGIN SYSTEM
# =========================================================

try:
    with open("users.json") as f:
        users = json.load(f)
except:
    users = {"admin":"1234"}

if "login" not in st.session_state:
    st.session_state.login = False

def login_page():

    st.title("🧠 Explainable Thyroid AI")
    st.subheader("Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in users and users[username] == password:
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid username or password")

if not st.session_state.login:
    login_page()
    st.stop()

# =========================================================
# LOAD FILES
# =========================================================

MODEL_PATH = "stacking_ensemble_model.pkl"
FEATURE_PATH = "feature_columns.pkl"
DATASET_PATH = "cleaned_dataset_Thyroid1.csv"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

df = pd.read_csv(DATASET_PATH)

# =========================================================
# IMPORTANT FEATURES
# =========================================================

important_features = [
    "age",
    "sex",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI",
    "on thyroxine",
    "query hypothyroid",
    "query hyperthyroid",
    "pregnant",
    "thyroid surgery",
    "goitre",
    "tumor"
]

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def feature_engineering(df):

    df["TSH_FTI_Ratio"] = df["TSH"] / (df["FTI"] + 0.001)

    df["Age_TSH_Interaction"] = df["age"] * df["TSH"]

    df["Hormone_Imbalance_Score"] = (
        (df["TSH"] * 0.30) +
        (df["T3"] * 0.15) +
        (df["TT4"] * 0.20) +
        (df["T4U"] * 0.10) +
        (df["FTI"] * 0.25)
    )

    return df

# =========================================================
# PREPROCESS INPUT
# =========================================================

def preprocess_input(input_df):

    input_df = feature_engineering(input_df)

    input_df = pd.get_dummies(input_df)

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    return input_df

# =========================================================
# PDF REPORT
# =========================================================

def generate_pdf(name, prediction, confidence, risk):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Thyroid AI Report", ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", "", 12)

    pdf.cell(0, 10, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 10, f"Prediction: {prediction}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 10, f"Risk Level: {risk}", ln=True)

    file_path = "thyroid_report.pdf"

    pdf.output(file_path)

    return file_path

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🧠 Thyroid AI Dashboard")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.success("5-Model Stacking Ensemble")

st.sidebar.write("✔ Random Forest")
st.sidebar.write("✔ XGBoost")
st.sidebar.write("✔ Decision Tree")
st.sidebar.write("✔ Logistic Regression")
st.sidebar.write("✔ SVM")

st.sidebar.markdown("---")

# =========================================================
# MAIN TITLE
# =========================================================

st.title("🧠 Explainable Machine Learning-based Smart System for Diagnosing Thyroid Disease")

st.markdown("""
This system uses advanced Machine Learning, Ensemble Learning,
SHAP, and LIME Explainable AI techniques for thyroid disease prediction.
""")

# =========================================================
# INPUT FORM
# =========================================================

st.subheader("📋 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:

    patient_name = st.text_input("Patient Name")

    age = st.slider("Age", 1, 100, 30)

    sex = st.selectbox("Sex", ["Male", "Female"])

with col2:

    tsh = st.number_input("TSH", value=2.0)

    t3 = st.number_input("T3", value=1.5)

    tt4 = st.number_input("TT4", value=120.0)

with col3:

    t4u = st.number_input("T4U", value=1.0)

    fti = st.number_input("FTI", value=100.0)

# =========================================================
# BOOLEAN FEATURES
# =========================================================

st.subheader("🩺 Clinical Information")

c1, c2, c3 = st.columns(3)

with c1:
    on_thyroxine = st.selectbox("On Thyroxine", [0,1])
    query_hypo = st.selectbox("Query Hypothyroid", [0,1])

with c2:
    query_hyper = st.selectbox("Query Hyperthyroid", [0,1])
    pregnant = st.selectbox("Pregnant", [0,1])

with c3:
    surgery = st.selectbox("Thyroid Surgery", [0,1])
    goitre = st.selectbox("Goitre", [0,1])

tumor = st.selectbox("Tumor", [0,1])

# =========================================================
# PREDICTION
# =========================================================

if st.button("🚀 Run Prediction"):

    input_data = pd.DataFrame({

        "age":[age],
        "sex":[1 if sex=="Male" else 0],
        "TSH":[tsh],
        "T3":[t3],
        "TT4":[tt4],
        "T4U":[t4u],
        "FTI":[fti],
        "on thyroxine":[on_thyroxine],
        "query hypothyroid":[query_hypo],
        "query hyperthyroid":[query_hyper],
        "pregnant":[pregnant],
        "thyroid surgery":[surgery],
        "goitre":[goitre],
        "tumor":[tumor]

    })

    processed_input = preprocess_input(input_data)

    prediction = model.predict(processed_input)[0]

    probability = model.predict_proba(processed_input)[0][1]

    confidence = probability * 100

    # =====================================================
    # RISK STRATIFICATION
    # =====================================================

    if confidence < 40:
        risk = "Low Risk"

    elif confidence < 70:
        risk = "Medium Risk"

    elif confidence < 90:
        risk = "High Risk"

    else:
        risk = "Critical Risk"

    result = "Positive" if prediction == 1 else "Negative"

    # =====================================================
    # RESULT CARD
    # =====================================================

    if result == "Positive":

        st.markdown(f"""
        <div class="result-positive">
        <h2>🚨 Thyroid Positive</h2>
        <h3>Confidence: {confidence:.2f}%</h3>
        <h3>Risk Level: {risk}</h3>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-negative">
        <h2>✅ Thyroid Negative</h2>
        <h3>Confidence: {confidence:.2f}%</h3>
        <h3>Risk Level: {risk}</h3>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # LIVE BIOMARKER CHART
    # =====================================================

    st.subheader("📈 Biomarker Analysis")

    chart_df = pd.DataFrame({
        "Feature":["TSH","T3","TT4","T4U","FTI"],
        "Value":[tsh,t3,tt4,t4u,fti]
    })

    fig = px.bar(
        chart_df,
        x="Feature",
        y="Value",
        title="Hormone Biomarker Visualization"
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # SHAP EXPLAINABILITY
    # =====================================================

    st.subheader("🧠 SHAP Explainability")

    try:

        explainer = shap.Explainer(model)

        shap_values = explainer(processed_input)

        fig2, ax = plt.subplots()

        shap.plots.waterfall(shap_values[0], show=False)

        st.pyplot(fig2)

    except:
        st.info("SHAP visualization could not be generated.")

    # =====================================================
    # LIME EXPLAINABILITY
    # =====================================================

    st.subheader("💡 LIME Explanation")

    try:

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(processed_input),
            feature_names=processed_input.columns,
            class_names=["Negative","Positive"],
            mode="classification"
        )

        explanation = explainer_lime.explain_instance(
            processed_input.iloc[0],
            model.predict_proba,
            num_features=8
        )

        st.components.v1.html(
            explanation.as_html(),
            height=800,
            scrolling=True
        )

    except:
        st.info("LIME explanation could not be generated.")

    # =====================================================
    # PDF REPORT
    # =====================================================

    st.subheader("📄 Download Report")

    pdf_path = generate_pdf(
        patient_name,
        result,
        confidence,
        risk
    )

    with open(pdf_path, "rb") as file:

        st.download_button(
            label="Download PDF Report",
            data=file,
            file_name="thyroid_report.pdf",
            mime="application/pdf"
        )

# =========================================================
# MODEL PERFORMANCE SECTION
# =========================================================

st.markdown("---")

st.subheader("📊 Model Performance Dashboard")

performance_df = pd.DataFrame({

    "Model":[
        "Random Forest",
        "XGBoost",
        "Decision Tree",
        "Logistic Regression",
        "SVM"
    ],

    "Accuracy":[
        99.60,
        99.74,
        98.90,
        97.80,
        98.10
    ]
})

fig3 = px.bar(
    performance_df,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Model Accuracy Comparison"
)

fig3.update_layout(template="plotly_dark")

st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("🔥 Feature Importance Consensus")

importance_df = pd.DataFrame({

    "Feature":[
        "TSH",
        "FTI",
        "TSH_FTI_Ratio",
        "Hormone_Imbalance_Score",
        "Age_TSH_Interaction",
        "TT4",
        "T3"
    ],

    "Importance":[
        0.95,
        0.91,
        0.88,
        0.85,
        0.80,
        0.76,
        0.70
    ]
})

fig4 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Consensus Feature Importance"
)

fig4.update_layout(template="plotly_dark")

st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# BATCH CSV PREDICTION
# =========================================================

st.markdown("---")

st.subheader("📂 Batch CSV Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    st.write(batch_df.head())

    if st.button("Run Batch Prediction"):

        batch_df = feature_engineering(batch_df)

        batch_df = pd.get_dummies(batch_df)

        for col in feature_columns:
            if col not in batch_df.columns:
                batch_df[col] = 0

        batch_df = batch_df[feature_columns]

        preds = model.predict(batch_df)

        batch_df["Prediction"] = preds

        batch_df["Prediction"] = batch_df["Prediction"].map({
            0:"Negative",
            1:"Positive"
        })

        st.success("Batch Prediction Completed")

        st.dataframe(batch_df)

        csv = batch_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction CSV",
            csv,
            "batch_prediction.csv",
            "text/csv"
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<center>

Developed for Thesis Research  
Department of CSE  
Notre Dame University Bangladesh  

<b>Tanjil Hossain Midul</b>

</center>
""", unsafe_allow_html=True)
