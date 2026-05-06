import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
from datetime import datetime
from fpdf import FPDF

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="AI Thyroid System", layout="wide")

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
    xgb = joblib.load("thyroid_model.pkl")
    rf = joblib.load("rf_model.pkl")
    explainer = shap.TreeExplainer(xgb)
    return xgb, rf, explainer

# ================================
# LOAD DATASET
# ================================
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset_Thyroid1.csv")

# ================================
# ACCURACY FUNCTION
# ================================
@st.cache_data
def get_metrics(model):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    df = load_data()
    X = df.drop("binaryClass", axis=1)
    y = df["binaryClass"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    return acc * 100, cm

# ================================
# LOGIN
# ================================
with open("users.json") as f:
    users = json.load(f)

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and str(users[u]) == str(p):
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid")
    st.stop()

# ================================
# LOAD EVERYTHING
# ================================
xgb_model, rf_model, explainer = load_models()

# ================================
# SIDEBAR (REAL ACCURACY)
# ================================
with st.sidebar:
    st.title("📊 Model Performance")

    xgb_acc, xgb_cm = get_metrics(xgb_model)
    rf_acc, rf_cm = get_metrics(rf_model)

    st.metric("XGBoost Accuracy", f"{xgb_acc:.2f}%")
    st.metric("Random Forest Accuracy", f"{rf_acc:.2f}%")

# ================================
# MAIN UI
# ================================
st.title("🧠 Thyroid AI Diagnosis System")

col1, col2, col3 = st.columns(3)

age = col1.slider("Age", 1, 100, 30)
tsh = col2.number_input("TSH", value=6.0)
fti = col3.number_input("FTI", value=50.0)

ratio = tsh / (fti + 0.001)

st.info(f"TSH/FTI Ratio: {ratio:.5f}")

# ================================
# PREDICTION
# ================================
if st.button("Predict"):

    input_df = pd.DataFrame([{
        'age': age,
        'sex': 0,
        'on thyroxine': 0,
        'query on thyroxine': 0,
        'on antithyroid medication': 0,
        'sick': 0,
        'pregnant': 0,
        'thyroid surgery': 0,
        'I131 treatment': 0,
        'query hypothyroid': 0,
        'query hyperthyroid': 0,
        'lithium': 0,
        'goitre': 0,
        'tumor': 0,
        'hypopituitary': 0,
        'psych': 0,
        'TSH measured': 1,
        'TSH': tsh,
        'T3 measured': 0,
        'TT4 measured': 0,
        'TT4': 0,
        'T4U measured': 0,
        'T4U': 0,
        'FTI measured': 1,
        'FTI': fti,
        'TSH_FTI_Ratio': ratio,
        'Age_Group': 0,
        'Symptom_Score': 0
    }])

    # Model choice auto best
    model = xgb_model if xgb_acc >= rf_acc else rf_model

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    if pred == 1:
        st.error(f"🚨 Positive ({prob:.2f}%)")
    else:
        st.success(f"✅ Negative ({100-prob:.2f}%)")

    # ================================
    # SHAP
    # ================================
    st.subheader("🔍 Explainable AI")

    shap_values = explainer(input_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# ================================
# MODEL COMPARISON
# ================================
st.subheader("📊 Model Comparison")

fig2, ax2 = plt.subplots()
ax2.bar(["XGBoost", "Random Forest"], [xgb_acc, rf_acc])
ax2.set_ylabel("Accuracy %")
st.pyplot(fig2)

# ================================
# CONFUSION MATRIX
# ================================
st.subheader("📉 Confusion Matrix")

fig3, ax3 = plt.subplots()
ax3.imshow(xgb_cm)
ax3.set_title("XGBoost Confusion Matrix")
st.pyplot(fig3)

# ================================
# GLOBAL SHAP
# ================================
st.subheader("🌍 Global Feature Importance")

df_sample = load_data().drop("binaryClass", axis=1).sample(100)
shap_vals = explainer(df_sample)

fig4, ax4 = plt.subplots()
shap.summary_plot(shap_vals, df_sample, show=False)
st.pyplot(fig4)
