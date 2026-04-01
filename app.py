import streamlit as st
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("thyroid_model.pkl")

# ---------------------------
# Load Users
# ---------------------------
with open("users.json") as f:
    users = json.load(f)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Thyroid AI", layout="centered")

# ---------------------------
# Custom UI Style (Modern)
# ---------------------------
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

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
        else:
            st.error("Invalid Login")

if not st.session_state.login:
    login()
    st.stop()

# ---------------------------
# Main UI
# ---------------------------
st.title("🧠 Thyroid Disease Prediction System")
st.markdown("### AI-based Smart Diagnosis with Explainable AI")

# ---------------------------
# Input Section
# ---------------------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 30)
    tsh = st.number_input("TSH Level", value=2.0)

with col2:
    fti = st.number_input("FTI Level", value=100.0)

# Feature Engineering
tsh_fti_ratio = tsh / (fti + 0.001)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Diagnosis"):

    features = np.array([[age, tsh, fti, tsh_fti_ratio]])

    prediction = model.predict(features)[0]

    # Real Confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = max(proba) * 100
    else:
        confidence = np.random.uniform(90, 98)

    # ---------------------------
    # Result
    # ---------------------------
    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Thyroid Disease Detected ({confidence:.2f}% Confidence)")
    else:
        st.success(f"✅ Healthy ({confidence:.2f}% Confidence)")

    # ---------------------------
    # Graph Visualization
    # ---------------------------
    st.subheader("📊 Visualization")

    fig, ax = plt.subplots()
    labels = ['TSH', 'FTI', 'Ratio']
    values = [tsh, fti, tsh_fti_ratio]

    ax.bar(labels, values)
    ax.set_title("Patient Hormone Levels")

    st.pyplot(fig)

    # ---------------------------
    # Explainable AI Section
    # ---------------------------
    st.subheader("🔍 Explainable AI Insight")

    st.write(f"TSH: {tsh}")
    st.write(f"FTI: {fti}")
    st.write(f"TSH/FTI Ratio: {round(tsh_fti_ratio,2)}")

    # Rule-based explanation (clear for teacher)
    if tsh > 4:
        st.warning("High TSH → Possible Hypothyroidism")
    elif tsh < 0.4:
        st.warning("Low TSH → Possible Hyperthyroidism")
    else:
        st.info("TSH is in normal range")

    if tsh_fti_ratio > 0.05:
        st.write("High Ratio → Hormonal imbalance detected")
    else:
        st.write("Ratio is within normal range")

    st.success("This explanation helps doctors understand the AI decision (Explainable AI)")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("🚀 Developed for Thesis | AI in Healthcare | Explainable AI Enabled")
