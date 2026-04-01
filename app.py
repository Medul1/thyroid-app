import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("thyroid_model.pkl")

# Page config
st.set_page_config(page_title="Thyroid AI", layout="centered")

st.title("🧠 Thyroid Disease Prediction System")
st.write("AI-based smart diagnosis system")

# Input fields
age = st.slider("Age", 1, 100, 30)
tsh = st.number_input("TSH Level", value=2.0)
fti = st.number_input("FTI Level", value=100.0)

# Feature Engineering (same as your model)
tsh_fti_ratio = tsh / (fti + 0.001)

# Predict button
if st.button("Predict"):
    features = np.array([[age, tsh, fti, tsh_fti_ratio]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ Thyroid Disease Detected")
    else:
        st.success("✅ Healthy")

    # Explainable AI (simple logic)
    st.subheader("Explanation")
    st.write(f"TSH/FTI Ratio: {round(tsh_fti_ratio,2)}")

    if tsh > 4:
        st.write("High TSH → Possible Hypothyroid")
    elif tsh < 0.4:
        st.write("Low TSH → Possible Hyperthyroid")
    else:
        st.write("TSH in normal range")