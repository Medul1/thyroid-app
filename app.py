import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ১. টাইটেল ও সাবটাইটেল
st.set_page_config(page_title="AI Thyroid Diagnosis", layout="centered")
st.title("🏥 AI-based Smart Diagnosis System")
st.write("Enter patient data to get an instant prediction and explanation.")

# ২. ইনপুট সেকশন
age = st.slider("Age", 1, 100, 30)
tsh = st.number_input("TSH Level", value=6.00)
fti = st.number_input("FTI Level", value=50.00)

# ৩. মডেল ও এক্সপ্লেইনার লোড করা
@st.cache_resource
def load_resources():
    model = joblib.load('thyroid_model.pkl')
    explainer = shap.TreeExplainer(model)
    return model, explainer

try:
    model, explainer = load_resources()
    
    if st.button("Predict & Explain"):
        # মডেলের চাহিদা অনুযায়ী ২৮টি ফিচারের ডাটাফ্রেম
        data = {
            'age': [age], 'sex': [0], 'on thyroxine': [0], 'query on thyroxine': [0],
            'on antithyroid medication': [0], 'sick': [0], 'pregnant': [0], 
            'thyroid surgery': [0], 'I131 treatment': [0], 'query hypothyroid': [0],
            'query hyperthyroid': [0], 'lithium': [0], 'goitre': [0], 'tumor': [0],
            'hypopituitary': [0], 'psych': [0], 'TSH measured': [1], 'TSH': [tsh],
            'T3 measured': [0], 'TT4 measured': [0], 'TT4': [0], 'T4U measured': [0],
            'T4U': [0], 'FTI measured': [1], 'FTI': [fti],
            'TSH_FTI_Ratio': [tsh / fti if fti != 0 else 0],
            'Age_Group': [0], 'Symptom_Score': [0]
        }
        features = pd.DataFrame(data)
        
        # প্রেডিকশন
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] * 100

        # ফলাফল দেখানো
        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **RESULT: POSITIVE (Thyroid Disease Detected)**")
        else:
            st.success(f"✅ **RESULT: NEGATIVE (Healthy)**")
        st.info(f"**Confidence Level:** {prob:.2f}%")

        # ৪. SHAP Explanation (গ্রাফ দেখানো)
        st.subheader("🔍 Model Explanation (Why this result?)")
        st.write("This chart shows which factors influenced the AI's decision.")
        
        shap_values = explainer(features)
        
        # গ্রাফ তৈরি
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")
