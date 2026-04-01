import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ১. টাইটেল ও পেজ সেটআপ
st.set_page_config(page_title="AI Thyroid Diagnosis", layout="centered")
st.title("🏥 AI-based Smart Diagnosis System")
st.write("Enter patient data to get an instant prediction.")

# ২. ইনপুট সেকশন (ব্যবহারকারীর কাছ থেকে ৩টি ইনপুট নেওয়া হচ্ছে)
age = st.slider("Age", 1, 100, 30)
tsh = st.number_input("TSH Level", value=6.00)
fti = st.number_input("FTI Level", value=50.00)

# ৩. মডেল লোড করা
@st.cache_resource
def get_model():
    return joblib.load('thyroid_model.pkl')

try:
    model = get_model()
    
    if st.button("Predict"):
        # মডেলের চাহিদা অনুযায়ী ২৮টি ফিচারের ডিকশনারি (সঠিক সিরিয়ালে)
        data = {
            'age': [age],
            'sex': [0],
            'on thyroxine': [0],
            'query on thyroxine': [0],
            'on antithyroid medication': [0],
            'sick': [0],
            'pregnant': [0],
            'thyroid surgery': [0],
            'I131 treatment': [0],
            'query hypothyroid': [0],
            'query hyperthyroid': [0],
            'lithium': [0],
            'goitre': [0],
            'tumor': [0],
            'hypopituitary': [0],
            'psych': [0],
            'TSH measured': [1], # TSH ইনপুট দেওয়া হচ্ছে তাই এটি ১
            'TSH': [tsh],
            'T3 measured': [0],
            'TT4 measured': [0],
            'TT4': [0],
            'T4U measured': [0],
            'T4U': [0],
            'FTI measured': [1], # FTI ইনপুট দেওয়া হচ্ছে তাই এটি ১
            'FTI': [fti],
            'TSH_FTI_Ratio': [tsh / fti if fti != 0 else 0],
            'Age_Group': [0],
            'Symptom_Score': [0]
        }
        
        # ডাটাফ্রেম তৈরি
        features = pd.DataFrame(data)
        
        # প্রেডিকশন
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] * 100

        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **RESULT: POSITIVE (Thyroid Disease Detected)**")
        else:
            st.success(f"✅ **RESULT: NEGATIVE (Healthy)**")
            
        st.info(f"**Confidence Level:** {prob:.2f}%")

except Exception as e:
    st.error(f"Error: {e}")
