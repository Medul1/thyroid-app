import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ১. টাইটেল ও সাবটাইটেল
st.set_page_config(page_title="AI Thyroid Diagnosis", layout="centered")
st.title("🏥 AI-based Smart Diagnosis System")
st.write("Enter patient data to get an instant prediction.")

# ২. ইনপুট সেকশন
age = st.slider("Age", 1, 100, 30)
tsh = st.number_input("TSH Level", value=6.00)
fti = st.number_input("FTI Level", value=50.00)

# অটোমেটিক রেশিও ক্যালকুলেশন
tsh_fti_ratio = tsh / fti if fti != 0 else 0

# ৩. মডেল লোড করা
@st.cache_resource
def get_model():
    return joblib.load('thyroid_model.pkl')

try:
    model = get_model()
    
    if st.button("Predict"):
        # আপনার মডেলে যে ৯টি ফিচার ছিল, সেগুলো ঠিক এই সিরিয়ালে সাজাতে হবে
        # আপনার গ্রাফ অনুযায়ী সিরিয়াল: TT4, TSH, TSH_FTI_Ratio, age, FTI, on thyroxine, sex, thyroid surgery, goitre
        
        # আমরা TT4 এবং অন্যান্য মাইনর ফিচারে ডিফল্ট মান (০) দিচ্ছি যা প্রেডিকশনে বড় প্রভাব ফেলবে না
        feature_dict = {
            'TT4': [0],
            'TSH': [tsh],
            'TSH_FTI_Ratio': [tsh_fti_ratio],
            'age': [age],
            'FTI': [fti],
            'on thyroxine': [0],
            'sex': [0],
            'thyroid surgery': [0],
            'goitre': [0]
        }
        
        features = pd.DataFrame(feature_dict)
        
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
    st.error(f"মডেলে কলামের সমস্যা হচ্ছে। এরর: {e}")
