import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import json

# ---------------------------------------------------------
# ১. গ্লোবাল কনফিগারেশন এবং রিসোর্স লোডিং
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Thyroid Diagnoser | Research Project",
    layout="wide", # এটি অ্যাপটিকে আরও আধুনিক এবং প্রশস্ত দেখাবে
    initial_sidebar_state="expanded"
)

# মডেল এবং SHAP এক্সপ্লেইনার লোড করা (Cache ব্যবহার করে স্পিড বাড়ানোর জন্য)
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('thyroid_model.pkl')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        st.error(f"মডেল লোড করতে সমস্যা হয়েছে। নিশ্চিত করুন ফাইলটি আছে। Error: {e}")
        st.stop()

# ইউজার লিস্ট লোড করা
try:
    with open("users.json") as f:
        users = json.load(f)
except Exception:
    st.error("users.json ফাইলটি পাওয়া যায়নি।")
    st.stop()

# ---------------------------------------------------------
# ২. লগইন সিস্টেম (পূর্বের মতোই কাজ করবে)
# ---------------------------------------------------------
if "login" not in st.session_state:
    st.session_state.login = False

def login_ui():
    st.title("🔐 AI Healthcare Portal Access")
    username = st.text_input("Username", placeholder="e.g., researcher01")
    password = st.text_input("Password", type="password", placeholder="••••••••")

    if st.button("Access Portal", type="primary"):
        # পাসওয়ার্ড স্ট্রিং এবং সংখ্যা দুইভাবেই চেক করার ব্যবস্থা
        if username in users and str(users[username]) == str(password):
            st.session_state.login = True
            st.rerun()
        else:
            st.error("অ্যাক্সেস প্রত্যাখ্যান করা হয়েছে: অবৈধ ইউজারনেম বা পাসওয়ার্ড।")

if not st.session_state.login:
    login_ui()
    st.stop() # লগইন না হওয়া পর্যন্ত অ্যাপের বাকি অংশ দেখাবে না

# ---------------------------------------------------------
# ৩. মেইন অ্যাপ ইন্টারফেস (লগইন সফল হলে এটি দেখাবে)
# ---------------------------------------------------------

# মডেল এবং এক্সপ্লেইনার লোড করা
model, explainer = load_resources()

# টাইটেল এবং সাইডবার
st.title("🧠 Advanced Thyroid Disease Intelligence System")
st.markdown("---")

with st.sidebar:
    st.image("https://cdn.icon-icons.com/icons2/2107/PNG/512/medical_icon_130384.png", width=100) # একটি মেডিকেল আইকন
    st.header("Diagnosis Dashboard")
    st.info("Fill patient details and click on 'Execute Intelligence' below.")
    st.markdown("**User:** Researcher | Bangladesh")
    st.write("Project: AI-based Healthcare Diagnosis")

# ---------------------------------------------------------
# ৪. ইনপুট সেকশন (আরও আধুনিক লেআউট)
# ---------------------------------------------------------
st.subheader("📋 Enter Patient Clinical Parameters")

# ইনপুটগুলোকে আমরা দুইটি কলামে ভাগ করছি
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    age = st.slider("Age of Patient (Years)", 1, 100, 30, help="Slide to select patient age.")
    sex_input = st.selectbox("Patient Sex", ["Female", "Male"], index=0)
    sex = 1 if sex_input == "Male" else 0 # 1=Male, 0=Female

with col2:
    tsh = st.number_input("TSH Level (mIU/L)", value=6.0, format="%.2f", help="Thyroid-Stimulating Hormone.")
    fti = st.number_input("FTI Level (μg/dL)", value=50.0, format="%.2f", help="Free Thyroxine Index.")
    # TSH/FTI অনুপাত ক্যালকুলেশন
    tsh_fti_ratio = tsh / (fti + 0.001)

with col3:
    st.info("ℹ️ General Range Info")
    st.write("- **TSH Normal:** 0.45 - 4.5 mIU/L")
    st.write("- **FTI Normal:** 85 - 160 μg/dL")
    st.write("- **Age:** Higher age can affect values.")

# ---------------------------------------------------------
# ৫. প্রেডিকশন এবং রেজাল্ট সেকশন (কালার কার্ড এবং মেট্রিক্স)
# ---------------------------------------------------------
st.markdown("---")
# Execute বাটনটিকে বড় এবং প্রমিনেন্ট করা হয়েছে
if st.button("Execute AI Diagnosis", type="primary", use_container_width=True):
    with st.spinner("AI is analyzing complex patterns..."):
        # মডেলের需求 অনুযায়ী ২৮টি কলামের ডাটাফ্রেম তৈরি
        data_dict = {
            'age': [age], 'sex': [sex], 'on thyroxine': [0], 'query on thyroxine': [0],
            'on antithyroid medication': [0], 'sick': [0], 'pregnant': [0], 
            'thyroid surgery': [0], 'I131 treatment': [0], 'query hypothyroid': [0],
            'query hyperthyroid': [0], 'lithium': [0], 'goitre': [0], 'tumor': [0],
            'hypopituitary': [0], 'psych': [0], 'TSH measured': [1], 'TSH': [tsh],
            'T3 measured': [0], 'TT4 measured': [0], 'TT4': [0], 'T4U measured': [0],
            'T4U': [0], 'FTI measured': [1], 'FTI': [fti],
            'TSH_FTI_Ratio': [tsh_fti_ratio], 'Age_Group': [0], 'Symptom_Score': [0]
        }
        features = pd.DataFrame(data_dict)

        # প্রেডিকশন এবং প্রবাবিলিটি ক্যালকুলেশন
        prediction = model.predict(features)[0]
        # রোগ হওয়ার সম্ভাবনা (Probability of Positive Class)
        prob_positive = model.predict_proba(features)[0][1] * 100
        # সুস্থ থাকার সম্ভাবনা
        prob_negative = 100 - prob_positive

        st.subheader("📊 Diagnostic Summary")
        
        # কার্ড ইন্টারফেস তৈরি (ফলাফল অনুযায়ী কালার বদলাবে)
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.error("🚨 THYROID DISEASE DETECTED")
                st.metric(label="System Verdict", value="Positive")
                # রোগ হওয়ার সম্ভাবনা দেখাবে
                st.metric(label="Probability Score", value=f"{prob_positive:.1f}%")
            else:
                st.success("✅ HEALTHY (NO DISEASE DETECTED)")
                st.metric(label="System Verdict", value="Negative")
                # সুস্থ থাকার সম্ভাবনা দেখাবে
                st.metric(label="Probability Score", value=f"{prob_negative:.1f}%")
        
        with res_col2:
            st.info("💡 Clinical Interpretation")
            
            # সাধারণ ক্লিনিক্যাল ইন্টারপ্রিটেশন লজিক
            if tsh > 4.5:
                st.warning("⚠️ High TSH suggests Hypothyroidism (Underactive).")
            elif tsh < 0.45:
                st.warning("⚠️ Low TSH suggests Hyperthyroidism (Overactive).")
            else:
                st.write("✓ TSH is within normal clinical range.")
                
            if fti < 85:
                st.warning("⚠️ Low FTI suggests Low Thyroxine Levels.")
            elif fti > 160:
                st.warning("⚠️ High FTI suggests Excess Thyroxine Levels.")
            else:
                st.write("✓ FTI is within normal clinical range.")

        # ---------------------------------------------------------
        # ৬. Explainable AI (SHAP Plot)
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("🔍 Local Model Interpretation (Why did AI make this decision?)")
        st.write("This waterfall plot shows how much each factor (feature) pushed the final decision from the system's baseline.")
        
        # SHAP ভ্যালু ক্যালকুলেশন
        shap_values = explainer(features)
        
        # ম্যাটপ্লটলিব ব্যবহার করে গ্রাফটি সুন্দরভাবে রেন্ডার করা (কোলাব নোটবুকের মতো)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
        st.caption("**How to read:** Factors in RED push the decision towards 'Positive', factors in BLUE push it towards 'Healthy'.")

# ---------------------------------------------------------
# ৭. Footer
# ---------------------------------------------------------
# ---------------------------------------------------------
# ৭. Footer (Fixing the TypeError)
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
    Developed for Thesis Project | Artificial Intelligence in Healthcare | Researcher Midul | Bangladesh
    </div>
    """, 
    unsafe_allow_html=True  # এখানে stdio এর বদলে html হবে
)
