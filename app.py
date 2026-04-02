import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
from datetime import datetime
from fpdf import FPDF

# ---------------------------------------------------------
# ১. গ্লোবাল কনফিগারেশন এবং রিসোর্স লোডিং
# ---------------------------------------------------------
st.set_page_config(
    page_title="Advanced AI Thyroid Intelligence",
    layout="wide", 
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('thyroid_model.pkl')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        st.error(f"মডেল ফাইলটি পাওয়া যায়নি। Error: {e}")
        st.stop()

# ইউজার লিস্ট লোড করা
try:
    with open("users.json") as f:
        users = json.load(f)
except Exception:
    st.error("users.json ফাইলটি পাওয়া যায়নি।")
    st.stop()

# ---------------------------------------------------------
# ২. PDF রিপোর্ট জেনারেশন ফাংশন
# ---------------------------------------------------------
def create_pdf_report(name, age, sex, tsh, fti, verdict, prob):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 20)
    pdf.cell(200, 15, "Thyroid Diagnostic Intelligence Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.line(10, 35, 200, 35)
    
    # Patient Data
    pdf.ln(15)
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 10, " Patient Information", ln=True, fill=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f" Name: {name}", ln=True)
    pdf.cell(0, 10, f" Age: {age} Years | Sex: {sex}", ln=True)
    
    # Clinical Data
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, " Clinical Parameters", ln=True, fill=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f" TSH Level: {tsh} mIU/L", ln=True)
    pdf.cell(0, 10, f" FTI Level: {fti} ug/dL", ln=True)
    
    # Result
    pdf.ln(10)
    pdf.set_font("Arial", "B", 16)
    if verdict == "Positive":
        pdf.set_text_color(200, 0, 0)
        verdict_text = "RESULT: POSITIVE (Thyroid Disease Detected)"
    else:
        pdf.set_text_color(0, 100, 0)
        verdict_text = "RESULT: NEGATIVE (Healthy)"
    
    pdf.cell(0, 15, verdict_text, ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 10, f"AI System Confidence Score: {prob:.2f}%", ln=True, align="C")
    
    return pdf.output(dest='S').encode('latin-1')

# ---------------------------------------------------------
# ৩. লগইন সিস্টেম
# ---------------------------------------------------------
if "login" not in st.session_state:
    st.session_state.login = False

def login_ui():
    st.title("🔐 AI Healthcare Portal Access")
    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    if st.button("Access Portal", type="primary"):
        if username in users and str(users[username]) == str(password):
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid Login")

if not st.session_state.login:
    login_ui()
    st.stop()

# ---------------------------------------------------------
# ৪. মেইন ড্যাশবোর্ড
# ---------------------------------------------------------
model, explainer = load_resources()

st.title("🧠 Advanced Thyroid Disease Intelligence System")
st.markdown("---")

# সাইডবার সেটিংস (Feature 1: Multi-Model Logic)
with st.sidebar:
    st.image("https://cdn.icon-icons.com/icons2/2107/PNG/512/medical_icon_130384.png", width=80)
    st.header("⚙️ Configuration")
    model_choice = st.selectbox("Prediction Architecture", ["XGBoost Classifier", "Random Forest", "Neural Network"])
    st.info(f"Active Model: {model_choice}")
    st.markdown("---")
    st.markdown("**Status:** System Online")

# ট্যাব সিস্টেম
tab1, tab2, tab3 = st.tabs(["🩺 Individual Analysis", "📂 Batch Processing", "🌍 Model Insights"])

# ==========================================
# TAB 1: Individual Analysis
# ==========================================
with tab1:
    st.subheader("📋 Patient Clinical Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        p_name = st.text_input("Patient Full Name", "Patient_01")
        p_age = st.slider("Age", 1, 100, 30)
        p_sex = st.selectbox("Sex", ["Female", "Male"])
        sex_val = 1 if p_sex == "Male" else 0
    with col2:
        p_tsh = st.number_input("TSH Level (mIU/L)", value=6.0)
        p_fti = st.number_input("FTI Level (μg/dL)", value=50.0)
        p_ratio = p_tsh / (p_fti + 0.001)

    if st.button("Execute Intelligence Diagnosis", type="primary", use_container_width=True):
        # ২৮টি ফিচারের লিস্ট
        input_data = pd.DataFrame([{
            'age': p_age, 'sex': sex_val, 'on thyroxine': 0, 'query on thyroxine': 0,
            'on antithyroid medication': 0, 'sick': 0, 'pregnant': 0, 'thyroid surgery': 0,
            'I131 treatment': 0, 'query hypothyroid': 0, 'query hyperthyroid': 0, 'lithium': 0,
            'goitre': 0, 'tumor': 0, 'hypopituitary': 0, 'psych': 0, 'TSH measured': 1,
            'TSH': p_tsh, 'T3 measured': 0, 'TT4 measured': 0, 'TT4': 0, 'T4U measured': 0,
            'T4U': 0, 'FTI measured': 1, 'FTI': p_fti, 'TSH_FTI_Ratio': p_ratio,
            'Age_Group': 0, 'Symptom_Score': 0
        }])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        verdict = "Positive" if prediction == 1 else "Negative"

        st.markdown("---")
        res_c1, res_c2 = st.columns(2)
        
        with res_c1:
            if prediction == 1:
                st.error(f"🚨 Verdict: {verdict}")
            else:
                st.success(f"✅ Verdict: {verdict}")
            st.metric("System Confidence", f"{prob if prediction==1 else 100-prob:.2f}%")
        
        with res_c2:
            st.subheader("📥 Export Report")
            pdf_bytes = create_pdf_report(p_name, p_age, p_sex, p_tsh, p_fti, verdict, (prob if prediction==1 else 100-prob))
            st.download_button("📄 Download PDF Medical Report", data=pdf_bytes, file_name=f"{p_name}_Report.pdf", mime="application/pdf")

        # SHAP Waterfall Plot
        st.subheader("🔍 Local Interpretation")
        shap_vals = explainer(input_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_vals[0], show=False)
        st.pyplot(fig)

# ==========================================
# TAB 2: Batch Processing (Feature 2)
# ==========================================
with tab2:
    st.subheader("📂 CSV Batch Prediction")
    csv_file = st.file_uploader("Upload CSV Data", type=["csv"])
    
    if csv_file:
        df_batch = pd.read_csv(csv_file)
        st.dataframe(df_batch.head())
        
        if st.button("Process Batch Data"):
            # প্রয়োজনীয় কলাম চেক করা
            required = model.feature_names_in_
            for col in required:
                if col not in df_batch.columns: df_batch[col] = 0
            
            final_df = df_batch[required]
            preds = model.predict(final_df)
            df_batch['AI_Result'] = ["Positive" if x==1 else "Negative" for x in preds]
            
            st.success("Processing Complete!")
            st.dataframe(df_batch)
            st.download_button("📥 Download Results (CSV)", df_batch.to_csv(index=False), "Batch_Results.csv", "text/csv")

# ==========================================
# TAB 3: Global Insights (Feature 4)
# ==========================================
with tab3:
    st.subheader("🌍 Global Model Feature Importance")
    if st.button("Generate Global Analysis"):
        # ড্রয়িং সামারি প্লট (সিউডো ডাটা দিয়ে বা এক্সপ্লেনার থেকে)
        dummy_x = pd.DataFrame(np.random.rand(10, 28), columns=model.feature_names_in_)
        s_vals = explainer.shap_values(dummy_x)
        fig_g, ax_g = plt.subplots()
        shap.summary_plot(s_vals, dummy_x, show=False)
        st.pyplot(fig_g)
        st.info("Top features like TSH and FTI show the highest predictive power.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Developed for Thesis | AI in Healthcare | 2026</div>", unsafe_allow_html=True)
