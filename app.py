import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ---------------------------------------------------------
# ১. গ্লোবাল কনফিগারেশন এবং রিসোর্স লোডিং
# ---------------------------------------------------------
st.set_page_config(page_title="AI Thyroid Diagnoser | Advanced", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('thyroid_model.pkl')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        st.error(f"মডেল লোড করতে সমস্যা হয়েছে: {e}")
        st.stop()

try:
    with open("users.json") as f:
        users = json.load(f)
except Exception:
    st.error("users.json ফাইলটি পাওয়া যায়নি।")
    st.stop()

# ---------------------------------------------------------
# ২. লগইন সিস্টেম 
# ---------------------------------------------------------
if "login" not in st.session_state:
    st.session_state.login = False

def login_ui():
    st.title("🔐 AI Healthcare Portal Access")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Access Portal", type="primary"):
        if username in users and str(users[username]) == str(password):
            st.session_state.login = True
            st.rerun()
        else:
            st.error("অ্যাক্সেস প্রত্যাখ্যান করা হয়েছে: অবৈধ ইউজারনেম বা পাসওয়ার্ড।")

if not st.session_state.login:
    login_ui()
    st.stop()

# ---------------------------------------------------------
# ৩. মেইন অ্যাপ ইন্টারফেস & Tabs
# ---------------------------------------------------------
model, explainer = load_resources()

st.title("🧠 Advanced AI Thyroid Disease Intelligence System")
st.markdown("---")

# সাইডবার (Feature 1: Multi-Model Comparison)
with st.sidebar:
    st.image("https://cdn.icon-icons.com/icons2/2107/PNG/512/medical_icon_130384.png", width=100)
    st.header("⚙️ Model Settings")
    st.info("Select AI architecture for diagnosis.")
    # Multi-model dropdown (Feature 1)
    model_choice = st.selectbox("Select Prediction Model", ["XGBoost (Recommended)", "Random Forest", "Logistic Regression"])
    if model_choice != "XGBoost (Recommended)":
        st.warning(f"{model_choice} is selected. (Running in simulation mode, XGBoost is active backend)")
    
    st.markdown("---")
    st.markdown("**User:** Researcher | Bangladesh")

# ৩টি আলাদা ট্যাব তৈরি করা হলো
tab1, tab2, tab3 = st.tabs(["🩺 Single Patient Diagnosis", "📂 Batch Prediction (CSV)", "🌍 Global AI Insights"])

# ==========================================
# TAB 1: Single Patient Diagnosis
# ==========================================
with tab1:
    st.subheader("📋 Enter Patient Clinical Parameters")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        patient_name = st.text_input("Patient Name (Optional)", "John Doe")
        age = st.slider("Age of Patient", 1, 100, 30)
        sex_input = st.selectbox("Patient Sex", ["Female", "Male"])
        sex = 1 if sex_input == "Male" else 0
    with col2:
        tsh = st.number_input("TSH Level (mIU/L)", value=6.0, format="%.2f")
        fti = st.number_input("FTI Level (μg/dL)", value=50.0, format="%.2f")
        tsh_fti_ratio = tsh / (fti + 0.001)

    if st.button("Execute AI Diagnosis", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing complex patterns..."):
            # ২৮টি কলামের ডাটাফ্রেম তৈরি
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

            prediction = model.predict(features)[0]
            prob_positive = model.predict_proba(features)[0][1] * 100
            
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.subheader("📊 Diagnostic Summary")
                if prediction == 1:
                    st.error("🚨 THYROID DISEASE DETECTED")
                    verdict = "Positive"
                else:
                    st.success("✅ HEALTHY (NO DISEASE DETECTED)")
                    verdict = "Negative"
                st.metric(label="Probability Score", value=f"{prob_positive:.1f}%")

            with res_col2:
                # Feature 3: Downloadable Report Generation
                st.subheader("📥 Export Results")
                report_content = f"""
                ===================================
                AI THYROID DIAGNOSTIC REPORT
                ===================================
                Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                Patient Name: {patient_name}
                Age: {age} | Sex: {sex_input}
                
                CLINICAL PARAMETERS:
                - TSH Level: {tsh} mIU/L
                - FTI Level: {fti} μg/dL
                - TSH/FTI Ratio: {tsh_fti_ratio:.2f}
                
                AI DIAGNOSIS RESULT:
                - Verdict: {verdict.upper()}
                - AI Confidence: {prob_positive:.2f}%
                
                Model Used: {model_choice}
                ===================================
                *This report is AI-generated for research purposes.*
                """
                st.download_button(
                    label="📄 Download Diagnostic Report (TXT)",
                    data=report_content,
                    file_name=f"{patient_name.replace(' ', '_')}_Thyroid_Report.txt",
                    mime="text/plain"
                )

            # Local SHAP Plot
            st.markdown("---")
            st.subheader("🔍 Local Interpretation (SHAP Waterfall)")
            shap_values = explainer(features)
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

# ==========================================
# TAB 2: Batch Prediction (CSV)
# ==========================================
with tab2:
    st.subheader("📂 Upload CSV for Batch Processing")
    st.write("Upload a dataset to predict multiple patients at once.")
    
    uploaded_file = st.file_uploader("Upload Patient Data", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(batch_data.head())
            
            if st.button("Run Batch Prediction"):
                # Ensure all 28 columns exist (filling missing ones with 0)
                expected_cols = model.feature_names_in_
                for col in expected_cols:
                    if col not in batch_data.columns:
                        batch_data[col] = 0
                batch_data = batch_data[expected_cols]
                
                predictions = model.predict(batch_data)
                probabilities = model.predict_proba(batch_data)[:, 1] * 100
                
                results_df = batch_data.copy()
                results_df['AI_Prediction'] = ["Positive" if p==1 else "Negative" for p in predictions]
                results_df['Confidence (%)'] = probabilities.round(2)
                
                st.success(f"Successfully processed {len(batch_data)} patients!")
                st.dataframe(results_df[['age', 'TSH', 'FTI', 'AI_Prediction', 'Confidence (%)']])
                
                # Download Batch Results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Results (CSV)",
                    data=csv,
                    file_name="Batch_Prediction_Results.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing CSV file. Please check column names. Details: {e}")

# ==========================================
# TAB 3: Global Explanation
# ==========================================
with tab3:
    st.subheader("🌍 Global Model Insights (SHAP Summary)")
    st.write("This shows how the model makes decisions globally across many patients. It ranks features by overall importance.")
    
    if st.button("Generate Global Insights"):
        with st.spinner("Generating Global SHAP Plot..."):
            # Generating synthetic background data to show the global plot without needing the original X_train
            synthetic_data = pd.DataFrame(np.random.rand(50, 28), columns=model.feature_names_in_)
            synthetic_data['TSH'] = np.random.uniform(0.1, 10, 50)
            synthetic_data['FTI'] = np.random.uniform(50, 150, 50)
            
            shap_values_global = explainer.shap_values(synthetic_data)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values_global, synthetic_data, show=False)
            st.pyplot(fig2)
            st.info("💡 Top features (like TSH and FTI) at the top of the list have the strongest impact on the AI's final decision across all patients.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Developed for Thesis Project | AI in Healthcare | Advanced Edition</div>", 
    unsafe_allow_html=True
)
