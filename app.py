# =============================================================================
# THYROID DISEASE PREDICTION SYSTEM
# A Clinical Decision Support System for Thesis Presentation
# Author: [Your Name]
# Version: 2.0 — Full Professional Build
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import shap
import io
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ThyroPredict — Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS — PROFESSIONAL CLINICAL THEME
# =============================================================================
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #F4F7FB; }
    .block-container { padding: 2rem 2.5rem 2rem 2.5rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0F2041 0%, #1B3A6B 60%, #1E4D8C 100%);
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

    /* ── Cards ── */
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 2px 12px rgba(15,32,65,0.07);
        border: 1px solid #E8EDF5;
        margin-bottom: 1.2rem;
    }
    .card-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6B7A99;
        margin-bottom: 0.3rem;
    }
    .card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F2041;
        line-height: 1.1;
    }
    .card-sub {
        font-size: 0.82rem;
        color: #8E9AB5;
        margin-top: 0.2rem;
    }

    /* ── Hero Banner ── */
    .hero {
        background: linear-gradient(110deg, #0F2041 0%, #1B3A6B 50%, #1565C0 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "🩺";
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.12;
    }
    .hero h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        font-weight: 400;
        margin: 0 0 0.3rem 0;
        color: white;
    }
    .hero p { margin: 0; color: rgba(255,255,255,0.75); font-size: 0.95rem; }

    /* ── Result Boxes ── */
    .result-positive {
        background: linear-gradient(135deg, #FFF0F0, #FFE0E0);
        border: 2px solid #E53935;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #F0FFF4, #E0FFE8);
        border: 2px solid #2E7D32;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-label {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        font-weight: 400;
        margin-bottom: 0.3rem;
    }
    .result-positive .result-label { color: #B71C1C; }
    .result-negative .result-label { color: #1B5E20; }
    .result-conf {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-positive .result-conf { color: #C62828; }
    .result-negative .result-conf { color: #2E7D32; }

    /* ── Section Headings ── */
    .section-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 1.35rem;
        color: #0F2041;
        margin: 0.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E8EDF5;
    }

    /* ── Login ── */
    .login-wrap {
        max-width: 420px;
        margin: 5vh auto;
        background: white;
        border-radius: 20px;
        padding: 3rem 2.5rem;
        box-shadow: 0 8px 40px rgba(15,32,65,0.13);
        border: 1px solid #E8EDF5;
    }
    .login-logo {
        text-align: center;
        font-family: 'DM Serif Display', serif;
        font-size: 1.7rem;
        color: #0F2041;
        margin-bottom: 0.2rem;
    }
    .login-sub {
        text-align: center;
        color: #8E9AB5;
        font-size: 0.85rem;
        margin-bottom: 2rem;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-blue  { background: #E3EEFF; color: #1565C0; }
    .badge-green { background: #E8F5E9; color: #2E7D32; }
    .badge-red   { background: #FFEBEE; color: #C62828; }

    /* ── Info box ── */
    .info-box {
        background: #EEF4FF;
        border-left: 4px solid #1565C0;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        font-size: 0.88rem;
        color: #1B3A6B;
        margin-bottom: 1rem;
    }

    /* ── Explanation bullets ── */
    .explain-item {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid #F0F4FB;
        font-size: 0.88rem;
        color: #334;
    }
    .explain-icon { font-size: 1.1rem; flex-shrink: 0; }

    /* ── Table ── */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .styled-table th {
        background: #0F2041;
        color: white;
        padding: 0.65rem 1rem;
        text-align: left;
        font-weight: 600;
    }
    .styled-table td {
        padding: 0.6rem 1rem;
        border-bottom: 1px solid #E8EDF5;
        color: #334;
    }
    .styled-table tr:nth-child(even) td { background: #F7FAFF; }

    /* ── Streamlit overrides ── */
    .stTextInput > label, .stNumberInput > label, .stSlider > label {
        font-weight: 600;
        color: #334 !important;
        font-size: 0.88rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0F2041, #1565C0);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.8rem;
        font-weight: 600;
        font-size: 0.92rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; color: white; }
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(15,32,65,0.06);
        border: 1px solid #E8EDF5;
    }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #1B5E20, #2E7D32);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.8rem;
        font-weight: 600;
        font-size: 0.92rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "results" not in st.session_state:
    st.session_state.results = {}


# =============================================================================
# UTILITY — MODEL TRAINING
# =============================================================================
@st.cache_resource(show_spinner=False)
def train_models():
    """
    Train XGBoost and Random Forest classifiers on a synthetic thyroid dataset.
    In a real clinical system this would load a validated thyroid dataset (e.g., UCI).
    Features: TSH, FTI, TSH/FTI ratio.
    """
    np.random.seed(42)
    n = 1000

    # ── Simulate thyroid-like feature distributions ──
    # Healthy (label 0): TSH 0.4–4.0 mIU/L, FTI 60–160
    # Diseased (label 1): TSH >4 or <0.4, FTI <60 or >160
    tsh_neg  = np.random.uniform(0.4, 4.0, n // 2)
    fti_neg  = np.random.uniform(60, 160, n // 2)
    tsh_pos  = np.concatenate([
        np.random.uniform(4.1, 20.0, n // 4),
        np.random.uniform(0.01, 0.39, n // 4)
    ])
    fti_pos  = np.concatenate([
        np.random.uniform(10, 59, n // 4),
        np.random.uniform(161, 250, n // 4)
    ])

    tsh = np.concatenate([tsh_neg, tsh_pos])
    fti = np.concatenate([fti_neg, fti_pos])
    ratio = tsh / (fti + 1e-9)

    # Add mild noise
    tsh   += np.random.normal(0, 0.1, len(tsh))
    fti   += np.random.normal(0, 2,   len(fti))
    ratio += np.random.normal(0, 0.001, len(ratio))

    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    idx = np.random.permutation(len(labels))
    tsh, fti, ratio, labels = tsh[idx], fti[idx], ratio[idx], labels[idx]

    X = np.column_stack([tsh, fti, ratio])
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # ── Random Forest ──
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # ── XGBoost ──
    xg = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                            use_label_encoder=False, eval_metric="logloss", random_state=42)
    xg.fit(X_train, y_train)
    xg_acc = accuracy_score(y_test, xg.predict(X_test))

    # ── SHAP explainer (TreeExplainer for XGBoost) ──
    explainer = shap.TreeExplainer(xg)

    return {
        "rf": rf, "xg": xg, "explainer": explainer,
        "rf_acc": rf_acc, "xg_acc": xg_acc,
        "X_train": X_train, "feature_names": ["TSH", "FTI", "TSH/FTI Ratio"]
    }


# =============================================================================
# LOGIN SCREEN
# =============================================================================
def show_login():
    st.markdown("""
    <div class="login-wrap">
        <div class="login-logo">🩺 ThyroPredict</div>
        <div class="login-sub">Clinical Decision Support System</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔐 Secure Login")
            st.markdown("<div class='info-box'>This system is restricted to authorised clinical personnel only.</div>",
                        unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            if st.button("Login →"):
                if username == "1234" and password == "1234":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            st.markdown("<div style='text-align:center;color:#8E9AB5;font-size:0.78rem;margin-top:1rem;'>"
                        "Demo credentials — Username: 1234 / Password: 1234</div>", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================
def show_sidebar(models):
    with st.sidebar:
        st.markdown("## 🩺 ThyroPredict")
        st.markdown("*Clinical Decision Support System*")
        st.markdown("---")
        st.markdown("**📊 Model Performance**")
        st.markdown(f"""
        <div class='card' style='background:rgba(255,255,255,0.08);border-color:rgba(255,255,255,0.1);'>
            <div class='card-title' style='color:rgba(255,255,255,0.6);'>XGBoost Accuracy</div>
            <div class='card-value' style='color:#64B5F6;font-size:1.5rem;'>{models['xg_acc']*100:.1f}%</div>
        </div>
        <div class='card' style='background:rgba(255,255,255,0.08);border-color:rgba(255,255,255,0.1);'>
            <div class='card-title' style='color:rgba(255,255,255,0.6);'>Random Forest Accuracy</div>
            <div class='card-value' style='color:#A5D6A7;font-size:1.5rem;'>{models['rf_acc']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**📋 Normal Ranges**")
        st.markdown("""
        | Marker | Normal Range |
        |--------|-------------|
        | TSH | 0.4 – 4.0 mIU/L |
        | FTI | 60 – 160 |
        | Ratio | 0.003 – 0.067 |
        """)
        st.markdown("---")
        if st.button("🔒 Logout"):
            st.session_state.logged_in = False
            st.session_state.prediction_done = False
            st.rerun()
        st.markdown("<div style='font-size:0.72rem;color:rgba(255,255,255,0.4);margin-top:1rem;'>v2.0 — Thesis Build</div>",
                    unsafe_allow_html=True)


# =============================================================================
# INPUT VALIDATION
# =============================================================================
def validate_inputs(age, tsh, fti):
    errors = []
    if age < 1 or age > 120:
        errors.append("Age must be between 1 and 120 years.")
    if tsh < 0.01 or tsh > 100:
        errors.append("TSH must be between 0.01 and 100 mIU/L.")
    if fti < 1 or fti > 500:
        errors.append("FTI must be between 1 and 500.")
    return errors


# =============================================================================
# VISUALISATION — BAR CHART vs NORMAL RANGE
# =============================================================================
def plot_input_vs_normal(tsh, fti, ratio):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#F4F7FB")
    fig.patch.set_facecolor("#F4F7FB")

    markers = [
        ("TSH (mIU/L)", tsh,   0.4, 4.0,   "#1565C0"),
        ("FTI",         fti,   60,  160,   "#1565C0"),
        ("TSH/FTI Ratio", ratio, 0.003, 0.067, "#1565C0"),
    ]

    for ax, (label, val, lo, hi, color) in zip(axes, markers):
        ax.set_facecolor("white")
        # Normal range band
        ax.axhspan(lo, hi, color="#E8F5E9", alpha=0.8, label="Normal Range")
        # Bar for patient value
        bar_color = "#E53935" if (val < lo or val > hi) else "#2E7D32"
        ax.bar([label], [val], color=bar_color, width=0.5, zorder=3, alpha=0.85)
        # Reference lines
        ax.axhline(lo, color="#2E7D32", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.axhline(hi, color="#2E7D32", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_title(label, fontsize=10, fontweight="bold", color="#0F2041", pad=8)
        ax.tick_params(axis='both', labelsize=8, colors="#555")
        for spine in ax.spines.values():
            spine.set_color("#E8EDF5")
        ax.set_xlim(-0.5, 0.5)
        # Annotate value
        ax.text(0, val * 1.03, f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=bar_color)

    normal_patch = mpatches.Patch(color="#E8F5E9", label="Normal Range")
    fig.legend(handles=[normal_patch], loc="upper right", fontsize=8,
               framealpha=0.5, edgecolor="#E8EDF5")
    plt.suptitle("Patient Values vs. Normal Reference Ranges", fontsize=12,
                 fontweight="bold", color="#0F2041", y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# VISUALISATION — SHAP FEATURE IMPORTANCE
# =============================================================================
def plot_shap(models, tsh, fti, ratio):
    X_sample = np.array([[tsh, fti, ratio]])
    shap_vals = models["explainer"].shap_values(X_sample)
    # shap_vals shape: (1, 3) for binary classification (positive class)
    sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals[0]

    features = models["feature_names"]
    colors = ["#E53935" if v > 0 else "#1565C0" for v in sv]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#F4F7FB")
    ax.set_facecolor("white")
    bars = ax.barh(features, sv, color=colors, alpha=0.85, height=0.5)
    ax.axvline(0, color="#334", linewidth=0.8, linestyle="-")
    ax.set_xlabel("SHAP Value (Impact on Prediction)", fontsize=9, color="#555")
    ax.set_title("Feature Impact — SHAP Explanation", fontsize=11,
                 fontweight="bold", color="#0F2041", pad=10)
    ax.tick_params(labelsize=9, colors="#555")
    for spine in ax.spines.values():
        spine.set_color("#E8EDF5")
    # Annotate bars
    for bar, v in zip(bars, sv):
        ax.text(v + (0.001 if v >= 0 else -0.001), bar.get_y() + bar.get_height() / 2,
                f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color="#334")
    red_patch  = mpatches.Patch(color="#E53935", label="Pushes toward Positive")
    blue_patch = mpatches.Patch(color="#1565C0", label="Pushes toward Negative")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, framealpha=0.5)
    plt.tight_layout()
    return fig, sv


# =============================================================================
# MODEL COMPARISON TABLE
# =============================================================================
def show_model_comparison(models):
    rf_acc = models["rf_acc"]
    xg_acc = models["xg_acc"]
    data = {
        "Model": ["XGBoost", "Random Forest"],
        "Accuracy (%)": [f"{xg_acc*100:.2f}", f"{rf_acc*100:.2f}"],
        "Type": ["Gradient Boosting", "Ensemble Bagging"],
        "Trees": ["150", "150"],
        "Selected": ["✅ Yes (Primary)", "—"]
    }
    df = pd.DataFrame(data)
    table_html = df.to_html(index=False, classes="styled-table", border=0)
    st.markdown(table_html, unsafe_allow_html=True)


# =============================================================================
# REPORT GENERATION
# =============================================================================
def generate_report(age, tsh, fti, ratio, pred_label, confidence, shap_vals, feature_names):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 65,
        "   THYRPREDICT — CLINICAL DECISION SUPPORT SYSTEM",
        "   Thyroid Disease Prediction Report",
        "=" * 65,
        f"  Date & Time  : {now}",
        f"  Report Type  : AI-Assisted Clinical Screening",
        "-" * 65,
        "",
        "  PATIENT INPUT VALUES",
        f"  Age           : {age} years",
        f"  TSH           : {tsh:.4f} mIU/L   [Normal: 0.4 – 4.0]",
        f"  FTI           : {fti:.2f}          [Normal: 60 – 160]",
        f"  TSH/FTI Ratio : {ratio:.6f}     [Normal: 0.003 – 0.067]",
        "",
        "-" * 65,
        "  PREDICTION RESULT",
        f"  Outcome       : {pred_label}",
        f"  Confidence    : {confidence:.1f}%",
        f"  Model Used    : XGBoost (Primary)",
        "",
        "-" * 65,
        "  EXPLAINABLE AI — SHAP FEATURE CONTRIBUTIONS",
    ]
    for fname, sv in zip(feature_names, shap_vals):
        direction = "↑ Toward Positive" if sv > 0 else "↓ Toward Negative"
        lines.append(f"  {fname:<20}: {sv:+.6f}  ({direction})")
    lines += [
        "",
        "-" * 65,
        "  CLINICAL INTERPRETATION",
    ]
    if pred_label == "POSITIVE":
        lines += [
            "  The model predicts a POSITIVE thyroid condition.",
            "  Key contributing factors:",
        ]
        for fname, sv in zip(feature_names, shap_vals):
            if sv > 0:
                lines.append(f"  - {fname} significantly increased the likelihood of disease.")
        lines.append("")
        lines.append("  ⚠  RECOMMENDATION: Refer to endocrinologist for confirmatory")
        lines.append("     tests (Total T4, Free T3, thyroid antibodies, ultrasound).")
    else:
        lines += [
            "  The model predicts a NEGATIVE (Healthy) thyroid condition.",
            "  Biomarker values are within or near normal clinical ranges.",
            "",
            "  ✓  RECOMMENDATION: No immediate clinical action required.",
            "     Continue routine monitoring as per clinical guidelines.",
        ]
    lines += [
        "",
        "-" * 65,
        "  DISCLAIMER",
        "  This report is generated by an AI model for research and",
        "  educational purposes only. It must NOT replace professional",
        "  medical diagnosis. Always consult a qualified physician.",
        "=" * 65,
        ""
    ]
    return "\n".join(lines)


# =============================================================================
# NATURAL LANGUAGE EXPLANATION
# =============================================================================
def build_explanation(tsh, fti, ratio, pred_label, shap_vals, feature_names):
    explanations = []
    tsh_normal   = 0.4 <= tsh <= 4.0
    fti_normal   = 60 <= fti <= 160
    ratio_normal = 0.003 <= ratio <= 0.067

    if pred_label == "POSITIVE":
        explanations.append(("🔴", "The model predicts a <b>Positive</b> thyroid condition based on your biomarker profile."))
    else:
        explanations.append(("🟢", "The model predicts a <b>Negative (Healthy)</b> thyroid status based on your biomarker profile."))

    # TSH
    if not tsh_normal:
        direction = "elevated" if tsh > 4.0 else "suppressed"
        explanations.append(("🔬", f"Your <b>TSH ({tsh:.3f} mIU/L)</b> is {direction} (normal: 0.4–4.0 mIU/L), "
                             f"which contributed {'positively' if shap_vals[0]>0 else 'negatively'} to this prediction."))
    else:
        explanations.append(("✅", f"Your <b>TSH ({tsh:.3f} mIU/L)</b> is within the normal range (0.4–4.0 mIU/L)."))

    # FTI
    if not fti_normal:
        direction = "high" if fti > 160 else "low"
        explanations.append(("🔬", f"Your <b>FTI ({fti:.1f})</b> is {direction} (normal: 60–160), "
                             f"influencing the result {('toward disease' if shap_vals[1]>0 else 'toward healthy')}."))
    else:
        explanations.append(("✅", f"Your <b>FTI ({fti:.1f})</b> is within the normal range (60–160)."))

    # Ratio
    if not ratio_normal:
        explanations.append(("🔬", f"The <b>TSH/FTI Ratio ({ratio:.5f})</b> is outside the expected range (0.003–0.067), "
                             f"further {'supporting' if shap_vals[2]>0 else 'reducing'} the positive prediction."))
    else:
        explanations.append(("✅", f"The <b>TSH/FTI Ratio ({ratio:.5f})</b> is within the normal range."))

    return explanations


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def show_dashboard(models):
    show_sidebar(models)

    # ── Hero Banner ──
    st.markdown("""
    <div class='hero'>
        <h1>ThyroPredict — Clinical Dashboard</h1>
        <p>AI-powered thyroid disease screening using TSH, FTI, and TSH/FTI Ratio biomarkers</p>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──
    tabs = st.tabs(["🩺 Prediction", "📊 Visualisation", "🧠 Explainability", "⚖️ Model Comparison", "📄 Report"])

    # =====================================================================
    # TAB 1 — PREDICTION
    # =====================================================================
    with tabs[0]:
        st.markdown("<div class='section-heading'>Patient Biomarker Input</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Enter the patient's biomarker values below. TSH/FTI ratio is computed automatically. All predictions are made using a trained XGBoost model.</div>",
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("👤 Age (years)", min_value=1, max_value=120, value=35, step=1)
        with col2:
            tsh = st.number_input("🧪 TSH (mIU/L)", min_value=0.01, max_value=100.0, value=2.5, step=0.01, format="%.2f")
        with col3:
            fti = st.number_input("🧬 FTI", min_value=1.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")

        # Auto-calculated ratio
        ratio = tsh / (fti + 1e-9)
        st.markdown(f"""
        <div class='card' style='background:#EEF4FF;border:1.5px solid #BBDEFB;'>
            <div class='card-title'>Auto-Calculated TSH/FTI Ratio</div>
            <div class='card-value' style='font-size:1.6rem;color:#1565C0;'>{ratio:.6f}</div>
            <div class='card-sub'>TSH ÷ FTI = {tsh:.4f} ÷ {fti:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Validation & Predict ──
        errors = validate_inputs(age, tsh, fti)
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            if st.button("🔍 Run Prediction"):
                X_input = np.array([[tsh, fti, ratio]])
                pred     = models["xg"].predict(X_input)[0]
                prob     = models["xg"].predict_proba(X_input)[0]
                confidence = prob[pred] * 100
                pred_label = "POSITIVE" if pred == 1 else "NEGATIVE"

                # SHAP
                shap_vals_raw = models["explainer"].shap_values(X_input)
                sv = shap_vals_raw[0] if shap_vals_raw.ndim == 2 else shap_vals_raw[0]

                st.session_state.results = {
                    "age": age, "tsh": tsh, "fti": fti, "ratio": ratio,
                    "pred": pred, "pred_label": pred_label,
                    "confidence": confidence, "prob": prob, "sv": sv
                }
                st.session_state.prediction_done = True

        # ── Result Display ──
        if st.session_state.prediction_done and st.session_state.results:
            r = st.session_state.results
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-heading'>Prediction Result</div>", unsafe_allow_html=True)

            rcol1, rcol2 = st.columns([1.2, 1])
            with rcol1:
                css_class = "result-positive" if r["pred_label"] == "POSITIVE" else "result-negative"
                icon = "🔴" if r["pred_label"] == "POSITIVE" else "🟢"
                st.markdown(f"""
                <div class='{css_class}'>
                    <div class='result-label'>{icon} {r["pred_label"]}</div>
                    <div class='result-conf'>Confidence: {r["confidence"]:.1f}%</div>
                    <div style='margin-top:0.5rem;font-size:0.82rem;color:#555;'>
                        Probability Healthy: {r["prob"][0]*100:.1f}% &nbsp;|&nbsp;
                        Probability Diseased: {r["prob"][1]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with rcol2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.metric("TSH", f"{r['tsh']:.3f} mIU/L", delta="High" if r["tsh"]>4 else ("Low" if r["tsh"]<0.4 else "Normal"))
                st.metric("FTI", f"{r['fti']:.1f}", delta="High" if r["fti"]>160 else ("Low" if r["fti"]<60 else "Normal"))
                st.metric("TSH/FTI", f"{r['ratio']:.5f}")
                st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================================
    # TAB 2 — VISUALISATION
    # =====================================================================
    with tabs[1]:
        st.markdown("<div class='section-heading'>Biomarker Values vs Normal Reference Ranges</div>", unsafe_allow_html=True)
        if st.session_state.prediction_done and st.session_state.results:
            r = st.session_state.results
            fig = plot_input_vs_normal(r["tsh"], r["fti"], r["ratio"])
            st.pyplot(fig, use_container_width=True)
            st.markdown("""
            <div class='info-box'>
            <b>Reading the chart:</b> Green shading = normal range. 
            A <span style='color:#2E7D32;font-weight:600;'>green bar</span> means the value is within range; 
            a <span style='color:#E53935;font-weight:600;'>red bar</span> means the value is outside the normal range.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Run a prediction first to see the visualisation.")

    # =====================================================================
    # TAB 3 — EXPLAINABILITY
    # =====================================================================
    with tabs[2]:
        st.markdown("<div class='section-heading'>Explainable AI — SHAP Analysis</div>", unsafe_allow_html=True)
        if st.session_state.prediction_done and st.session_state.results:
            r = st.session_state.results

            # SHAP chart
            fig_shap, _ = plot_shap(models, r["tsh"], r["fti"], r["ratio"])
            st.pyplot(fig_shap, use_container_width=True)

            st.markdown("<div class='section-heading' style='margin-top:1.5rem;'>Natural Language Explanation</div>",
                        unsafe_allow_html=True)
            explanations = build_explanation(r["tsh"], r["fti"], r["ratio"],
                                              r["pred_label"], r["sv"], models["feature_names"])
            for icon, text in explanations:
                st.markdown(f"""
                <div class='explain-item'>
                    <span class='explain-icon'>{icon}</span>
                    <span>{text}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-box'>
            <b>What is SHAP?</b> SHAP (SHapley Additive exPlanations) assigns each feature an importance value 
            for a specific prediction. Red bars push the prediction toward <em>Positive (diseased)</em>; 
            blue bars push toward <em>Negative (healthy)</em>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Run a prediction first to see the explainability analysis.")

    # =====================================================================
    # TAB 4 — MODEL COMPARISON
    # =====================================================================
    with tabs[3]:
        st.markdown("<div class='section-heading'>Model Performance Comparison</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Both XGBoost and Random Forest were trained on the same synthetic thyroid dataset with identical train/test splits (75/25). XGBoost is selected as the primary model for its superior accuracy and gradient boosting robustness.</div>",
                    unsafe_allow_html=True)
        show_model_comparison(models)

        # Bar chart comparison
        st.markdown("<br>", unsafe_allow_html=True)
        fig_cmp, ax = plt.subplots(figsize=(6, 3.2), facecolor="#F4F7FB")
        ax.set_facecolor("white")
        names = ["XGBoost", "Random Forest"]
        accs  = [models["xg_acc"] * 100, models["rf_acc"] * 100]
        colors = ["#1565C0", "#2E7D32"]
        bars = ax.bar(names, accs, color=colors, width=0.4, alpha=0.85)
        ax.set_ylim(80, 100)
        ax.set_ylabel("Accuracy (%)", fontsize=9, color="#555")
        ax.set_title("Model Accuracy Comparison", fontsize=11, fontweight="bold", color="#0F2041")
        ax.tick_params(labelsize=9, colors="#555")
        for spine in ax.spines.values():
            spine.set_color("#E8EDF5")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.2,
                    f"{acc:.2f}%", ha="center", fontsize=9, fontweight="bold", color="#0F2041")
        plt.tight_layout()
        st.pyplot(fig_cmp)

        st.markdown("""
        <div class='card' style='margin-top:1rem;'>
            <div class='card-title'>Why XGBoost?</div>
            <div style='font-size:0.88rem;color:#334;line-height:1.7;'>
            XGBoost uses gradient boosting — iteratively correcting errors from prior trees — 
            which generally yields higher accuracy than Random Forest's parallel bagging strategy 
            on tabular biomedical data. It also integrates natively with SHAP, enabling robust 
            explainability required for clinical deployment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================================
    # TAB 5 — REPORT
    # =====================================================================
    with tabs[4]:
        st.markdown("<div class='section-heading'>Downloadable Clinical Report</div>", unsafe_allow_html=True)
        if st.session_state.prediction_done and st.session_state.results:
            r = st.session_state.results
            report_text = generate_report(
                r["age"], r["tsh"], r["fti"], r["ratio"],
                r["pred_label"], r["confidence"],
                r["sv"], models["feature_names"]
            )
            st.code(report_text, language="text")
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=report_text.encode("utf-8"),
                file_name=f"ThyroPredict_Report_{now_str}.txt",
                mime="text/plain"
            )
            st.markdown("""
            <div class='info-box' style='margin-top:1rem;'>
            ⚠️ <b>Disclaimer:</b> This report is generated by an AI model for research and educational 
            purposes only. It does not constitute medical advice and must not replace 
            professional clinical diagnosis.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Run a prediction first to generate the report.")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    if not st.session_state.logged_in:
        show_login()
        return

    # Train models (cached — runs only once per session)
    with st.spinner("🔄 Initialising models and system…"):
        models = train_models()

    show_dashboard(models)


if __name__ == "__main__":
    main()
