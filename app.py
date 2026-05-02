# =============================================================================
# THYROID DISEASE PREDICTION SYSTEM
# A Clinical Decision Support System for Thesis Presentation
# Author: [Your Name]
# Version: 2.1 — UI Fixed Build
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap 
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from reportlab.lib import colors
NAVY = colors.HexColor("#0F2041")

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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #F4F7FB; }
    .block-container { padding: 2rem 2.5rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0F2041 0%, #1B3A6B 60%, #1E4D8C 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

    /* ── Cards ── */
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 12px rgba(15,32,65,0.07);
        border: 1px solid #E8EDF5;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6B7A99;
        margin-bottom: 0.25rem;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F2041;
        line-height: 1.15;
    }
    .card-sub { font-size: 0.80rem; color: #8E9AB5; margin-top: 0.15rem; }

    /* ── Sidebar metric cards ── */
    .sb-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
    .sb-card-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
                     text-transform: uppercase; color: rgba(255,255,255,0.55); }
    .sb-card-value { font-size: 1.5rem; font-weight: 700; }

    /* ── Hero Banner ── */
    .hero {
        background: linear-gradient(110deg, #0F2041 0%, #1B3A6B 55%, #1565C0 100%);
        border-radius: 16px;
        padding: 1.8rem 2.5rem;
        color: white;
        margin-bottom: 1.8rem;
    }
    .hero h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.9rem;
        font-weight: 400;
        margin: 0 0 0.25rem 0;
        color: white;
    }
    .hero p { margin: 0; color: rgba(255,255,255,0.72); font-size: 0.92rem; }

    /* ── Result boxes ── */
    .result-positive {
        background: linear-gradient(135deg, #FFF0F0, #FFE0E0);
        border: 2px solid #E53935;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-negative {
        background: linear-gradient(135deg, #F0FFF4, #E0FFE8);
        border: 2px solid #2E7D32;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-label { font-family: 'DM Serif Display', serif; font-size: 2.2rem;
                    font-weight: 400; margin-bottom: 0.3rem; }
    .result-positive .result-label { color: #B71C1C; }
    .result-negative .result-label { color: #1B5E20; }
    .result-conf { font-size: 1.1rem; font-weight: 600; }
    .result-positive .result-conf { color: #C62828; }
    .result-negative .result-conf { color: #2E7D32; }

    /* ── Section Headings ── */
    .section-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #0F2041;
        margin: 0.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E8EDF5;
    }

    /* ── Info box ── */
    .info-box {
        background: #EEF4FF;
        border-left: 4px solid #1565C0;
        border-radius: 0 10px 10px 0;
        padding: 0.85rem 1.2rem;
        font-size: 0.87rem;
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
        font-size: 0.87rem;
        color: #334;
    }
    .explain-icon { font-size: 1.1rem; flex-shrink: 0; }

    /* ── Tables ── */
    .styled-table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
    .styled-table th { background: #0F2041; color: white; padding: 0.6rem 1rem;
                       text-align: left; font-weight: 600; }
    .styled-table td { padding: 0.55rem 1rem; border-bottom: 1px solid #E8EDF5; color: #334; }
    .styled-table tr:nth-child(even) td { background: #F7FAFF; }

    /* ── Login ── */
    .login-wrap {
        max-width: 420px; margin: 4vh auto;
        background: white; border-radius: 20px;
        padding: 3rem 2.5rem;
        box-shadow: 0 8px 40px rgba(15,32,65,0.13);
        border: 1px solid #E8EDF5;
    }
    .login-logo { text-align: center; font-family: 'DM Serif Display', serif;
                  font-size: 1.7rem; color: #0F2041; margin-bottom: 0.2rem; }
    .login-sub { text-align: center; color: #8E9AB5;
                 font-size: 0.85rem; margin-bottom: 2rem; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(90deg, #0F2041, #1565C0);
        color: white; border: none; border-radius: 8px;
        padding: 0.55rem 1.8rem; font-weight: 600; font-size: 0.92rem;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; color: white; }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #1B5E20, #2E7D32);
        color: white; border: none; border-radius: 8px;
        padding: 0.55rem 1.8rem; font-weight: 600; font-size: 0.92rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
if "logged_in"        not in st.session_state: st.session_state.logged_in        = False
if "prediction_done"  not in st.session_state: st.session_state.prediction_done  = False
if "results"          not in st.session_state: st.session_state.results          = {}


# =============================================================================
# MODEL TRAINING  (cached — runs once per session)
# =============================================================================
@st.cache_resource(show_spinner=False)
def train_models():
    """
    Train XGBoost and Random Forest on a synthetic thyroid-like dataset.
    Features: TSH, FTI, TSH/FTI ratio.  Labels: 0 = Healthy, 1 = Diseased.
    """
    np.random.seed(42)
    n = 1000

    # Simulate realistic thyroid biomarker distributions
    tsh_neg = np.random.uniform(0.4, 4.0, n // 2)
    fti_neg = np.random.uniform(60, 160,  n // 2)
    tsh_pos = np.concatenate([np.random.uniform(4.1, 20.0, n // 4),
                               np.random.uniform(0.01, 0.39, n // 4)])
    fti_pos = np.concatenate([np.random.uniform(10,  59,   n // 4),
                               np.random.uniform(161, 250,  n // 4)])

    tsh   = np.concatenate([tsh_neg, tsh_pos]) + np.random.normal(0, 0.1, n)
    fti   = np.concatenate([fti_neg, fti_pos]) + np.random.normal(0, 2,   n)
    ratio = tsh / (fti + 1e-9)
    y     = np.array([0] * (n // 2) + [1] * (n // 2))

    idx = np.random.permutation(n)
    X   = np.column_stack([tsh, fti, ratio])[idx]
    y   = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    xg = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                            use_label_encoder=False, eval_metric="logloss", random_state=42)
    xg.fit(X_train, y_train)
    xg_acc = accuracy_score(y_test, xg.predict(X_test))

    explainer = shap.TreeExplainer(xg)

    return {
        "rf": rf, "xg": xg, "explainer": explainer,
        "rf_acc": rf_acc, "xg_acc": xg_acc,
        "feature_names": ["TSH", "FTI", "TSH/FTI Ratio"]
    }


# =============================================================================
# LOGIN SCREEN
# =============================================================================
def show_login():
    st.markdown("""
    <div class='login-wrap'>
        <div class='login-logo'>🩺 ThyroPredict</div>
        <div class='login-sub'>Clinical Decision Support System</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔐 Secure Login")
        st.markdown("<div class='info-box'>Access is restricted to authorised clinical personnel.</div>",
                    unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Login →"):
            if username == "1234" and password == "1234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials.")
        st.markdown("<div style='text-align:center;color:#8E9AB5;font-size:0.77rem;margin-top:1rem;'>"
                    "Demo — Username: 1234 / Password: 1234</div>", unsafe_allow_html=True)


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
        <div class='sb-card'>
            <div class='sb-card-label'>XGBoost Accuracy</div>
            <div class='sb-card-value' style='color:#64B5F6;'>{models['xg_acc']*100:.1f}%</div>
        </div>
        <div class='sb-card'>
            <div class='sb-card-label'>Random Forest Accuracy</div>
            <div class='sb-card-value' style='color:#A5D6A7;'>{models['rf_acc']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**📋 Normal Reference Ranges**")
        st.markdown("""
        | Marker | Normal Range |
        |--------|-------------|
        | TSH | 0.4 – 4.0 mIU/L |
        | FTI | 60 – 160 |
        | Ratio | 0.003 – 0.067 |
        """)
        st.markdown("---")
        if st.button("🔒 Logout"):
            st.session_state.logged_in       = False
            st.session_state.prediction_done = False
            st.session_state.results         = {}
            st.rerun()
        st.markdown("<div style='font-size:0.70rem;color:rgba(255,255,255,0.35);margin-top:0.8rem;'>"
                    "v2.1 — Thesis Build</div>", unsafe_allow_html=True)


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
# CHART 1 — Biomarker vs Normal Range  (clean default matplotlib style)
# =============================================================================
def plot_input_vs_normal(tsh, fti, ratio):
    """
    Three-panel bar chart.
    Each panel: patient value bar + green normal-range band + dashed reference lines.
    Uses default matplotlib style — consistent with Google Colab output.
    """
    labels     = ["TSH (mIU/L)", "FTI", "TSH/FTI Ratio"]
    values     = [tsh,  fti,  ratio]
    normals_lo = [0.4,  60,   0.003]
    normals_hi = [4.0,  160,  0.067]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, label, val, lo, hi in zip(axes, labels, values, normals_lo, normals_hi):
        # Normal-range shading
        ax.axhspan(lo, hi, alpha=0.20, color="green", label="Normal Range")
        # Reference dashed lines
        ax.axhline(lo, color="green", linewidth=1.2, linestyle="--", label=f"Lower: {lo}")
        ax.axhline(hi, color="blue",  linewidth=1.2, linestyle="--", label=f"Upper: {hi}")
        # Patient bar — red if outside, green if inside range
        bar_color = "red" if (val < lo or val > hi) else "green"
        ax.bar(["Patient"], [val], color=bar_color, width=0.4, alpha=0.75)
        # Value label on top of bar
        ax.text(0, val, f" {val:.4f}", va="bottom", ha="center",
                fontsize=9, fontweight="bold")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Value")
        ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Patient Biomarker Values vs. Normal Reference Ranges",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# =============================================================================
# CHART 2 — SHAP Feature Importance  (clean default matplotlib style)
# =============================================================================
def plot_shap(models, tsh, fti, ratio):
    """
    Horizontal bar chart showing SHAP values for each feature.
    Red = pushes toward Positive; Blue = pushes toward Negative.
    """
    X_sample  = np.array([[tsh, fti, ratio]])
    shap_vals = models["explainer"].shap_values(X_sample)
    sv        = shap_vals[0] if np.array(shap_vals).ndim >= 2 else shap_vals[0]

    features = models["feature_names"]
    colors   = ["red" if v > 0 else "steelblue" for v in sv]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(features, sv, color=colors, alpha=0.80, height=0.45)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("SHAP Value  (positive → Diseased, negative → Healthy)")
    ax.set_title("Feature Impact — SHAP Explanation", fontsize=11, fontweight="bold")

    # Annotate bar values
    for bar, v in zip(bars, sv):
        offset = 0.0005 if v >= 0 else -0.0005
        ha     = "left"  if v >= 0 else "right"
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:+.4f}", va="center", ha=ha, fontsize=9)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color="red",      label="Pushes toward Positive"),
                      Patch(color="steelblue", label="Pushes toward Negative")]
    ax.legend(handles=legend_handles, fontsize=8)
    plt.tight_layout()
    return fig, sv


# =============================================================================
# CHART 3 — Model Accuracy Comparison  (clean default matplotlib style)
# =============================================================================
def plot_model_comparison(xg_acc, rf_acc):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = ["XGBoost", "Random Forest"]
    accs  = [xg_acc * 100, rf_acc * 100]
    bars  = ax.bar(names, accs, width=0.4, alpha=0.80)
    ax.set_ylim(80, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison", fontsize=11, fontweight="bold")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.3,
                f"{acc:.2f}%", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


# =============================================================================
# MODEL COMPARISON TABLE
# =============================================================================
def show_model_comparison_table(models):
    data = {
        "Model":         ["XGBoost", "Random Forest"],
        "Accuracy (%)":  [f"{models['xg_acc']*100:.2f}", f"{models['rf_acc']*100:.2f}"],
        "Type":          ["Gradient Boosting", "Ensemble Bagging"],
        "Trees":         ["150", "150"],
        "Selected":      ["✅ Yes (Primary)", "—"]
    }
    df = pd.DataFrame(data)
    st.markdown(df.to_html(index=False, classes="styled-table", border=0),
                unsafe_allow_html=True)


# =============================================================================
# NATURAL LANGUAGE EXPLANATION
# =============================================================================
def build_explanation(tsh, fti, ratio, pred_label, sv, feature_names):
    items = []
    if pred_label == "POSITIVE":
        items.append(("🔴", "The model predicts a <b>Positive</b> thyroid condition based on your biomarker profile."))
    else:
        items.append(("🟢", "The model predicts a <b>Negative (Healthy)</b> thyroid status based on your biomarker profile."))

    # TSH explanation
    if not (0.4 <= tsh <= 4.0):
        dir_tsh = "elevated" if tsh > 4.0 else "suppressed"
        items.append(("🔬", f"Your <b>TSH ({tsh:.3f} mIU/L)</b> is {dir_tsh} "
                      f"(normal: 0.4–4.0), which {'increased' if sv[0]>0 else 'decreased'} disease likelihood."))
    else:
        items.append(("✅", f"Your <b>TSH ({tsh:.3f} mIU/L)</b> is within the normal range (0.4–4.0 mIU/L)."))

    # FTI explanation
    if not (60 <= fti <= 160):
        dir_fti = "high" if fti > 160 else "low"
        items.append(("🔬", f"Your <b>FTI ({fti:.1f})</b> is {dir_fti} "
                      f"(normal: 60–160), influencing result {'toward disease' if sv[1]>0 else 'toward healthy'}."))
    else:
        items.append(("✅", f"Your <b>FTI ({fti:.1f})</b> is within the normal range (60–160)."))

    # Ratio explanation
    if not (0.003 <= ratio <= 0.067):
        items.append(("🔬", f"The <b>TSH/FTI Ratio ({ratio:.5f})</b> is outside the expected range (0.003–0.067), "
                      f"further {'supporting' if sv[2]>0 else 'reducing'} the positive prediction."))
    else:
        items.append(("✅", f"The <b>TSH/FTI Ratio ({ratio:.5f})</b> is within the normal range."))

    return items


# =============================================================================
# REPORT GENERATOR — PROFESSIONAL PDF (ReportLab)
# =============================================================================

# ── Colour palette (matches UI) ──
NAVY       = colors.HexColor("#0F2041")
BLUE       = colors.HexColor("#1565C0")
LIGHT_BLUE = colors.HexColor("#EEF4FF")
GREEN_D    = colors.HexColor("#2E7D32")
LIGHT_GRN  = colors.HexColor("#E8F5E9")
RED_D      = colors.HexColor("#C62828")
LIGHT_RED  = colors.HexColor("#FFEBEE")
GREY_LINE  = colors.HexColor("#E8EDF5")
DARK_TEXT  = colors.HexColor("#1A1A2E")
MID_TEXT   = colors.HexColor("#6B7A99")


def _pdf_styles():
    """Return a dict of named ParagraphStyles for the report."""
    return {
        "title": ParagraphStyle("RPT_title",
            fontName="Helvetica-Bold", fontSize=18,
            textColor=colors.white, alignment=TA_LEFT, spaceAfter=2),
        "subtitle": ParagraphStyle("RPT_subtitle",
            fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#B0C4E8"), alignment=TA_LEFT),
        "section": ParagraphStyle("RPT_section",
            fontName="Helvetica-Bold", fontSize=10.5,
            textColor=NAVY, spaceBefore=4, spaceAfter=4),
        "body": ParagraphStyle("RPT_body",
            fontName="Helvetica", fontSize=9.5,
            textColor=DARK_TEXT, leading=15, spaceAfter=4),
        "small": ParagraphStyle("RPT_small",
            fontName="Helvetica", fontSize=8.5,
            textColor=MID_TEXT, leading=13),
        "disclaimer": ParagraphStyle("RPT_disclaimer",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=MID_TEXT, leading=13, alignment=TA_CENTER),
        "result_pos": ParagraphStyle("RPT_rpos",
            fontName="Helvetica-Bold", fontSize=18,
            textColor=RED_D, alignment=TA_CENTER, spaceAfter=4),
        "result_neg": ParagraphStyle("RPT_rneg",
            fontName="Helvetica-Bold", fontSize=18,
            textColor=GREEN_D, alignment=TA_CENTER, spaceAfter=4),
        "conf": ParagraphStyle("RPT_conf",
            fontName="Helvetica-Bold", fontSize=11,
            textColor=DARK_TEXT, alignment=TA_CENTER, spaceAfter=2),
        "footer": ParagraphStyle("RPT_footer",
            fontName="Helvetica", fontSize=7.5,
            textColor=MID_TEXT, alignment=TA_CENTER),
    }


def _header_block(styles, now_str):
    tbl = Table([[
        Paragraph("ThyroPredict", styles["title"]),
        Paragraph(
            "Clinical Decision Support System<br/>"
            "Thyroid Disease Screening Report<br/>"
            f"<font size='8' color='#90B0D8'>Generated: {now_str}</font>",
            styles["subtitle"]),
    ]], colWidths=[5.5*cm, 12*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), NAVY),
        ("ROWPADDING",  (0, 0), (-1, -1), 16),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW",   (0, 0), (-1, -1), 3, BLUE),
    ]))
    return tbl


def _section_heading(text, styles):
    tbl = Table([[Paragraph(text, styles["section"])]], colWidths=[17.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), LIGHT_BLUE),
        ("ROWPADDING",  (0, 0), (-1, -1), 6),
        ("LINEBELOW",   (0, 0), (-1, -1), 1.5, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return tbl


def _status_para(val, lo, hi, styles):
    ok  = lo <= val <= hi
    lbl = "Normal" if ok else ("High" if val > hi else "Low")
    hex_c = "2E7D32" if ok else "C62828"
    return Paragraph(f'<font color="#{hex_c}"><b>{lbl}</b></font>', styles["body"])


def _patient_table(age, tsh, fti, ratio, styles):
    data = [
        [Paragraph("<b>Parameter</b>", styles["body"]),
         Paragraph("<b>Value</b>",     styles["body"]),
         Paragraph("<b>Normal Range</b>", styles["body"]),
         Paragraph("<b>Status</b>",    styles["body"])],
        [Paragraph("Age", styles["body"]),
         Paragraph(f"{age} years", styles["body"]), "—", "—"],
        [Paragraph("TSH (mIU/L)", styles["body"]),
         Paragraph(f"<b>{tsh:.4f}</b>", styles["body"]),
         "0.4 – 4.0", _status_para(tsh, 0.4, 4.0, styles)],
        [Paragraph("FTI", styles["body"]),
         Paragraph(f"<b>{fti:.2f}</b>", styles["body"]),
         "60 – 160",  _status_para(fti, 60, 160, styles)],
        [Paragraph("TSH / FTI Ratio", styles["body"]),
         Paragraph(f"<b>{ratio:.6f}</b>", styles["body"]),
         "0.003 – 0.067", _status_para(ratio, 0.003, 0.067, styles)],
    ]
    tbl = Table(data, colWidths=[4.5*cm, 4*cm, 4.5*cm, 4.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWPADDING",     (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
        ("GRID",           (0, 0), (-1, -1), 0.5, GREY_LINE),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _result_block(pred_label, confidence, prob, styles):
    is_pos   = pred_label == "POSITIVE"
    bg       = LIGHT_RED if is_pos else LIGHT_GRN
    border_c = RED_D     if is_pos else GREEN_D
    icon     = "POSITIVE — Thyroid Disease Detected" if is_pos else "NEGATIVE — No Disease Detected"
    txt_sty  = styles["result_pos"] if is_pos else styles["result_neg"]
    inner = Table([
        [Paragraph(icon, txt_sty)],
        [Paragraph(f"Confidence: {confidence:.1f}%", styles["conf"])],
        [Paragraph(
            f"Probability Healthy: {prob[0]*100:.1f}%     |     "
            f"Probability Diseased: {prob[1]*100:.1f}%", styles["small"])],
    ], colWidths=[17.5*cm])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("ROWPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("LINEABOVE",  (0, 0), (-1, 0),  2.5, border_c),
        ("LINEBELOW",  (0, -1), (-1, -1), 2.5, border_c),
    ]))
    return inner


def _shap_table(shap_vals, feature_names, styles):
    data = [[
        Paragraph("<b>Feature</b>", styles["body"]),
        Paragraph("<b>SHAP Value</b>", styles["body"]),
        Paragraph("<b>Direction</b>", styles["body"]),
        Paragraph("<b>Strength</b>", styles["body"]),
    ]]
    for fname, sv in zip(feature_names, shap_vals):
        direction = "Toward Positive" if sv > 0 else "Toward Negative"
        hex_c     = "C62828"          if sv > 0 else "1565C0"
        strength  = min(int(abs(sv) / 0.02), 10)
        bar_str   = ("|||" * strength).ljust(30)[:30]
        data.append([
            Paragraph(fname, styles["body"]),
            Paragraph(f'<font color="#{hex_c}"><b>{sv:+.5f}</b></font>', styles["body"]),
            Paragraph(f'<font color="#{hex_c}">{direction}</font>', styles["small"]),
            Paragraph(f'<font color="#{hex_c}">{bar_str}</font>', styles["small"]),
        ])
    tbl = Table(data, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWPADDING",     (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
        ("GRID",           (0, 0), (-1, -1), 0.5, GREY_LINE),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def generate_pdf_report(age, tsh, fti, ratio, pred_label, confidence, prob, sv, feature_names):
    """
    Build and return a professional PDF clinical report as bytes.
    Uses ReportLab Platypus for multi-section, styled layout.
    """
    buf     = io.BytesIO()
    now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    styles  = _pdf_styles()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.8*cm, bottomMargin=2*cm,
        title="ThyroPredict Clinical Report",
        author="ThyroPredict AI System",
    )

    story = []

    # 1 ── Header ────────────────────────────────────────────
    story.append(_header_block(styles, now_str))
    story.append(Spacer(1, 12))

    # 2 ── Patient Input ──────────────────────────────────────
    story.append(_section_heading("1.  Patient Biomarker Input", styles))
    story.append(Spacer(1, 6))
    story.append(_patient_table(age, tsh, fti, ratio, styles))
    story.append(Spacer(1, 12))

    # 3 ── Prediction Result ──────────────────────────────────
    story.append(_section_heading("2.  Prediction Result", styles))
    story.append(Spacer(1, 8))
    story.append(_result_block(pred_label, confidence, prob, styles))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Primary Model: <b>XGBoost</b> (Gradient Boosting, 150 estimators, max_depth=6)",
        styles["small"]))
    story.append(Spacer(1, 12))

    # 4 ── SHAP Explainability ────────────────────────────────
    story.append(_section_heading("3.  Explainable AI — SHAP Feature Contributions", styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "SHAP (SHapley Additive exPlanations) values indicate each biomarker's contribution to the "
        "prediction. Positive SHAP values push toward <b>disease</b>; negative values push toward <b>healthy</b>.",
        styles["body"]))
    story.append(Spacer(1, 6))
    story.append(_shap_table(sv, feature_names, styles))
    story.append(Spacer(1, 12))

    # 5 ── Clinical Interpretation ────────────────────────────
    story.append(_section_heading("4.  Clinical Interpretation & Recommendation", styles))
    story.append(Spacer(1, 6))
    is_pos = pred_label == "POSITIVE"
    if is_pos:
        story.append(Paragraph(
            "The AI model has detected a <b>POSITIVE</b> thyroid condition. "
            "The following biomarkers contributed most to this result:", styles["body"]))
        for fname, val in zip(feature_names, sv):
            if val > 0:
                story.append(Paragraph(
                    f"&nbsp;&nbsp;&nbsp;• <b>{fname}</b> pushed toward disease (SHAP = {val:+.5f}).",
                    styles["body"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "<b>Recommendation:</b> Refer patient to an endocrinologist for confirmatory tests including "
            "Total T4, Free T3, thyroid antibody panel (TPO-Ab, TG-Ab), and thyroid ultrasound.",
            styles["body"]))
    else:
        story.append(Paragraph(
            "The AI model predicts a <b>NEGATIVE (Healthy)</b> thyroid status. Biomarker values are "
            "within or near clinically accepted normal ranges.", styles["body"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "<b>Recommendation:</b> No immediate clinical intervention is required. "
            "Continue routine annual thyroid function screening as per clinical guidelines.",
            styles["body"]))
    story.append(Spacer(1, 20))

    # 6 ── Disclaimer ─────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.8, color=GREY_LINE))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an Artificial Intelligence model for research and educational "
        "purposes only. It does NOT constitute professional medical advice and must NOT replace a qualified "
        "clinical diagnosis. Always consult a licensed physician or endocrinologist.",
        styles["disclaimer"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "ThyroPredict v2.0  |  AI-Powered Clinical Decision Support  |  Thesis Research Project",
        styles["footer"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def show_dashboard(models):
    show_sidebar(models)

    # Hero banner
    st.markdown("""
    <div class='hero'>
        <h1>🩺 ThyroPredict — Clinical Dashboard</h1>
        <p>AI-powered thyroid disease screening using TSH, FTI, and TSH/FTI Ratio biomarkers</p>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs(["🩺 Prediction", "📊 Visualisation", "🧠 Explainability",
                    "⚖️ Model Comparison", "📄 Report"])

    # =========================================================================
    # TAB 1 — PREDICTION
    # =========================================================================
    with tabs[0]:
        st.markdown("<div class='section-heading'>Patient Biomarker Input</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Enter the patient's biomarker values. "
                    "The TSH/FTI ratio is computed automatically. "
                    "All predictions use a trained XGBoost model.</div>", unsafe_allow_html=True)

        # Input row
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("👤 Age (years)", min_value=1, max_value=120, value=35, step=1)
        with c2:
            tsh = st.number_input("🧪 TSH (mIU/L)", min_value=0.01, max_value=100.0,
                                  value=2.50, step=0.01, format="%.2f")
        with c3:
            fti = st.number_input("🧬 FTI", min_value=1.0, max_value=500.0,
                                  value=100.0, step=0.1, format="%.1f")

        # Auto ratio
        ratio = tsh / (fti + 1e-9)
        st.markdown(f"""
        <div class='card' style='background:#EEF4FF;border:1.5px solid #BBDEFB;'>
            <div class='card-title'>Auto-Calculated TSH/FTI Ratio</div>
            <div class='card-value' style='font-size:1.6rem;color:#1565C0;'>{ratio:.6f}</div>
            <div class='card-sub'>TSH ÷ FTI &nbsp;=&nbsp; {tsh:.4f} ÷ {fti:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Validation + predict button
        errors = validate_inputs(age, tsh, fti)
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            if st.button("🔍 Run Prediction"):
                X_input   = np.array([[tsh, fti, ratio]])
                pred      = models["xg"].predict(X_input)[0]
                prob      = models["xg"].predict_proba(X_input)[0]
                conf      = prob[pred] * 100
                pred_label = "POSITIVE" if pred == 1 else "NEGATIVE"

                shap_raw = models["explainer"].shap_values(X_input)
                sv = np.array(shap_raw)[0] if np.array(shap_raw).ndim >= 2 else np.array(shap_raw)[0]

                st.session_state.results = {
                    "age": age, "tsh": tsh, "fti": fti, "ratio": ratio,
                    "pred": pred, "pred_label": pred_label,
                    "confidence": conf, "prob": prob, "sv": sv
                }
                st.session_state.prediction_done = True

        # ── Result — only rendered after prediction ──
        if st.session_state.prediction_done and st.session_state.results:
            r = st.session_state.results
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-heading'>Prediction Result</div>", unsafe_allow_html=True)

            # Full-width result banner (no empty right column)
            css  = "result-positive" if r["pred_label"] == "POSITIVE" else "result-negative"
            icon = "🔴" if r["pred_label"] == "POSITIVE" else "🟢"
            st.markdown(f"""
            <div class='{css}'>
                <div class='result-label'>{icon} {r["pred_label"]}</div>
                <div class='result-conf'>Confidence: {r["confidence"]:.1f}%</div>
                <div style='margin-top:0.5rem;font-size:0.84rem;color:#555;'>
                    Probability Healthy: {r["prob"][0]*100:.1f}% &nbsp;|&nbsp;
                    Probability Diseased: {r["prob"][1]*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Biomarker summary — 3 HTML cards (no st.metric white boxes)
            tsh_status = ("🔴 High"   if r["tsh"] > 4.0  else
                          "🔵 Low"    if r["tsh"] < 0.4   else "✅ Normal")
            fti_status = ("🔴 High"   if r["fti"] > 160   else
                          "🔵 Low"    if r["fti"] < 60     else "✅ Normal")
            rat_status = ("🔴 High"   if r["ratio"] > 0.067 else
                          "🔵 Low"    if r["ratio"] < 0.003  else "✅ Normal")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class='card'>
                    <div class='card-title'>TSH Level</div>
                    <div class='card-value'>{r['tsh']:.3f}</div>
                    <div class='card-sub'>mIU/L &nbsp;·&nbsp; {tsh_status}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='card'>
                    <div class='card-title'>FTI Level</div>
                    <div class='card-value'>{r['fti']:.1f}</div>
                    <div class='card-sub'>Index &nbsp;·&nbsp; {fti_status}</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class='card'>
                    <div class='card-title'>TSH / FTI Ratio</div>
                    <div class='card-value'>{r['ratio']:.5f}</div>
                    <div class='card-sub'>Ratio &nbsp;·&nbsp; {rat_status}</div>
                </div>""", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2 — VISUALISATION
    # =========================================================================
    with tabs[1]:
        st.markdown("<div class='section-heading'>Biomarker Values vs Normal Reference Ranges</div>",
                    unsafe_allow_html=True)
        if st.session_state.prediction_done and st.session_state.results:
            r = st.session_state.results
            fig = plot_input_vs_normal(r["tsh"], r["fti"], r["ratio"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("""
            <div class='info-box'>
            <b>Reading the chart:</b> The green-shaded band is the normal clinical range.
            A <b style='color:green;'>green bar</b> means the value is within normal limits;
            a <b style='color:red;'>red bar</b> means the value is outside the normal range.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("ℹ️ Run a prediction first (Tab 1) to see the visualisation.")

    # =========================================================================
    # TAB 3 — EXPLAINABILITY
    # =========================================================================
    with tabs[2]:
        st.markdown("<div class='section-heading'>Explainable AI — SHAP Analysis</div>",
                    unsafe_allow_html=True)
        if st.session_state.prediction_done and st.session_state.results:
            r = st.session_state.results
            fig_shap, _ = plot_shap(models, r["tsh"], r["fti"], r["ratio"])
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)

            st.markdown("<div class='section-heading' style='margin-top:1.5rem;'>"
                        "Natural Language Explanation</div>", unsafe_allow_html=True)
            for icon, text in build_explanation(r["tsh"], r["fti"], r["ratio"],
                                                r["pred_label"], r["sv"],
                                                models["feature_names"]):
                st.markdown(f"""
                <div class='explain-item'>
                    <span class='explain-icon'>{icon}</span>
                    <span>{text}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>")
            st.markdown("""
            <div class='info-box'>
            <b>What is SHAP?</b> SHAP (SHapley Additive exPlanations) assigns each feature an
            importance score for a specific prediction. <b style='color:red;'>Red bars</b> push
            toward <em>Positive (diseased)</em>; <b style='color:steelblue;'>blue bars</b> push
            toward <em>Negative (healthy)</em>.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("ℹ️ Run a prediction first (Tab 1) to see the explainability analysis.")

    # =========================================================================
    # TAB 4 — MODEL COMPARISON
    # =========================================================================
    with tabs[3]:
        st.markdown("<div class='section-heading'>Model Performance Comparison</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        Both models were trained on the same synthetic thyroid dataset with an identical 75/25
        train-test split. XGBoost is selected as the primary model for its gradient-boosting
        accuracy and native SHAP compatibility.
        </div>""", unsafe_allow_html=True)

        show_model_comparison_table(models)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_cmp = plot_model_comparison(models["xg_acc"], models["rf_acc"])
        st.pyplot(fig_cmp, use_container_width=False)
        plt.close(fig_cmp)

        st.markdown("""
        <div class='card' style='margin-top:1rem;'>
            <div class='card-title'>Why XGBoost?</div>
            <div style='font-size:0.87rem;color:#334;line-height:1.75;'>
            XGBoost uses gradient boosting — iteratively correcting errors from prior trees —
            which generally yields higher accuracy than Random Forest's parallel bagging strategy
            on tabular biomedical data. It also integrates natively with SHAP, enabling robust
            explainability required for clinical AI deployment.
            </div>
        </div>""", unsafe_allow_html=True)

    # =========================================================================
    # TAB 5 — REPORT
    # =========================================================================
    with tabs[4]:
        st.markdown("<div class='section-heading'>Downloadable Clinical Report</div>",
                    unsafe_allow_html=True)
        if st.session_state.prediction_done and st.session_state.results:
            r = st.session_state.results
            report = generate_report(
                r["age"], r["tsh"], r["fti"], r["ratio"],
                r["pred_label"], r["confidence"],
                r["sv"], models["feature_names"]
            )
            st.code(report, language="text")
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=report.encode("utf-8"),
                file_name=f"ThyroPredict_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            st.markdown("""
            <div class='info-box' style='margin-top:1rem;'>
            ⚠️ <b>Disclaimer:</b> This report is AI-generated for research and educational
            purposes only. It does not constitute medical advice and must not replace
            professional clinical diagnosis.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("ℹ️ Run a prediction first (Tab 1) to generate the report.")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    if not st.session_state.logged_in:
        show_login()
        return

    with st.spinner("🔄 Initialising models…"):
        models = train_models()

    show_dashboard(models)


if __name__ == "__main__":
    main()
