import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import warnings
from datetime import datetime
from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Explainable Thyroid AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# GLOBAL STYLE
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #07111f 0%, #0b1220 100%);
        color: #e5e7eb;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    [data-testid="stToolbar"] {
        right: 12px;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #0b1324 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    h1, h2, h3, h4, p, label, span, div {
        color: #e5e7eb;
    }
    .top-banner {
        background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(168,85,247,0.18));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        margin-bottom: 0.8rem;
    }
    .result-pos {
        background: linear-gradient(135deg, rgba(239,68,68,0.16), rgba(127,29,29,0.35));
        border: 1px solid rgba(239,68,68,0.35);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        animation: pulse 1.8s infinite;
    }
    .result-neg {
        background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(20,83,45,0.35));
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    }
    .recommendation {
        background: rgba(15, 23, 42, 0.82);
        border-left: 5px solid #38bdf8;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    }
    .small-muted {
        color: #94a3b8;
        font-size: 0.92rem;
    }
    @keyframes pulse {
        0% { transform: scale(1.0); box-shadow: 0 0 0 rgba(239,68,68,0.15); }
        50% { transform: scale(1.01); box-shadow: 0 0 18px rgba(239,68,68,0.18); }
        100% { transform: scale(1.0); box-shadow: 0 0 0 rgba(239,68,68,0.15); }
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# NORMAL RANGES
# =========================================================
TSH_NORMAL = (0.4, 4.0)
FTI_NORMAL = (60.0, 160.0)
RATIO_NORMAL = (0.003, 0.067)

# =========================================================
# HELPERS
# =========================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def detect_target_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "binaryClass", "target", "Target", "Class", "class",
        "label", "Label", "Outcome", "diagnosis", "Diagnosis", "Result"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_age_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() == "age":
            return c
    return None


def is_positive_label(value) -> bool:
    s = str(value).strip().lower()
    positive_tokens = ["positive", "disease", "diseased", "abnormal", "hyper", "hypo", "yes", "1"]
    negative_tokens = ["negative", "normal", "healthy", "no", "0"]

    if any(tok in s for tok in positive_tokens) and not any(tok in s for tok in negative_tokens):
        return True
    if any(tok in s for tok in negative_tokens):
        return False

    try:
        return int(float(value)) == 1
    except Exception:
        return False


def label_to_text(value) -> str:
    return "Positive" if is_positive_label(value) else "Negative"


def safe_feature_list(model, df: pd.DataFrame, target_col: str | None) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in model.feature_names_in_]
    cols = [c for c in df.columns if c != target_col]
    numeric_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return numeric_cols


def align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    df = clean_columns(df)
    out = df.copy()
    for col in feature_names:
        if col not in out.columns:
            out[col] = 0
    out = out[feature_names]
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0)
    return out


def build_input_row(
    feature_names: list[str],
    age: int,
    sex_val: int,
    tsh: float,
    fti: float,
) -> pd.DataFrame:
    ratio = tsh / (fti + 0.001)
    age_group = 0 if age < 30 else (1 if age <= 60 else 2)

    row = {col: 0 for col in feature_names}
    lookup = {c.lower(): c for c in feature_names}

    def set_if_present(name: str, value):
        key = lookup.get(name.lower())
        if key is not None:
            row[key] = value

    # Strongly relevant fields
    set_if_present("age", age)
    set_if_present("sex", sex_val)
    set_if_present("TSH", tsh)
    set_if_present("FTI", fti)
    set_if_present("TSH_FTI_Ratio", ratio)
    set_if_present("Age_Group", age_group)
    set_if_present("Symptom_Score", 0)

    # Measured flags if present
    set_if_present("TSH measured", 1)
    set_if_present("FTI measured", 1)
    set_if_present("T3 measured", 0)
    set_if_present("TT4 measured", 0)
    set_if_present("T4U measured", 0)

    # Common thyroid-history flags if present
    for col_name in [
        "on thyroxine", "query on thyroxine", "on antithyroid medication",
        "sick", "pregnant", "thyroid surgery", "I131 treatment",
        "query hypothyroid", "query hyperthyroid", "lithium", "goitre",
        "tumor", "hypopituitary", "psych"
    ]:
        set_if_present(col_name, 0)

    return pd.DataFrame([row])


def create_pdf_report(name, age, sex, tsh, fti, ratio, verdict, confidence, recommendation, risk):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Thyroid AI Diagnostic Report", ln=True, align="C")

    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Name: {name}", ln=True)
    pdf.cell(0, 7, f"Age: {age}", ln=True)
    pdf.cell(0, 7, f"Sex: {sex}", ln=True)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Clinical Parameters", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"TSH: {tsh:.3f}", ln=True)
    pdf.cell(0, 7, f"FTI: {fti:.3f}", ln=True)
    pdf.cell(0, 7, f"TSH/FTI Ratio: {ratio:.5f}", ln=True)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "AI Result", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Prediction: {verdict}", ln=True)
    pdf.cell(0, 7, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 7, f"Risk Level: {risk}", ln=True)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Clinical Recommendation", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, recommendation)

    return pdf.output(dest="S").encode("latin-1")


def make_live_chart(tsh, fti, verdict_text=None):
    point_color = "#94a3b8"
    if verdict_text == "Positive":
        point_color = "#ef4444"
    elif verdict_text == "Negative":
        point_color = "#22c55e"

    fig = go.Figure()

    # Normal zone rectangle
    fig.add_shape(
        type="rect",
        x0=FTI_NORMAL[0], x1=FTI_NORMAL[1],
        y0=TSH_NORMAL[0], y1=TSH_NORMAL[1],
        fillcolor="rgba(34,197,94,0.16)",
        line=dict(color="rgba(34,197,94,0.55)", width=2),
        layer="below",
    )

    # Reference lines
    fig.add_vline(x=FTI_NORMAL[0], line_width=1, line_dash="dash", line_color="rgba(34,197,94,0.55)")
    fig.add_vline(x=FTI_NORMAL[1], line_width=1, line_dash="dash", line_color="rgba(34,197,94,0.55)")
    fig.add_hline(y=TSH_NORMAL[0], line_width=1, line_dash="dash", line_color="rgba(34,197,94,0.55)")
    fig.add_hline(y=TSH_NORMAL[1], line_width=1, line_dash="dash", line_color="rgba(34,197,94,0.55)")

    fig.add_trace(
        go.Scatter(
            x=[fti],
            y=[tsh],
            mode="markers+text",
            text=["Patient"],
            textposition="top center",
            marker=dict(size=16, color=point_color, line=dict(color="white", width=1.5)),
            name="Patient Point",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=470,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Real-time TSH vs FTI Chart",
        xaxis_title="FTI",
        yaxis_title="TSH",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def doctor_recommendation(tsh, fti, ratio, verdict_text, confidence):
    if tsh > TSH_NORMAL[1] and fti < FTI_NORMAL[0]:
        rec = "Possible hypothyroid indication. Recommend endocrinologist consultation."
        risk = "High"
    elif tsh < TSH_NORMAL[0] and fti > FTI_NORMAL[1]:
        rec = "Possible hyperthyroid indication. Recommend endocrinologist consultation."
        risk = "High"
    elif verdict_text == "Positive":
        rec = "Abnormal thyroid pattern detected. Clinical review is advised to confirm the diagnosis."
        risk = "Medium" if confidence >= 70 else "High"
    else:
        rec = "Hormone balance appears stable. Routine follow-up is reasonable."
        risk = "Low" if confidence >= 80 else "Medium"

    # Confidence-based refinement
    if confidence < 70:
        risk = "Medium" if risk == "Low" else risk

    return rec, risk


def confidence_explanation(tsh, fti, ratio, confidence):
    reasons = []

    tsh_normal = TSH_NORMAL[0] <= tsh <= TSH_NORMAL[1]
    fti_normal = FTI_NORMAL[0] <= fti <= FTI_NORMAL[1]
    ratio_normal = RATIO_NORMAL[0] <= ratio <= RATIO_NORMAL[1]

    if confidence < 70:
        if tsh_normal:
            reasons.append("TSH is close to the normal range, so the pattern is less clear.")
        if fti_normal:
            reasons.append("FTI is close to the normal range, so the model has weaker evidence.")
        if ratio_normal:
            reasons.append("TSH/FTI ratio is near the normal band, which makes the decision less certain.")
        if abs(tsh - TSH_NORMAL[1]) < 1 or abs(tsh - TSH_NORMAL[0]) < 1:
            reasons.append("TSH is borderline.")
        if abs(fti - FTI_NORMAL[0]) < 15 or abs(fti - FTI_NORMAL[1]) < 15:
            reasons.append("FTI is borderline.")
    elif confidence < 85:
        reasons.append("The input shows a mixed pattern: some values are normal while others are mildly abnormal.")
    else:
        reasons.append("The input values form a clear pattern, so the model is confident.")

    if not reasons:
        reasons.append("The model detected a strong clinical pattern.")

    return reasons


def positive_negative_banner(verdict_text, confidence):
    if verdict_text == "Positive":
        st.markdown(
            f"""
            <div class="result-pos">
                <h3 style="margin:0;">🚨 Positive</h3>
                <p style="margin:0.35rem 0 0 0;">Disease pattern detected</p>
                <p style="margin:0.35rem 0 0 0;">Confidence: <b>{confidence:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-neg">
                <h3 style="margin:0;">✅ Negative</h3>
                <p style="margin:0.35rem 0 0 0;">Healthy / stable pattern</p>
                <p style="margin:0.35rem 0 0 0;">Confidence: <b>{confidence:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_recommendation_card(recommendation, risk):
    st.markdown(
        f"""
        <div class="recommendation">
            <h4 style="margin-top:0; margin-bottom:0.3rem;">🩺 Doctor Recommendation</h4>
            <p style="margin:0.2rem 0 0.5rem 0;">{recommendation}</p>
            <p style="margin:0;"><b>Risk level:</b> {risk}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_confusion_figure(cm, title):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=420,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def build_metric_chart(metrics_df):
    plot_df = metrics_df.reset_index().rename(columns={"index": "Model"})
    long_df = plot_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
    fig = px.bar(
        long_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".2f",
        title="Multi-model Performance Comparison",
    )
    fig.update_layout(
        template="plotly_dark",
        height=470,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="Score",
    )
    return fig


def build_pie_chart(df, target_col):
    counts = df[target_col].value_counts(dropna=False)
    labels = [str(x) for x in counts.index]
    values = counts.values.tolist()
    fig = px.pie(
        names=labels,
        values=values,
        hole=0.42,
        title="Disease Distribution",
    )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_correlation_heatmap(df, exclude_col=None):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if exclude_col and exclude_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[exclude_col])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
    )
    fig.update_layout(template="plotly_dark", height=650, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_biomarker_trend(df):
    age_col = detect_age_column(df)
    columns = [c for c in ["TSH", "FTI"] if c in df.columns]

    if not columns:
        return None

    plot_df = df.copy()
    if age_col and pd.api.types.is_numeric_dtype(plot_df[age_col]):
        plot_df = plot_df.sort_values(age_col)
        x_col = age_col
    else:
        plot_df = plot_df.reset_index(drop=True)
        x_col = plot_df.index

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if "TSH" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df["TSH"], mode="lines", name="TSH"), secondary_y=False)
    if "FTI" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df["FTI"], mode="lines", name="FTI"), secondary_y=True)

    fig.update_layout(
        template="plotly_dark",
        title="Biomarker Trend",
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(title_text=age_col if age_col else "Sample Index")
    fig.update_yaxes(title_text="TSH", secondary_y=False)
    fig.update_yaxes(title_text="FTI", secondary_y=True)
    return fig


def get_top_feature_importance(explainer, df_train_features, feature_names, top_n=15):
    try:
        sample_n = min(120, len(df_train_features))
        sample = df_train_features.sample(sample_n, random_state=42) if len(df_train_features) > sample_n else df_train_features.copy()
        shap_values = explainer(sample)
        vals = np.abs(shap_values.values).mean(axis=0)
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": vals})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(top_n)
        return imp_df
    except Exception:
        return None


def build_bar_chart_from_df(df_plot, x, y, title):
    fig = px.bar(df_plot, x=x, y=y, text_auto=".3f", title=title)
    fig.update_layout(template="plotly_dark", height=470, margin=dict(l=10, r=10, t=50, b=10))
    return fig

# =========================================================
# LOAD USER FILE
# =========================================================
try:
    with open("users.json") as f:
        users = json.load(f)
except Exception as e:
    st.error(f"users.json ফাইলটি পাওয়া যায়নি. Error: {e}")
    st.stop()

# =========================================================
# LOGIN
# =========================================================
if "login" not in st.session_state:
    st.session_state.login = False

def login_ui():
    st.markdown(
        """
        <div class="top-banner">
            <h1 style="margin:0;">🧠 Explainable Thyroid AI</h1>
            <p style="margin:0.35rem 0 0 0;" class="small-muted">Secure portal for thesis demonstration and clinical decision support</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("🔐 Login")
    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    if st.button("Access Portal", type="primary", use_container_width=True):
        if username in users and str(users[username]) == str(password):
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid Login")

if not st.session_state.login:
    login_ui()
    st.stop()

# =========================================================
# LOAD RESOURCES
# =========================================================
@st.cache_resource
def load_core_model():
    model = joblib.load("thyroid_model.pkl")
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model)
    return model, explainer

@st.cache_data
def load_dataset():
    df = pd.read_csv("cleaned_dataset_Thyroid1.csv")
    df = clean_columns(df)
    return df

xgb_model, xgb_explainer = load_core_model()
df_raw = load_dataset()

target_col = detect_target_column(df_raw)
if target_col is None:
    st.error(
        "Dataset-এ target column পাওয়া যায়নি। "
        "Target column-এর নাম binaryClass / Class / target জাতীয় কিছু হতে হবে।"
    )
    st.stop()

feature_names = safe_feature_list(xgb_model, df_raw, target_col)
df_features = align_features(df_raw, feature_names)
y_all = df_raw[target_col]

# =========================================================
# TRAIN/TEST SPLIT FOR REAL METRICS + COMPARISON MODELS
# =========================================================
@st.cache_resource
def build_comparison_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, rf, lr

X_train, X_test, y_train, y_test, rf_model, lr_model = build_comparison_models(df_features, y_all)

def calc_scores(model, X_test, y_test):
    pred = model.predict(X_test)
    avg = "binary" if pd.Series(y_test).nunique() == 2 and set(pd.Series(y_test).dropna().astype(str).unique()) <= {"0", "1"} else "weighted"

    scores = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, pred, average=avg, zero_division=0),
        "CM": confusion_matrix(y_test, pred),
    }
    return scores

xgb_scores = calc_scores(xgb_model, X_test, y_test)
rf_scores = calc_scores(rf_model, X_test, y_test)
lr_scores = calc_scores(lr_model, X_test, y_test)

metrics_df = pd.DataFrame(
    {
        "XGBoost": {
            "Accuracy": xgb_scores["Accuracy"],
            "Precision": xgb_scores["Precision"],
            "Recall": xgb_scores["Recall"],
            "F1": xgb_scores["F1"],
        },
        "Random Forest": {
            "Accuracy": rf_scores["Accuracy"],
            "Precision": rf_scores["Precision"],
            "Recall": rf_scores["Recall"],
            "F1": rf_scores["F1"],
        },
        "Logistic Regression": {
            "Accuracy": lr_scores["Accuracy"],
            "Precision": lr_scores["Precision"],
            "Recall": lr_scores["Recall"],
            "F1": lr_scores["F1"],
        },
    }
).T

best_model_name = metrics_df["Accuracy"].idxmax()
best_model = {
    "XGBoost": xgb_model,
    "Random Forest": rf_model,
    "Logistic Regression": lr_model,
}[best_model_name]

best_cm = {
    "XGBoost": xgb_scores["CM"],
    "Random Forest": rf_scores["CM"],
    "Logistic Regression": lr_scores["CM"],
}[best_model_name]

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 ThyroPredict AI")
    st.caption("Explainable clinical dashboard")

    st.markdown("### 📊 Best Model")
    st.success(best_model_name)

    st.markdown("### 📈 Test Accuracy")
    st.metric("XGBoost", f"{xgb_scores['Accuracy']*100:.2f}%")
    st.metric("Random Forest", f"{rf_scores['Accuracy']*100:.2f}%")
    st.metric("Logistic Regression", f"{lr_scores['Accuracy']*100:.2f}%")

    st.markdown("### 🧪 Normal Reference Ranges")
    st.info(
        f"""
**TSH:** {TSH_NORMAL[0]} – {TSH_NORMAL[1]}  
**FTI:** {FTI_NORMAL[0]} – {FTI_NORMAL[1]}  
**TSH/FTI Ratio:** {RATIO_NORMAL[0]} – {RATIO_NORMAL[1]}
        """
    )

    st.markdown("### ℹ️ Dataset Overview")
    st.write(f"Rows: **{len(df_raw):,}**")
    st.write(f"Columns: **{len(df_raw.columns):,}**")
    st.write(f"Target: **{target_col}**")

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="top-banner">
        <h1 style="margin:0;">Explainable Machine Learning-based Smart System for Diagnosing Thyroid Disease</h1>
        <p style="margin:0.35rem 0 0 0;" class="small-muted">
            Real-time prediction, clinical recommendation, multi-model comparison, and explainable AI
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Prediction", "📈 Model Comparison", "🌍 Analytics", "📂 Batch Prediction"])

# =========================================================
# TAB 1: PREDICTION
# =========================================================
with tab1:
    st.markdown("### 📋 Patient Input")

    c1, c2, c3 = st.columns(3)

    with c1:
        patient_name = st.text_input("Patient Name", "Patient_01")
        age = st.slider("Age", 1, 100, 30)
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex_val = 1 if sex == "Male" else 0

    with c2:
        tsh = st.number_input("TSH", value=6.0, min_value=0.0, step=0.1)
        fti = st.number_input("FTI", value=50.0, min_value=0.0, step=0.1)

    with c3:
        ratio = tsh / (fti + 0.001)
        st.metric("TSH/FTI Ratio", f"{ratio:.5f}")
        st.metric("Age Group", "Youth" if age < 30 else ("Adult" if age <= 60 else "Senior"))
        st.caption("Move the inputs to update the live chart instantly.")

    # Live chart updates with inputs
    st.markdown("### 📊 Live TSH vs FTI Chart")
    live_fig = make_live_chart(tsh, fti, None)
    st.plotly_chart(live_fig, use_container_width=True)

    if st.button("🚀 Run Diagnosis", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            input_df = build_input_row(feature_names, age, sex_val, tsh, fti)

            pred_value = best_model.predict(input_df)[0]
            pred_text = label_to_text(pred_value)

            if hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(input_df)[0]
                classes = list(best_model.classes_) if hasattr(best_model, "classes_") else None
                if classes is not None and pred_value in classes:
                    confidence = float(proba[classes.index(pred_value)]) * 100
                else:
                    confidence = float(np.max(proba)) * 100
            else:
                confidence = 100.0 if pred_text == "Positive" else 0.0

            rec, risk = doctor_recommendation(tsh, fti, ratio, pred_text, confidence)
            reasons = confidence_explanation(tsh, fti, ratio, confidence)

        st.markdown("---")

        res_col1, res_col2 = st.columns([1.2, 1.0])

        with res_col1:
            positive_negative_banner(pred_text, confidence)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 🧠 Clinical Interpretation")
            st.write(f"**TSH:** {tsh:.3f}")
            st.write(f"**FTI:** {fti:.3f}")
            st.write(f"**TSH/FTI Ratio:** {ratio:.5f}")

            if tsh > TSH_NORMAL[1]:
                st.write("• TSH is high, which may indicate hypothyroid tendency.")
            elif tsh < TSH_NORMAL[0]:
                st.write("• TSH is low, which may indicate hyperthyroid tendency.")
            else:
                st.write("• TSH is within the normal range.")

            if fti < FTI_NORMAL[0]:
                st.write("• FTI is low, suggesting reduced thyroid hormone output.")
            elif fti > FTI_NORMAL[1]:
                st.write("• FTI is high, suggesting increased thyroid hormone output.")
            else:
                st.write("• FTI is within the normal range.")

            st.write(f"• TSH/FTI ratio helps the model capture hormone relationship and imbalance.")
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            render_recommendation_card(rec, risk)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 📊 Confidence Meter")
            st.progress(int(min(max(confidence, 0), 100)))

            if confidence >= 85:
                st.success(f"High confidence ({confidence:.2f}%)")
            elif confidence >= 70:
                st.warning(f"Medium confidence ({confidence:.2f}%)")
            else:
                st.error(f"Low confidence ({confidence:.2f}%)")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🔍 Why confidence is high / low")
        if confidence < 70:
            st.warning("The model is less certain because the input is close to normal reference values or shows a mixed pattern.")
        for r in reasons:
            st.write("•", r)

        st.markdown("### 🧠 Explainable AI (SHAP)")
        try:
            shap_values = xgb_explainer(input_df)
            fig, ax = plt.subplots(figsize=(11, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig, use_container_width=True, clear_figure=True)
        except Exception as e:
            st.info(f"SHAP waterfall could not be rendered in this run. ({e})")

        st.markdown("### 📈 Patient Live Chart")
        result_chart = make_live_chart(tsh, fti, pred_text)
        st.plotly_chart(result_chart, use_container_width=True)

        st.markdown("### 📄 Export Report")
        pdf_bytes = create_pdf_report(
            patient_name,
            age,
            sex,
            tsh,
            fti,
            ratio,
            pred_text,
            confidence,
            rec,
            risk,
        )
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"{patient_name}_thyroid_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# =========================================================
# TAB 2: MODEL COMPARISON
# =========================================================
with tab2:
    st.markdown("### 📊 Multi-model Performance")

    comparison_fig = build_metric_chart(metrics_df)
    st.plotly_chart(comparison_fig, use_container_width=True)

    best_row = metrics_df.loc[best_model_name]
    st.success(
        f"Best model automatically selected: **{best_model_name}** "
        f"with accuracy **{best_row['Accuracy']*100:.2f}%**"
    )

    st.markdown("### 📋 Comparison Table")
    show_df = (metrics_df * 100).round(2)
    st.dataframe(show_df, use_container_width=True)

    st.markdown("### 📉 Confusion Matrix of Best Model")
    best_cm_fig = build_confusion_figure(best_cm, f"{best_model_name} Confusion Matrix")
    st.plotly_chart(best_cm_fig, use_container_width=True)

# =========================================================
# TAB 3: ANALYTICS
# =========================================================
with tab3:
    st.markdown("### 🧾 Dataset Overview")

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        st.metric("Rows", f"{len(df_raw):,}")
    with o2:
        st.metric("Columns", f"{len(df_raw.columns):,}")
    with o3:
        st.metric("Features used", f"{len(feature_names):,}")
    with o4:
        missing_total = int(df_raw.isna().sum().sum())
        st.metric("Missing values", f"{missing_total:,}")

    st.markdown("### 🥧 Disease Distribution")
    pie_fig = build_pie_chart(df_raw, target_col)
    st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("### 🌡️ Correlation Heatmap")
    corr_fig = build_correlation_heatmap(df_raw, exclude_col=target_col)
    if corr_fig is not None:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Correlation heatmap could not be generated because numeric columns are insufficient.")

    st.markdown("### 📈 Biomarker Trend")
    trend_fig = build_biomarker_trend(df_raw)
    if trend_fig is not None:
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("Biomarker trend graph could not be generated.")

    st.markdown("### 🌍 Global Explainability")
    try:
        top_imp = get_top_feature_importance(xgb_explainer, df_features, feature_names, top_n=15)
        if top_imp is not None and not top_imp.empty:
            imp_fig = build_bar_chart_from_df(top_imp.sort_values("Importance", ascending=True), "Importance", "Feature", "Top SHAP Feature Importance")
            st.plotly_chart(imp_fig, use_container_width=True)
        else:
            st.info("Could not compute global feature importance in this session.")
    except Exception as e:
        st.info(f"Global explainability unavailable. ({e})")

# =========================================================
# TAB 4: BATCH PREDICTION
# =========================================================
with tab4:
    st.markdown("### 📂 Upload CSV for Batch Prediction")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        batch_df = clean_columns(batch_df)

        st.markdown("#### Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", use_container_width=True):
            with st.spinner("Running batch prediction..."):
                batch_aligned = align_features(batch_df, feature_names)

                preds = best_model.predict(batch_aligned)
                pred_texts = [label_to_text(p) for p in preds]

                batch_result = batch_df.copy()
                batch_result["AI_Result"] = pred_texts

                if hasattr(best_model, "predict_proba"):
                    probs = best_model.predict_proba(batch_aligned)
                    classes = list(best_model.classes_) if hasattr(best_model, "classes_") else None
                    confs = []
                    for i, p in enumerate(preds):
                        if classes is not None and p in classes:
                            confs.append(float(probs[i][classes.index(p)]) * 100)
                        else:
                            confs.append(float(np.max(probs[i])) * 100)
                    batch_result["Confidence"] = np.round(confs, 2)

            st.success("Batch prediction completed.")
            st.dataframe(batch_result, use_container_width=True)

            st.download_button(
                "Download Prediction CSV",
                data=batch_result.to_csv(index=False).encode("utf-8"),
                file_name="thyroid_batch_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("Upload a CSV file to run batch prediction.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding: 0.4rem 0 0.2rem 0;">
        <b>Explainable Machine Learning-based Smart System for Diagnosing Thyroid Disease</b><br/>
        Department of CSE, Notre Dame University Bangladesh<br/>
        Developed by Tanjil Hossain Midul
    </div>
    """,
    unsafe_allow_html=True,
)
