import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import json
from datetime import datetime
from fpdf import FPDF
from pathlib import Path
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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
# STYLE
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
# CLINICAL RANGES
# =========================================================
TSH_NORMAL = (0.4, 4.0)
FTI_NORMAL = (60.0, 160.0)
RATIO_NORMAL = (0.003, 0.067)

# =========================================================
# HELPERS
# =========================================================
def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df

def detect_target_column(df):
    candidates = [
        "binaryClass", "target", "Target", "Class", "class",
        "label", "Label", "Outcome", "diagnosis", "Diagnosis", "Result"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_target(series):
    if series.dtype == "O":
        s = series.astype(str).str.strip().str.lower()

        def map_target(v):
            v = str(v).strip().lower()
            if v in ["1", "true", "yes", "positive", "disease", "diseased", "abnormal", "hyper", "hypo", "present"]:
                return 1
            if v in ["0", "false", "no", "negative", "normal", "healthy", "ok", "absent"]:
                return 0
            if "positive" in v or "disease" in v or "abnormal" in v:
                return 1
            if "negative" in v or "normal" in v or "healthy" in v:
                return 0
            return 1 if v not in ["0", "false", "no"] else 0

        return s.map(map_target).astype(int)

    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

def normalize_for_model(df):
    df = clean_columns(df)

    if "TSH" in df.columns and "FTI" in df.columns and "TSH_FTI_Ratio" not in df.columns:
        df["TSH_FTI_Ratio"] = df["TSH"] / (df["FTI"].replace(0, np.nan) + 0.001)
        df["TSH_FTI_Ratio"] = df["TSH_FTI_Ratio"].fillna(0)

    if "age" in df.columns and "Age_Group" not in df.columns:
        try:
            df["Age_Group"] = pd.cut(
                pd.to_numeric(df["age"], errors="coerce").fillna(0),
                bins=[0, 29, 60, 120],
                labels=[0, 1, 2],
                include_lowest=True
            ).astype("int64")
        except Exception:
            df["Age_Group"] = 0

    if "Symptom_Score" not in df.columns:
        symptom_cols = [
            "on thyroxine", "query on thyroxine", "on antithyroid medication",
            "sick", "pregnant", "thyroid surgery", "I131 treatment",
            "query hypothyroid", "query hyperthyroid", "lithium", "goitre",
            "tumor", "hypopituitary", "psych"
        ]
        existing = [c for c in symptom_cols if c in df.columns]
        if existing:
            df["Symptom_Score"] = df[existing].sum(axis=1)
        else:
            df["Symptom_Score"] = 0

    return df

@st.cache_data
def load_dataset():
    df = pd.read_csv("cleaned_dataset_Thyroid1.csv")
    return clean_columns(df)

@st.cache_resource
def load_model_file(path):
    return joblib.load(path)

def get_available_model_files():
    model_files = {
        "XGBoost": "xgboost_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "SVM": "svm_model.pkl",
    }
    available = {}
    for name, file in model_files.items():
        if Path(file).exists():
            available[name] = file

    if not available and Path("thyroid_model.pkl").exists():
        available["Best Model"] = "thyroid_model.pkl"

    return available

def align_features(df, feature_columns):
    df = clean_columns(df)
    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0
    out = out[feature_columns]
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0)
    return out

def build_default_raw_row(template_df):
    row = {}
    for col in template_df.columns:
        series = template_df[col]
        if pd.api.types.is_numeric_dtype(series):
            vals = pd.to_numeric(series, errors="coerce").dropna()
            row[col] = float(vals.median()) if len(vals) else 0.0
        else:
            vals = series.astype(str).dropna()
            mode = vals.mode(dropna=True)
            row[col] = mode.iloc[0] if len(mode) else ""
    return row

def build_patient_raw_input(template_df, age, sex, tsh, fti):
    row = build_default_raw_row(template_df)

    if "age" in row:
        row["age"] = age
    if "Age" in row:
        row["Age"] = age

    if "sex" in row:
        if pd.api.types.is_numeric_dtype(template_df["sex"]):
            row["sex"] = 1 if sex == "Male" else 0
        else:
            row["sex"] = sex
    if "Sex" in row:
        if pd.api.types.is_numeric_dtype(template_df["Sex"]):
            row["Sex"] = 1 if sex == "Male" else 0
        else:
            row["Sex"] = sex

    if "TSH" in row:
        row["TSH"] = tsh
    if "FTI" in row:
        row["FTI"] = fti
    if "TSH_FTI_Ratio" in row:
        row["TSH_FTI_Ratio"] = tsh / (fti + 0.001)

    if "Age_Group" in row:
        if age < 30:
            row["Age_Group"] = 0
        elif age <= 60:
            row["Age_Group"] = 1
        else:
            row["Age_Group"] = 2

    if "Symptom_Score" in row:
        row["Symptom_Score"] = 0

    for measured in ["TSH measured", "FTI measured", "T3 measured", "TT4 measured", "T4U measured"]:
        if measured in row:
            row[measured] = 1 if measured in ["TSH measured", "FTI measured"] else 0

    return pd.DataFrame([row])

def prepare_input_features(patient_df, feature_columns):
    patient_df = normalize_for_model(patient_df)
    patient_df = pd.get_dummies(patient_df, drop_first=False)
    patient_df = align_features(patient_df, feature_columns)
    return patient_df

def choose_confidence(pred_class, proba, classes):
    if proba is None:
        return 100.0
    if classes is not None and pred_class in classes:
        idx = list(classes).index(pred_class)
        return float(proba[idx]) * 100
    return float(np.max(proba)) * 100

def label_to_text(value):
    try:
        return "Positive" if int(value) == 1 else "Negative"
    except Exception:
        s = str(value).strip().lower()
        if s in ["positive", "disease", "diseased", "abnormal", "hyper", "hypo", "yes"]:
            return "Positive"
        return "Negative"

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

    if confidence < 70 and risk == "Low":
        risk = "Medium"

    return rec, risk

def confidence_explanation(tsh, fti, ratio, confidence):
    reasons = []

    if confidence < 70:
        if TSH_NORMAL[0] <= tsh <= TSH_NORMAL[1]:
            reasons.append("TSH is close to the normal range, so the pattern is less clear.")
        if FTI_NORMAL[0] <= fti <= FTI_NORMAL[1]:
            reasons.append("FTI is close to the normal range, so the model has weaker evidence.")
        if RATIO_NORMAL[0] <= ratio <= RATIO_NORMAL[1]:
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

def make_live_chart(tsh, fti, verdict_text=None):
    point_color = "#94a3b8"
    if verdict_text == "Positive":
        point_color = "#ef4444"
    elif verdict_text == "Negative":
        point_color = "#22c55e"

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=FTI_NORMAL[0], x1=FTI_NORMAL[1],
        y0=TSH_NORMAL[0], y1=TSH_NORMAL[1],
        fillcolor="rgba(34,197,94,0.16)",
        line=dict(color="rgba(34,197,94,0.55)", width=2),
        layer="below",
    )

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
        height=460,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Real-time TSH vs FTI Chart",
        xaxis_title="FTI",
        yaxis_title="TSH",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def build_metric_chart(metrics_df):
    plot_df = metrics_df.reset_index().rename(columns={"index": "Model"})
    long_df = plot_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score"
    )
    fig = px.bar(
        long_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".3f",
        title="Multi-model Performance Comparison",
    )
    fig.update_layout(template="plotly_dark", height=480, margin=dict(l=10, r=10, t=50, b=10))
    return fig

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

def build_pie_chart(df, target_col):
    counts = df[target_col].value_counts(dropna=False)
    labels = [str(x) for x in counts.index]
    values = counts.values.tolist()
    fig = px.pie(names=labels, values=values, hole=0.42, title="Disease Distribution")
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
    age_col = None
    for c in df.columns:
        if c.lower() == "age":
            age_col = c
            break

    if "TSH" not in df.columns and "FTI" not in df.columns:
        return None

    plot_df = df.copy()
    if age_col and pd.api.types.is_numeric_dtype(plot_df[age_col]):
        plot_df = plot_df.sort_values(age_col)
        x_col = age_col
    else:
        plot_df = plot_df.reset_index(drop=True)
        plot_df["sample_idx"] = plot_df.index
        x_col = "sample_idx"

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

def make_pdf_report(name, age, sex, tsh, fti, ratio, verdict, confidence, recommendation, risk, model_name, model_acc):
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
    pdf.cell(0, 7, f"Selected Model: {model_name}", ln=True)
    pdf.cell(0, 7, f"Model Test Accuracy: {model_acc*100:.2f}%", ln=True)
    pdf.cell(0, 7, f"Prediction: {verdict}", ln=True)
    pdf.cell(0, 7, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 7, f"Risk Level: {risk}", ln=True)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Clinical Recommendation", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, recommendation)

    return pdf.output(dest="S").encode("latin-1")

def get_shap_explainer(model_name, model):
    try:
        if model_name in ["XGBoost", "Random Forest", "Decision Tree", "Best Model"]:
            return shap.TreeExplainer(model)
        return None
    except Exception:
        return None

def get_model_prob_and_confidence(model, input_df, pred_value):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_) if hasattr(model, "classes_") else None
        confidence = choose_confidence(pred_value, proba, classes)
        return proba, classes, confidence
    return None, None, 100.0

# =========================================================
# LOGIN
# =========================================================
try:
    with open("users.json") as f:
        users = json.load(f)
except Exception as e:
    st.error(f"users.json file not found. Error: {e}")
    st.stop()

if "login" not in st.session_state:
    st.session_state.login = False

def login_ui():
    st.markdown(
        """
        <div class="top-banner">
            <h1 style="margin:0;">🧠 Explainable Thyroid AI</h1>
            <p style="margin:0.35rem 0 0 0;" class="small-muted">
                Secure portal for thesis demonstration and clinical decision support
            </p>
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
# LOAD DATASET
# =========================================================
df_raw = load_dataset()
target_col = detect_target_column(df_raw)
if target_col is None:
    st.error("No target column found in the dataset. Use binaryClass / Class / target-like column.")
    st.stop()

y_all = normalize_target(df_raw[target_col])

raw_features_df = df_raw.drop(columns=[target_col]).copy()
raw_features_df = normalize_for_model(raw_features_df)

if Path("feature_columns.pkl").exists():
    try:
        feature_columns = joblib.load("feature_columns.pkl")
    except Exception:
        feature_columns = pd.get_dummies(raw_features_df, drop_first=False).columns.tolist()
else:
    feature_columns = pd.get_dummies(raw_features_df, drop_first=False).columns.tolist()

X_all = pd.get_dummies(raw_features_df, drop_first=False)
X_all = align_features(X_all, feature_columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all if y_all.nunique() == 2 else None,
)

# =========================================================
# LOAD MODELS
# =========================================================
available_model_files = get_available_model_files()
models = {}

for name, path in available_model_files.items():
    try:
        models[name] = load_model_file(path)
    except Exception as e:
        st.warning(f"Could not load {name}: {e}")

if not models:
    st.error("No model could be loaded. Upload at least thyroid_model.pkl.")
    st.stop()

# =========================================================
# EVALUATE MODELS
# =========================================================
def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    avg = "binary" if pd.Series(y_test).nunique() == 2 else "weighted"
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, average=avg, zero_division=0)
    rec = recall_score(y_test, pred, average=avg, zero_division=0)
    f1v = f1_score(y_test, pred, average=avg, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    return {
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1": f1v,
        "CM": cm
    }

model_scores = {}
for name, model in models.items():
    try:
        model_scores[name] = evaluate_model(model, X_test, y_test)
    except Exception as e:
        st.warning(f"Could not evaluate {name}: {e}")

if not model_scores:
    st.error("No model could be evaluated. Please verify feature_columns.pkl and model files.")
    st.stop()

metrics_df = pd.DataFrame(
    {
        name: {
            "Accuracy": vals["Accuracy"],
            "Precision": vals["Precision"],
            "Recall": vals["Recall"],
            "F1": vals["F1"],
        }
        for name, vals in model_scores.items()
    }
).T

metrics_df = metrics_df.sort_values("Accuracy", ascending=False)
best_model_name = metrics_df.index[0]
best_model = models[best_model_name]
best_metrics = model_scores[best_model_name]

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 ThyroPredict AI")
    st.caption("Explainable clinical dashboard")

    model_choice = st.selectbox(
        "Active Model",
        list(metrics_df.index),
        index=list(metrics_df.index).index(best_model_name)
    )
    selected_model = models[model_choice]
    selected_metrics = model_scores[model_choice]

    st.markdown("### 🏆 Best Model")
    st.success(best_model_name)

    st.markdown("### 📈 Test Accuracy")
    for name in metrics_df.index:
        st.metric(name, f"{metrics_df.loc[name, 'Accuracy']*100:.2f}%")

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
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Prediction", "📊 Model Comparison", "🌍 Analytics", "📂 Batch Prediction"])

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

    with c2:
        tsh = st.number_input("TSH", value=6.0, min_value=0.0, step=0.1)
        fti = st.number_input("FTI", value=50.0, min_value=0.0, step=0.1)

    with c3:
        ratio = tsh / (fti + 0.001)
        st.metric("TSH/FTI Ratio", f"{ratio:.5f}")
        st.metric("Selected Model", model_choice)
        st.metric("Best Model", best_model_name)

    st.markdown("### 📊 Live TSH vs FTI Chart")
    live_fig = make_live_chart(tsh, fti, None)
    st.plotly_chart(live_fig, use_container_width=True, key="live_chart")

    if st.button("🚀 Run Diagnosis", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            input_raw = build_patient_raw_input(raw_features_df, age, sex, tsh, fti)
            input_df = prepare_input_features(input_raw, feature_columns)

            pred_value = selected_model.predict(input_df)[0]
            pred_text = label_to_text(pred_value)

            proba, classes, confidence = get_model_prob_and_confidence(
                selected_model, input_df, pred_value
            )

            recommendation, risk = doctor_recommendation(tsh, fti, ratio, pred_text, confidence)
            reasons = confidence_explanation(tsh, fti, ratio, confidence)

        st.markdown("---")

        res_col1, res_col2 = st.columns([1.2, 1.0])

        with res_col1:
            if pred_text == "Positive":
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

            st.write("• TSH/FTI ratio helps the model capture hormone relationship and imbalance.")
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
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
        for r in reasons:
            st.write("•", r)

        st.markdown("### 🧠 Explainable AI (SHAP)")
        selected_explainer = get_shap_explainer(model_choice, selected_model)
        if selected_explainer is not None:
            try:
                shap_values = selected_explainer(input_df)
                fig, ax = plt.subplots(figsize=(11, 5))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig, use_container_width=True, clear_figure=True)
            except Exception as e:
                st.info(f"SHAP waterfall could not be rendered in this run. ({e})")
        else:
            st.info("SHAP is shown for tree-based models like XGBoost / Random Forest / Decision Tree.")

        st.markdown("""
### 🔍 Explainable AI Interpretation

The Explainable AI framework identified TSH as the most influential contributor toward the final prediction, while FTI and the engineered TSH/FTI ratio provided complementary diagnostic significance.

The visualization illustrates how each biomarker contributed either positively or negatively toward the AI-generated diagnostic outcome.
""")

        st.markdown("### 📈 Patient Live Chart")
        result_chart = make_live_chart(tsh, fti, pred_text)
        st.plotly_chart(result_chart, use_container_width=True, key="result_chart")

        st.markdown("### 📄 Export Report")
        pdf_bytes = make_pdf_report(
            patient_name,
            age,
            sex,
            tsh,
            fti,
            ratio,
            pred_text,
            confidence,
            recommendation,
            risk,
            model_choice,
            metrics_df.loc[model_choice, "Accuracy"],
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
    st.plotly_chart(comparison_fig, use_container_width=True, key="comparison_chart")

    best_row = metrics_df.loc[best_model_name]
    st.success(
        f"Best model automatically selected: **{best_model_name}** "
        f"with accuracy **{best_row['Accuracy']*100:.2f}%**"
    )

    st.markdown("### 📋 Comparison Table")
    show_df = (metrics_df * 100).round(2)
    st.dataframe(show_df, use_container_width=True)

    st.markdown("### 📉 Confusion Matrix of Selected Model")
    cm_fig = build_confusion_figure(selected_metrics["CM"], f"{model_choice} Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True, key="selected_cm")

    st.markdown("### 🏆 Best Model Confusion Matrix")
    best_cm_fig = build_confusion_figure(best_metrics["CM"], f"{best_model_name} Confusion Matrix")
    st.plotly_chart(best_cm_fig, use_container_width=True, key="best_cm")

    st.markdown("### 📈 ROC Curve")
    if hasattr(selected_model, "predict_proba"):
        try:
            selected_probs = selected_model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, selected_probs)
            roc_auc = auc(fpr, tpr)

            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.4f}"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
            roc_fig.update_layout(
                template="plotly_dark",
                height=450,
                title=f"ROC Curve - {model_choice}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(roc_fig, use_container_width=True, key="roc_curve")
        except Exception as e:
            st.info(f"ROC curve could not be rendered. ({e})")
    else:
        st.info("ROC curve not available because the selected model does not provide probabilities.")

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
        st.metric("Features used", f"{len(feature_columns):,}")
    with o4:
        missing_total = int(df_raw.isna().sum().sum())
        st.metric("Missing values", f"{missing_total:,}")

    st.markdown("### 🥧 Disease Distribution")
    pie_fig = build_pie_chart(df_raw, target_col)
    st.plotly_chart(pie_fig, use_container_width=True, key="pie_chart")

    st.markdown("### 🌡️ Correlation Heatmap")
    corr_fig = build_correlation_heatmap(df_raw, exclude_col=target_col)
    if corr_fig is not None:
        st.plotly_chart(corr_fig, use_container_width=True, key="corr_heatmap")
    else:
        st.info("Correlation heatmap could not be generated because numeric columns are insufficient.")

    st.markdown("### 📈 Biomarker Trend")
    trend_fig = build_biomarker_trend(df_raw)
    if trend_fig is not None:
        st.plotly_chart(trend_fig, use_container_width=True, key="trend_chart")
    else:
        st.info("Biomarker trend graph could not be generated.")

    st.markdown("### 🌍 Global Explainability")
    global_model_name = best_model_name
    global_model = best_model
    global_explainer = get_shap_explainer(global_model_name, global_model)

    if global_explainer is not None:
        try:
            sample_n = min(120, len(X_train))
            background = X_train.sample(sample_n, random_state=42) if len(X_train) > sample_n else X_train.copy()
            shap_values = global_explainer(background)

            try:
                vals = np.abs(shap_values.values).mean(axis=0)
                imp_df = pd.DataFrame({"Feature": background.columns, "Importance": vals})
                imp_df = imp_df.sort_values("Importance", ascending=False).head(15)

                imp_fig = px.bar(
                    imp_df.sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top SHAP Feature Importance"
                )
                imp_fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(imp_fig, use_container_width=True, key="global_importance")
            except Exception:
                st.info("Could not render global feature importance chart.")
        except Exception as e:
            st.info(f"Global explainability unavailable. ({e})")
    else:
        st.info("Global explainability is available mainly for tree-based models.")

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
                batch_raw = batch_df.copy()
                batch_raw = normalize_for_model(batch_raw)
                batch_raw = pd.get_dummies(batch_raw, drop_first=False)
                batch_X = align_features(batch_raw, feature_columns)

                preds = selected_model.predict(batch_X)
                pred_texts = [label_to_text(p) for p in preds]

                batch_result = batch_df.copy()
                batch_result["AI_Result"] = pred_texts

                if hasattr(selected_model, "predict_proba"):
                    probs = selected_model.predict_proba(batch_X)
                    classes = list(selected_model.classes_) if hasattr(selected_model, "classes_") else None
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
