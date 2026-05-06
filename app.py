# =============================================================================
# model_evaluation.py
# ThyroPredict — Dynamic Model Evaluation Module
#
# Calculates REAL accuracy from your trained models + dataset.
# Drop this file alongside app.py and call show_model_evaluation(tab) from
# your Tab 4 (Model Comparison) section.
#
# Exact setup matching your Google Colab training:
#   • Dataset : cleaned_dataset_Thyroid1.csv  (3771 rows, 25 features)
#   • Target  : binaryClass  (0 = Healthy, 1 = Diseased)
#   • Split   : 80 / 20  |  random_state=42  |  stratify=y
#   • RF      : RandomForestClassifier(n_estimators=100, random_state=42)
#   • XGB     : XGBClassifier(n_estimators=100, random_state=42)
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ── Try importing XGBoost; fall back to GradientBoosting if unavailable ──────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False

# =============================================================================
# CONSTANTS — must match your Colab training exactly
# =============================================================================
DATASET_PATH  = "cleaned_dataset_Thyroid1.csv"   # put CSV next to app.py
XGB_PKL_PATH  = "thyroid_model.pkl"               # your pkl from GitHub
RF_PKL_PATH   = "rf_model.pkl"                    # optional separate RF pkl
TARGET_COL    = "binaryClass"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

# All 25 features (same order as Colab)
FEATURE_COLS = [
    'age', 'sex', 'on thyroxine', 'query on thyroxine',
    'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
    'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
    'T3 measured', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
    'FTI measured', 'FTI'
]


# =============================================================================
# DATA & MODEL LOADING  (cached — runs only once per session)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_and_split_data():
    """
    Load the CSV and reproduce the exact 80/20 stratified split from Colab.
    Returns X_train, X_test, y_train, y_test as numpy arrays.
    """
    df = pd.read_csv(DATASET_PATH)

    # Validate columns
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns in CSV: {missing}")
        st.stop()

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # same as Colab — preserves class ratio
    )
    return X_train, X_test, y_train, y_test, df


@st.cache_resource(show_spinner=False)
def load_or_train_models(X_train, y_train):
    """
    1. Try loading your saved pkl files (thyroid_model.pkl / rf_model.pkl).
    2. If pkl not found, retrain from scratch with the same hyperparameters.

    Returns (xgb_model, rf_model, source) where source is 'pkl' or 'retrained'.
    """
    xg_model = None
    rf_model  = None
    source    = "retrained"

    # ── Attempt to load XGBoost pkl ──────────────────────────────────────────
    if os.path.exists(XGB_PKL_PATH):
        try:
            with open(XGB_PKL_PATH, "rb") as f:
                loaded = pickle.load(f)
            # pkl might be a dict {"xgb": ..., "rf": ...} or a bare model
            if isinstance(loaded, dict):
                xg_model = loaded.get("xgb") or loaded.get("xgboost") or loaded.get("model")
                rf_model  = loaded.get("rf")  or loaded.get("random_forest")
            else:
                xg_model = loaded
            source = "pkl"
        except Exception as e:
            st.warning(f"⚠️ Could not load {XGB_PKL_PATH}: {e}. Retraining models.")

    # ── Attempt to load separate RF pkl ──────────────────────────────────────
    if rf_model is None and os.path.exists(RF_PKL_PATH):
        try:
            with open(RF_PKL_PATH, "rb") as f:
                rf_model = pickle.load(f)
            source = "pkl"
        except Exception:
            pass

    # ── Train whichever model is still missing ────────────────────────────────
    if xg_model is None:
        if XGB_AVAILABLE:
            xg_model = xgb.XGBClassifier(
                n_estimators=100, random_state=RANDOM_STATE,
                use_label_encoder=False, eval_metric="logloss"
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            xg_model = GradientBoostingClassifier(
                n_estimators=100, random_state=RANDOM_STATE
            )
        xg_model.fit(X_train, y_train)

    if rf_model is None:
        rf_model = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        )
        rf_model.fit(X_train, y_train)

    return xg_model, rf_model, source


@st.cache_data(show_spinner=False)
def compute_metrics(_xg_model, _rf_model, X_test, y_test):
    """
    Compute all evaluation metrics on the held-out test set.
    Results are cached so they never recompute on user input changes.
    """
    results = {}
    for name, model in [("XGBoost", _xg_model), ("Random Forest", _rf_model)]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy" : accuracy_score(y_test, y_pred) * 100,
            "f1"       : f1_score(y_test, y_pred, zero_division=0) * 100,
            "auc"      : roc_auc_score(y_test, y_prob) * 100,
            "cm"       : confusion_matrix(y_test, y_pred),
            "report"   : classification_report(
                             y_test, y_pred,
                             target_names=["Healthy (0)", "Diseased (1)"],
                             output_dict=True
                         ),
            "y_pred"   : y_pred,
            "y_prob"   : y_prob,
        }
    return results


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def _plot_confusion_matrices(results):
    """Side-by-side confusion matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#F4F7FB")
    fig.patch.set_facecolor("#F4F7FB")

    NAVY  = "#0F2041"
    BLUE  = "#1565C0"
    WHITE = "white"
    cmap  = plt.cm.Blues

    for ax, (model_name, res) in zip(axes, results.items()):
        cm = res["cm"]
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_facecolor(WHITE)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center", fontsize=16, fontweight="bold",
                        color="white" if cm[i, j] > thresh else NAVY)

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Healthy (0)", "Diseased (1)"], fontsize=9, color=NAVY)
        ax.set_yticklabels(["Healthy (0)", "Diseased (1)"], fontsize=9, color=NAVY,
                           rotation=90, va="center")
        ax.set_xlabel("Predicted Label", fontsize=9, color="#555", labelpad=8)
        ax.set_ylabel("True Label",      fontsize=9, color="#555", labelpad=8)
        ax.set_title(f"{model_name}\nAccuracy: {res['accuracy']:.2f}%",
                     fontsize=11, fontweight="bold", color=NAVY, pad=10)
        for spine in ax.spines.values():
            spine.set_color("#E8EDF5")

    plt.suptitle("Confusion Matrices — Test Set (80/20 Split)",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    return fig


def _plot_model_comparison(results):
    """Grouped bar chart comparing Accuracy, F1, AUC-ROC for both models."""
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#F4F7FB")
    ax.set_facecolor("white")

    metrics      = ["Accuracy (%)", "F1 Score (%)", "AUC-ROC (%)"]
    metric_keys  = ["accuracy", "f1", "auc"]
    model_names  = list(results.keys())
    colors_bar   = ["#1565C0", "#2E7D32"]

    x     = np.arange(len(metrics))
    width = 0.32

    for i, (mname, color) in enumerate(zip(model_names, colors_bar)):
        vals = [results[mname][k] for k in metric_keys]
        bars = ax.bar(x + i * width - width / 2, vals,
                      width, label=mname, color=color, alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{val:.2f}%",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#1A1A2E")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, color="#334")
    ax.set_ylim(94, 101)
    ax.set_ylabel("Score (%)", fontsize=10, color="#555")
    ax.set_title("Model Performance Comparison — XGBoost vs Random Forest",
                 fontsize=11, fontweight="bold", color="#0F2041", pad=10)
    ax.legend(fontsize=9, framealpha=0.6, edgecolor="#E8EDF5")
    ax.yaxis.grid(True, color="#E8EDF5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#E8EDF5")
    ax.tick_params(colors="#555")

    plt.tight_layout()
    return fig


def _plot_feature_importance(rf_model):
    """Top-15 feature importances from Random Forest."""
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]
    top_names   = [FEATURE_COLS[i] for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#F4F7FB")
    ax.set_facecolor("white")
    colors_fi = ["#1565C0" if i == 0 else "#5C8FD6" for i in range(len(top_names))]
    ax.barh(top_names[::-1], top_vals[::-1], color=colors_fi[::-1], alpha=0.88)
    ax.set_xlabel("Importance Score", fontsize=9, color="#555")
    ax.set_title("Top-15 Feature Importances (Random Forest)",
                 fontsize=11, fontweight="bold", color="#0F2041", pad=10)
    ax.tick_params(labelsize=8.5, colors="#555")
    for spine in ax.spines.values():
        spine.set_color("#E8EDF5")
    ax.xaxis.grid(True, color="#E8EDF5", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN PUBLIC FUNCTION — call this from your Tab 4
# =============================================================================

def show_model_evaluation():
    """
    Renders the complete Model Evaluation section inside Streamlit.
    Call this function from your Tab 4 (Model Comparison) block:

        with tabs[3]:
            show_model_evaluation()
    """

    # ── Section header ────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(110deg,#0F2041,#1565C0);
                border-radius:12px;padding:1.4rem 2rem;margin-bottom:1.5rem;color:white;'>
        <div style='font-size:1.3rem;font-weight:700;margin-bottom:0.2rem;'>
            ⚖️ Model Performance Evaluation
        </div>
        <div style='font-size:0.85rem;opacity:0.8;'>
            Real accuracy computed from your dataset — no hardcoded values.
            Split: 80/20 · stratified · random_state=42 · cached for this session.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        st.error(f"❌ Dataset not found: `{DATASET_PATH}`\n\n"
                 f"Please place `{DATASET_PATH}` in the same folder as `app.py`.")
        return

    with st.spinner("📊 Loading dataset and computing metrics…"):
        X_train, X_test, y_train, y_test, df = load_and_split_data()
        xg_model, rf_model, source           = load_or_train_models(X_train, y_train)
        results                              = compute_metrics(
                                                 xg_model, rf_model, X_test, y_test
                                             )

    # ── Source badge ──────────────────────────────────────────────────────────
    badge_color = "#1B5E20" if source == "pkl" else "#E65100"
    badge_text  = "✅ Loaded from .pkl files" if source == "pkl" else "🔄 Retrained (pkl not found)"
    st.markdown(f"""
    <div style='display:inline-block;background:#F4F7FB;border:1px solid #E8EDF5;
                border-radius:20px;padding:0.3rem 1rem;font-size:0.8rem;
                color:{badge_color};font-weight:600;margin-bottom:1.2rem;'>
        {badge_text}
    </div>
    <span style='font-size:0.8rem;color:#8E9AB5;margin-left:0.6rem;'>
        Dataset: {len(df):,} rows · Train: {len(X_train):,} · Test: {len(X_test):,}
    </span>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──────────────────────────────────────────────────────────
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#0F2041;"
                "margin-bottom:0.8rem;'>📋 Performance Metrics</div>",
                unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    card_pairs = [
        (col1, col2, "XGBoost",       results["XGBoost"],       "#1565C0"),
        (col3, col4, "Random Forest", results["Random Forest"], "#2E7D32"),
    ]
    for ca, cb, cc, mname, res, color in [
        (col1, col2, col3, "XGBoost",       results["XGBoost"],       "#1565C0"),
        (col4, col5, col6, "Random Forest", results["Random Forest"], "#2E7D32"),
    ]:
        with ca:
            st.markdown(f"""
            <div style='background:white;border-radius:12px;padding:1rem;
                        box-shadow:0 2px 8px rgba(15,32,65,0.07);
                        border-top:4px solid {color};text-align:center;'>
                <div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                            text-transform:uppercase;color:#6B7A99;'>
                    {mname}<br>Accuracy
                </div>
                <div style='font-size:1.8rem;font-weight:800;color:{color};
                            line-height:1.2;margin-top:0.3rem;'>
                    {res['accuracy']:.2f}%
                </div>
            </div>""", unsafe_allow_html=True)
        with cb:
            st.markdown(f"""
            <div style='background:white;border-radius:12px;padding:1rem;
                        box-shadow:0 2px 8px rgba(15,32,65,0.07);
                        border-top:4px solid {color};text-align:center;'>
                <div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                            text-transform:uppercase;color:#6B7A99;'>
                    {mname}<br>F1 Score
                </div>
                <div style='font-size:1.8rem;font-weight:800;color:{color};
                            line-height:1.2;margin-top:0.3rem;'>
                    {res['f1']:.2f}%
                </div>
            </div>""", unsafe_allow_html=True)
        with cc:
            st.markdown(f"""
            <div style='background:white;border-radius:12px;padding:1rem;
                        box-shadow:0 2px 8px rgba(15,32,65,0.07);
                        border-top:4px solid {color};text-align:center;'>
                <div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                            text-transform:uppercase;color:#6B7A99;'>
                    {mname}<br>AUC-ROC
                </div>
                <div style='font-size:1.8rem;font-weight:800;color:{color};
                            line-height:1.2;margin-top:0.3rem;'>
                    {res['auc']:.2f}%
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#0F2041;"
                "margin-bottom:0.8rem;'>🔲 Confusion Matrices</div>",
                unsafe_allow_html=True)
    fig_cm = _plot_confusion_matrices(results)
    st.pyplot(fig_cm, use_container_width=True)
    plt.close(fig_cm)

    # ── CM interpretation callout ─────────────────────────────────────────────
    cm_xg = results["XGBoost"]["cm"]
    tn, fp, fn, tp = cm_xg.ravel()
    st.markdown(f"""
    <div style='background:#EEF4FF;border-left:4px solid #1565C0;
                border-radius:0 10px 10px 0;padding:0.9rem 1.2rem;
                font-size:0.87rem;color:#1B3A6B;margin-bottom:1.5rem;'>
        <b>Reading the matrix (XGBoost):</b>
        &nbsp;✅ True Healthy: <b>{tn}</b>
        &nbsp;|&nbsp; ✅ True Diseased: <b>{tp}</b>
        &nbsp;|&nbsp; ⚠️ False Positive: <b>{fp}</b>
        &nbsp;|&nbsp; ❌ False Negative: <b>{fn}</b>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Comparison Bar Chart ────────────────────────────────────────────
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#0F2041;"
                "margin-bottom:0.8rem;'>📊 Head-to-Head Comparison</div>",
                unsafe_allow_html=True)
    fig_cmp = _plot_model_comparison(results)
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("<div style='font-size:1rem;font-weight:700;color:#0F2041;"
                "margin-bottom:0.8rem;'>🏆 Feature Importance (Random Forest)</div>",
                unsafe_allow_html=True)
    fig_fi = _plot_feature_importance(rf_model)
    st.pyplot(fig_fi, use_container_width=True)
    plt.close(fig_fi)

    # ── Detailed Classification Report ────────────────────────────────────────
    with st.expander("📄 Full Classification Report"):
        for mname, res in results.items():
            st.markdown(f"**{mname}**")
            rpt = res["report"]
            rows = []
            for label in ["Healthy (0)", "Diseased (1)", "macro avg", "weighted avg"]:
                if label in rpt:
                    r = rpt[label]
                    rows.append({
                        "Class"    : label,
                        "Precision": f"{r['precision']*100:.2f}%",
                        "Recall"   : f"{r['recall']*100:.2f}%",
                        "F1-Score" : f"{r['f1-score']*100:.2f}%",
                        "Support"  : int(r.get("support", 0)),
                    })
            st.dataframe(pd.DataFrame(rows).set_index("Class"),
                         use_container_width=True)
            st.markdown("---")

    # ── Training info footer ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:#F8F9FB;border:1px solid #E8EDF5;border-radius:10px;
                padding:0.9rem 1.2rem;font-size:0.82rem;color:#6B7A99;
                margin-top:1rem;'>
        <b>Training Details:</b>
        &nbsp; Dataset: cleaned_dataset_Thyroid1.csv ({len(df):,} samples)
        &nbsp;·&nbsp; Features: {len(FEATURE_COLS)} clinical biomarkers
        &nbsp;·&nbsp; Split: 80 / 20 (stratified, random_state=42)
        &nbsp;·&nbsp; XGBoost: n_estimators=100
        &nbsp;·&nbsp; Random Forest: n_estimators=100
        &nbsp;·&nbsp; Metrics cached with <code>st.cache_data</code>
    </div>
    """, unsafe_allow_html=True)
