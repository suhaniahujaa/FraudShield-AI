
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="FraudShield AI | Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_model.pkl"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
DATA_PATH = ROOT / "data" / "creditcard.csv"

FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

# -------------------- THEME --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --cyan:#52e7ff;
    --blue:#628cff;
    --mint:#50efc0;
    --violet:#b28aff;
    --pink:#ff76b7;
    --bg:#070d18;
    --muted:#91a6bd;
}

html, body, [class*="css"] {
    font-family:'Inter',sans-serif;
}

.stApp {
    background:
      radial-gradient(ellipse at 75% 8%,rgba(28,112,165,.19),transparent 32%),
      radial-gradient(ellipse at 8% 65%,rgba(26,125,125,.10),transparent 30%),
      radial-gradient(ellipse at 90% 90%,rgba(104,65,170,.10),transparent 30%),
      #070d18;
    color:#edf6ff;
}

[data-testid="stHeader"] {
    background:rgba(5,10,18,.65);
}

.block-container {
    max-width:1500px;
    padding:2rem 3rem 3rem;
}

section[data-testid="stSidebar"] {
    background:
      linear-gradient(180deg,rgba(17,31,49,.97),rgba(7,15,28,.99));
    border-right:1px solid rgba(82,231,255,.18);
}

section[data-testid="stSidebar"] > div {
    padding:2.2rem 1.1rem;
}

section[data-testid="stSidebar"] [role="radiogroup"] {
    gap:8px;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
    padding:12px 13px;
    border-radius:12px;
    border:1px solid transparent;
    color:#b8c8d9;
    transition:all .2s ease;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background:rgba(82,231,255,.08);
    border-color:rgba(82,231,255,.2);
}

section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(100deg,rgba(82,231,255,.17),rgba(178,138,255,.12));
    border:1px solid rgba(82,231,255,.38);
    color:white;
    box-shadow:0 0 22px rgba(82,231,255,.08);
}

.brand {
    font-family:'Space Grotesk',sans-serif;
    font-size:29px;
    font-weight:800;
    letter-spacing:-1.8px;
    color:#f4f9ff;
}

.brand span {
    color:var(--cyan);
    text-shadow:0 0 18px rgba(82,231,255,.55);
}

.brand-sub {
    font-size:9px;
    letter-spacing:4px;
    color:#829ab4;
    margin-top:6px;
}

.side-label {
    color:var(--cyan);
    font-size:10px;
    font-weight:800;
    letter-spacing:3px;
    margin:24px 0 10px;
}

.side-status {
    padding:15px;
    border-radius:14px;
    border:1px solid rgba(80,239,192,.24);
    background:rgba(80,239,192,.07);
    color:#50efc0;
    font-size:10px;
    letter-spacing:1.5px;
    font-weight:800;
}

.side-muted {
    color:#71879d;
    font-size:11px;
    line-height:1.8;
}

.hero {
    position:relative;
    overflow:hidden;
    min-height:330px;
    padding:45px 48px;
    border-radius:27px;
    border:1px solid rgba(82,231,255,.32);
    background:
      radial-gradient(circle at 85% 50%,rgba(82,231,255,.19),transparent 27%),
      radial-gradient(circle at 75% 90%,rgba(178,138,255,.13),transparent 30%),
      linear-gradient(120deg,rgba(20,55,72,.94),rgba(17,27,50,.97) 65%,rgba(25,24,52,.96));
    box-shadow:
      0 25px 80px rgba(0,0,0,.25),
      inset 0 1px 1px rgba(255,255,255,.12),
      0 0 35px rgba(82,231,255,.06);
}

.hero:after {
    content:"";
    position:absolute;
    width:340px;
    height:340px;
    border-radius:50%;
    border:1px solid rgba(82,231,255,.22);
    right:4%;
    top:-135px;
    box-shadow:
      0 0 0 20px rgba(82,231,255,.025),
      0 0 0 48px rgba(82,231,255,.018),
      0 0 60px rgba(82,231,255,.08);
}

.hero-content {
    position:relative;
    z-index:2;
    max-width:850px;
}

.eyebrow {
    color:var(--cyan);
    font-family:'Space Grotesk',sans-serif;
    font-size:10px;
    font-weight:700;
    letter-spacing:3.5px;
}

.hero h1 {
    font-family:'Space Grotesk',sans-serif;
    color:#f0f7ff;
    font-size:clamp(36px,4.3vw,58px);
    line-height:1.12;
    letter-spacing:-2.5px;
    margin:23px 0 17px;
    font-weight:700;
}

.hero p {
    color:#b5c8d9;
    font-size:14px;
    line-height:1.9;
    max-width:730px;
}

.shield {
    position:absolute;
    z-index:1;
    right:9%;
    top:50%;
    transform:translateY(-50%);
    width:180px;
    height:210px;
    display:flex;
    justify-content:center;
    align-items:center;
    font-size:120px;
    color:#a8f8ff;
    filter:
      drop-shadow(0 0 10px #43dff7)
      drop-shadow(0 0 35px rgba(82,231,255,.8));
}

.pill {
    display:inline-flex;
    align-items:center;
    gap:9px;
    padding:10px 16px;
    border-radius:30px;
    border:1px solid rgba(80,239,192,.38);
    background:rgba(80,239,192,.09);
    color:#65f4c4;
    font-size:10px;
    font-weight:800;
    letter-spacing:1.3px;
}

.dot {
    width:7px;
    height:7px;
    background:#50efc0;
    border-radius:50%;
    box-shadow:0 0 12px #50efc0;
}

.glass {
    background:linear-gradient(145deg,rgba(25,43,62,.78),rgba(12,24,41,.86));
    border:1px solid rgba(151,200,228,.17);
    border-radius:19px;
    padding:23px;
    box-shadow:
      inset 0 1px 1px rgba(255,255,255,.055),
      0 16px 40px rgba(0,0,0,.15);
    backdrop-filter:blur(16px);
    height:100%;
}

.glass h3 {
    font-family:'Space Grotesk',sans-serif;
    font-size:18px;
    color:#f0f7ff;
    margin:12px 0 7px;
}

.glass p {
    font-size:13px;
    line-height:1.8;
    color:#94aac0;
}

.kicker {
    color:var(--cyan);
    font-size:10px;
    font-weight:800;
    letter-spacing:2.5px;
}

.section {
    margin:35px 0 18px;
}

.section h2 {
    font-family:'Space Grotesk',sans-serif;
    color:#f1f7ff;
    font-size:25px;
    margin:7px 0 5px;
}

.section p {
    color:#8da3b9;
    font-size:13px;
}

[data-testid="stMetric"] {
    background:linear-gradient(145deg,rgba(24,45,65,.9),rgba(11,23,39,.95));
    border:1px solid rgba(82,231,255,.19);
    border-radius:17px;
    padding:18px 20px;
    box-shadow:inset 0 1px rgba(255,255,255,.05);
}

[data-testid="stMetricLabel"] {
    color:#a2b7ca !important;
    font-size:10px !important;
    letter-spacing:1.5px;
}

[data-testid="stMetricValue"] {
    color:#f3fbff !important;
    font-family:'Space Grotesk',sans-serif;
    font-weight:700;
}

.stButton > button, .stDownloadButton > button {
    background:linear-gradient(105deg,#52e7ff,#50efc0);
    color:#071421;
    border:0;
    border-radius:11px;
    font-weight:800;
    min-height:45px;
    box-shadow:0 5px 24px rgba(82,231,255,.16);
}

.stButton > button:hover, .stDownloadButton > button:hover {
    color:#071421;
    border:1px solid #c2ffff;
    box-shadow:0 0 28px rgba(82,231,255,.35);
}

.stTextInput input, .stNumberInput input {
    background:#0e1c2d !important;
    color:white !important;
    border:1px solid rgba(82,231,255,.25) !important;
    border-radius:10px !important;
}

[data-testid="stFileUploader"] {
    border:1px dashed rgba(82,231,255,.4);
    border-radius:15px;
    background:rgba(15,30,48,.55);
    padding:12px;
}

[data-testid="stDataFrame"] {
    border:1px solid rgba(82,231,255,.2);
    border-radius:14px;
}

.result-fraud {
    padding:24px;
    border-radius:17px;
    border:1px solid rgba(255,104,137,.45);
    background:linear-gradient(110deg,rgba(150,40,75,.2),rgba(30,20,43,.8));
}

.result-safe {
    padding:24px;
    border-radius:17px;
    border:1px solid rgba(80,239,192,.4);
    background:linear-gradient(110deg,rgba(35,145,110,.17),rgba(15,35,43,.8));
}

.pipeline {
    padding:13px 0;
    border-bottom:1px solid rgba(150,190,215,.13);
    color:#c3d3e1;
    font-size:13px;
}

.footer {
    margin-top:45px;
    padding:22px 0;
    border-top:1px solid rgba(150,190,215,.13);
    color:#7389a0;
    font-size:10px;
    letter-spacing:2px;
    text-align:center;
}

hr {
    border-color:rgba(150,190,215,.14);
}

#MainMenu, footer {
    visibility:hidden;
}

@media(max-width:900px) {
    .block-container {padding:1rem;}
    .hero {padding:28px;}
    .shield {opacity:.18;right:0;}
}
</style>
""", unsafe_allow_html=True)


# -------------------- MODEL --------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    saved = joblib.load(MODEL_PATH)

    if isinstance(saved, dict) and "model" in saved:
        return {
            "model": saved["model"],
            "features": saved.get("features", FEATURES),
            "threshold": float(saved.get("threshold", 0.5)),
            "name": saved.get("model_name", "Random Forest")
        }

    return {
        "model": saved,
        "features": FEATURES,
        "threshold": 0.5,
        "name": "Saved ML Model"
    }


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


try:
    artifact = load_model()
    model_error = None
except Exception as error:
    artifact = None
    model_error = str(error)

metrics_data = load_metrics()
dataset = load_dataset()
model_ready = artifact is not None


def metric_value(data, *keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        if data.get(key) is not None:
            return data[key]
    return None


def pct(value):
    try:
        return f"{float(value)*100:.2f}%"
    except (ValueError, TypeError):
        return "—"


def predict_data(df):
    missing = [f for f in artifact["features"] if f not in df.columns]
    if missing:
        raise ValueError("Missing features: " + ", ".join(missing))

    X = df[artifact["features"]].copy()

    if X.isnull().any().any():
        raise ValueError("Your CSV contains empty feature values.")

    probabilities = artifact["model"].predict_proba(X)[:, 1]
    predictions = (probabilities >= artifact["threshold"]).astype(int)

    result = df.copy()
    result["Fraud Probability"] = probabilities
    result["Prediction"] = np.where(predictions == 1, "FRAUD", "GENUINE")
    result["Risk Level"] = np.where(
        probabilities >= 0.5, "HIGH",
        np.where(probabilities >= 0.2, "MEDIUM", "LOW")
    )
    return result


def section(kicker, title, subtitle):
    st.markdown(f"""
    <div class="section">
      <div class="eyebrow">{kicker}</div>
      <h2>{title}</h2>
      <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def chart_style(fig, height=330):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#b7c9dc"),
        margin=dict(l=15, r=15, t=55, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#398cc7")
        )
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(150,190,215,.10)", zeroline=False)
    return fig


# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.markdown("""
    <div class="brand">FRAUD<span>SHIELD</span></div>
    <div class="brand-sub">AI FINANCIAL INTELLIGENCE</div>
    """, unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    st.markdown('<div class="side-label">MAIN WORKSPACE</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "◈  Overview",
            "⌕  Fraud Scanner",
            "▤  Batch Intelligence",
            "◉  Transaction Explorer",
            "✧  Model Intelligence",
            "▧  Dataset Insights",
            "ⓘ  About Platform"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown('<div class="side-label">SYSTEM STATUS</div>', unsafe_allow_html=True)

    if model_ready:
        st.markdown(
            '<div class="side-status">● &nbsp; MODEL CONNECTED</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p class="side-muted">{artifact["name"]}<br>TRAINED MODEL · READY FOR INFERENCE</p>',
            unsafe_allow_html=True
        )
    else:
        st.error("Model unavailable")
        st.caption(model_error or "Check models/fraud_model.pkl")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="side-muted">FRAUDSHIELD AI<br>VERSION 4.0<br><br>INTELLIGENCE. SECURITY. TRUST.</div>',
        unsafe_allow_html=True
    )


# -------------------- OVERVIEW --------------------

if page == "◈  Overview":

    st.markdown("""
    <div class="hero">
      <div class="hero-content">
        <div class="eyebrow">✦ &nbsp; AI-POWERED FINANCIAL SECURITY</div>
        <h1>See threats before<br>they happen.</h1>
        <p>
        An intelligent transaction monitoring environment powered by
        machine learning. Detect suspicious patterns, estimate fraud risk
        and explore financial transaction intelligence.
        </p>
        <div class="pill"><span class="dot"></span> SYSTEM OPERATIONAL</div>
      </div>
      <div class="shield">⛨</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    saved = metrics_data.get("metrics", metrics_data.get("best_metrics", {}))

    total = len(dataset) if dataset is not None else 284807
    fraud = int(dataset["Class"].sum()) if dataset is not None and "Class" in dataset else 492
    genuine = total - fraud

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL TRANSACTIONS", f"{total:,}")
    c2.metric("GENUINE PAYMENTS", f"{genuine:,}")
    c3.metric("FRAUD CASES", f"{fraud:,}")
    c4.metric("ACTIVE MODEL", artifact["name"] if model_ready else "Offline")

    section("INTELLIGENCE CENTER", "Your security overview",
            "A real-time-style overview of the trained fraud detection environment.")

    left, right = st.columns([1.25, .75], gap="large")

    with left:
        st.markdown("""
        <div class="glass">
          <div class="kicker">01 / INTELLIGENCE ENGINE</div>
          <h3>Transaction analysis pipeline</h3>
          <p>From raw financial data to machine-learning risk estimation.</p>
          <div class="pipeline">◈ &nbsp; 01 &nbsp; Transaction data ingestion</div>
          <div class="pipeline">◈ &nbsp; 02 &nbsp; Feature validation & preprocessing</div>
          <div class="pipeline">◈ &nbsp; 03 &nbsp; Trained model inference</div>
          <div class="pipeline">◈ &nbsp; 04 &nbsp; Fraud probability estimation</div>
          <div class="pipeline">◈ &nbsp; 05 &nbsp; Risk classification & reporting</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        model_name = artifact["name"] if model_ready else "Model unavailable"
        threshold = artifact["threshold"] if model_ready else 0.5

        st.markdown(f"""
        <div class="glass">
          <div class="kicker">02 / ACTIVE MODEL</div>
          <h3>{model_name}</h3>
          <p>Your saved machine-learning model is ready to analyse transaction patterns.</p>
          <br>
          <div class="pill"><span class="dot"></span> TRAINED & SAVED</div>
          <br><br>
          <p>Decision threshold</p>
          <h3 style="color:#52e7ff;font-size:30px">{threshold:.3f}</h3>
          <p>30 transaction attributes</p>
        </div>
        """, unsafe_allow_html=True)

    section("PERFORMANCE SNAPSHOT", "Model performance",
            "Key evaluation scores from your saved training results.")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("PR-AUC", pct(metric_value(saved, "pr_auc", "PR-AUC")))
    p2.metric("ROC-AUC", pct(metric_value(saved, "roc_auc", "ROC-AUC")))
    p3.metric("PRECISION", pct(metric_value(saved, "precision", "Precision")))
    p4.metric("RECALL", pct(metric_value(saved, "recall", "Recall")))

    if dataset is not None and "Class" in dataset:
        section("DATA INTELLIGENCE", "Transaction distribution",
                "The dataset contains a highly imbalanced distribution of payment classes.")

        fig = go.Figure(data=[go.Pie(
            labels=["Genuine", "Fraud"],
            values=[genuine, fraud],
            hole=.72,
            marker=dict(colors=["#52ff89", "#ff6e9e"]),
            textinfo="label+percent",
            textfont=dict(color="white", size=13)
        )])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            annotations=[dict(
                text="TRANSACTIONS",
                x=.5, y=.5,
                font=dict(size=13, color="#0DA23F"),
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)


# -------------------- FRAUD SCANNER --------------------

elif page == "⌕  Fraud Scanner":

    section("LIVE ANALYSIS", "Fraud scanner",
            "Inspect a single transaction using your trained machine-learning model.")

    st.markdown("""
    <div class="glass">
      <div class="kicker">SCANNER INFORMATION</div>
      <h3>Transaction intelligence input</h3>
      <p>V1–V28 are anonymised PCA features from the original dataset.
      Enter actual feature values for a meaningful prediction. Zero values
      are placeholders, not a real transaction.</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_ready:
        st.error(model_error or "Model unavailable.")
    else:
        with st.form("scanner_form"):
            st.subheader("Transaction details")
            a, b = st.columns(2)
            time_value = a.number_input("Transaction time", min_value=0.0, value=0.0)
            amount_value = b.number_input("Transaction amount", min_value=0.0, value=100.0)

            st.markdown("#### Anonymised PCA features")
            values = {}
            cols = st.columns(4)

            for i in range(1, 29):
                with cols[(i - 1) % 4]:
                    values[f"V{i}"] = st.number_input(
                        f"V{i}", value=0.0, step=0.1,
                        format="%.4f", key=f"scanner_{i}"
                    )

            submitted = st.form_submit_button(
                "✦  ANALYSE TRANSACTION",
                use_container_width=True
            )

        if submitted:
            row = {"Time": time_value, "Amount": amount_value, **values}

            try:
                result = predict_data(pd.DataFrame([row]))
                probability = float(result.iloc[0]["Fraud Probability"])
                prediction = result.iloc[0]["Prediction"]
                risk = result.iloc[0]["Risk Level"]

                if prediction == "FRAUD":
                    st.markdown(f"""
                    <div class="result-fraud">
                      <div class="eyebrow" style="color:#ff86a5">⚠ &nbsp; SECURITY ALERT · {risk} RISK</div>
                      <h2 style="color:#ff9eb5">Potential fraud detected</h2>
                      <p style="color:#e5b8c6">Estimated fraud probability: <b>{probability:.2%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                      <div class="eyebrow">✓ &nbsp; ANALYSIS COMPLETE · {risk} RISK</div>
                      <h2 style="color:#65f1c3">No fraud signal detected</h2>
                      <p style="color:#b1dfd2">Estimated fraud probability: <b>{probability:.2%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(probability, text="Estimated fraud probability")
                st.caption("Model estimates are not guarantees of transaction safety.")

            except Exception as error:
                st.error(str(error))


# -------------------- BATCH ANALYSIS --------------------

elif page == "▤  Batch Intelligence":

    section("BATCH PROCESSING", "Batch intelligence",
            "Screen multiple transactions in one analysis.")

    if not model_ready:
        st.error(model_error or "Model unavailable.")
    else:
        st.markdown("""
        <div class="glass">
          <div class="kicker">BULK TRANSACTION SCAN</div>
          <h3>Upload your transaction dataset</h3>
          <p>Required columns: Time, Amount, V1–V28. The Class column is optional.</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded:
            try:
                batch = pd.read_csv(uploaded)
                st.write(f"**{len(batch):,} transactions uploaded**")

                missing = [f for f in FEATURES if f not in batch.columns]

                if missing:
                    st.error("Missing columns: " + ", ".join(missing))
                elif st.button("RUN BATCH ANALYSIS", use_container_width=True):
                    scored = predict_data(batch)

                    fraud_count = int((scored["Prediction"] == "FRAUD").sum())
                    genuine_count = len(scored) - fraud_count

                    a, b, c = st.columns(3)
                    a.metric("ANALYSED", f"{len(scored):,}")
                    b.metric("FLAGGED", f"{fraud_count:,}")
                    c.metric("GENUINE", f"{genuine_count:,}")

                    st.dataframe(scored, use_container_width=True, height=400)

                    st.download_button(
                        "DOWNLOAD ANALYSIS REPORT",
                        scored.to_csv(index=False).encode("utf-8"),
                        "fraudshield_report.csv",
                        "text/csv",
                        use_container_width=True
                    )

                    fig = px.histogram(
                        scored,
                        x="Fraud Probability",
                        color="Risk Level",
                        nbins=35,
                        color_discrete_map={
                            "HIGH": "#ff668b",
                            "MEDIUM": "#b28aff",
                            "LOW": "#05421CC1"
                        },
                        title="Risk distribution across transactions"
                    )
                    st.plotly_chart(chart_style(fig), use_container_width=True)

            except Exception as error:
                st.error(f"Could not analyse file: {error}")


# -------------------- TRANSACTION EXPLORER --------------------

elif page == "◉  Transaction Explorer":

    section("DATA EXPLORATION", "Transaction explorer",
            "Explore transaction records and compare fraud patterns.")

    if dataset is None:
        st.warning("Dataset not found in data/creditcard.csv.")
    else:
        st.markdown(f"**Dataset size:** {len(dataset):,} transactions")

        if "Class" in dataset:
            choice = st.selectbox(
                "Select transaction category",
                ["All transactions", "Genuine transactions", "Fraudulent transactions"]
            )

            view = dataset.copy()
            if choice == "Genuine transactions":
                view = view[view["Class"] == 0]
            elif choice == "Fraudulent transactions":
                view = view[view["Class"] == 1]

            st.dataframe(view.head(500), use_container_width=True, height=430)
            st.caption("Showing up to 500 rows. The original dataset is unchanged.")

        else:
            st.dataframe(dataset.head(500), use_container_width=True)


# -------------------- MODEL INTELLIGENCE --------------------

elif page == "✧  Model Intelligence":

    section("MODEL LAB", "Model intelligence",
            "Compare trained algorithms and understand evaluation performance.")

    saved = metrics_data.get("metrics", metrics_data.get("best_metrics", {}))

    cols = st.columns(5)
    metric_keys = [
        ("PR-AUC", ("pr_auc", "PR-AUC")),
        ("ROC-AUC", ("roc_auc", "ROC-AUC")),
        ("PRECISION", ("precision", "Precision")),
        ("RECALL", ("recall", "Recall")),
        ("F1 SCORE", ("f1", "F1", "F1 Score"))
    ]

    for col, (label, keys) in zip(cols, metric_keys):
        col.metric(label, pct(metric_value(saved, *keys)))

    rows = metrics_data.get(
        "all_models",
        metrics_data.get("model_comparison", metrics_data.get("models", []))
    )

    if isinstance(rows, dict):
        rows = [
            {"model": name, **(values if isinstance(values, dict) else {})}
            for name, values in rows.items()
        ]

    if rows:
        records = []

        for item in rows:
            records.append({
                "Model": metric_value(item, "model", "Model", "name", "Name") or "Model",
                "PR-AUC": metric_value(item, "pr_auc", "PR-AUC", "PR AUC"),
                "ROC-AUC": metric_value(item, "roc_auc", "ROC-AUC", "ROC AUC"),
                "Precision": metric_value(item, "precision", "Precision"),
                "Recall": metric_value(item, "recall", "Recall"),
                "F1 Score": metric_value(item, "f1", "F1", "F1 Score")
            })

        table = pd.DataFrame(records)

        section("ALGORITHM COMPARISON", "Training results",
                "Evaluation scores from your saved model training report.")

        st.dataframe(table, use_container_width=True, hide_index=True)

        chart = table.dropna(subset=["PR-AUC"])

        if not chart.empty:
            fig = go.Figure()

            palette = ["#52e7ff", "#50efc0", "#b28aff", "#ff76b7"]

            fig.add_trace(go.Bar(
                x=chart["Model"],
                y=chart["PR-AUC"],
                marker=dict(
                    color=palette[:len(chart)],
                    line=dict(color="rgba(255,255,255,.4)", width=1)
                ),
                text=[f"{float(v):.3f}" for v in chart["PR-AUC"]],
                textposition="outside"
            ))

            fig.update_layout(
                title="Precision-Recall AUC comparison",
                yaxis=dict(range=[0, 1]),
                showlegend=False
            )

            st.plotly_chart(chart_style(fig), use_container_width=True)
    else:
        st.info("Model comparison data is not available in model_metrics.json.")

    with st.expander("Understand the metrics"):
        st.write("""
        - **Precision:** How many flagged transactions were actually fraudulent.
        - **Recall:** How many actual fraudulent transactions the model detected.
        - **F1 Score:** Balance between precision and recall.
        - **ROC-AUC:** How well the model distinguishes the two classes.
        - **PR-AUC:** Particularly useful for highly imbalanced datasets.
        """)


# -------------------- DATASET INSIGHTS --------------------

elif page == "▧  Dataset Insights":

    section("DATA INTELLIGENCE", "Dataset insights",
            "Discover the patterns and imbalance in the credit card dataset.")

    if dataset is None:
        st.warning("Dataset unavailable.")
    else:
        if "Class" in dataset:
            counts = dataset["Class"].value_counts().sort_index()

            a, b, c = st.columns(3)
            a.metric("TOTAL RECORDS", f"{len(dataset):,}")
            a.metric("GENUINE", f"{int(counts.get(0, 0)):,}")
            c.metric("FRAUD", f"{int(counts.get(1, 0)):,}")

            left, right = st.columns(2, gap="large")

            with left:
                fig = px.pie(
                    names=["Genuine", "Fraud"],
                    values=[int(counts.get(0, 0)), int(counts.get(1, 0))],
                    hole=.65,
                    color_discrete_sequence=["#228d5d", "#ff76b7"],
                    title="Class distribution"
                )
                st.plotly_chart(chart_style(fig), use_container_width=True)

            with right:
                amount_fig = px.box(
                    dataset,
                    x="Class",
                    y="Amount",
                    color="Class",
                    color_discrete_map={0: "#239f6b", 1: "#ff76b7"},
                    title="Transaction amount by class"
                )
                st.plotly_chart(chart_style(amount_fig), use_container_width=True)

            fig = px.histogram(
                dataset,
                x="Amount",
                color="Class",
                nbins=50,
                color_discrete_map={0: "#15824A", 1: "#ff76b7"},
                title="Transaction amount distribution"
            )
            st.plotly_chart(chart_style(fig, 400), use_container_width=True)

            st.caption("The source dataset uses anonymised PCA features V1–V28.")
        else:
            st.info("Class column not found in dataset.")


# -------------------- ABOUT --------------------

elif page == "ⓘ  About Platform":

    section("PLATFORM PROFILE", "About FraudShield AI",
            "An intelligent machine-learning prototype for financial security.")

    st.markdown("""
    <div class="hero">
      <div class="hero-content">
        <div class="eyebrow">MCA ARTIFICIAL INTELLIGENCE & MACHINE LEARNING</div>
        <h1>Intelligence.<br>Security. Trust.</h1>
        <p>
        FraudShield AI analyses credit card transaction patterns using
        supervised machine learning to estimate potential fraudulent activity.
        </p>
        <div class="pill"><span class="dot"></span> MACHINE LEARNING PLATFORM</div>
      </div>
      <div class="shield">⛨</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("""
        <div class="glass">
          <div class="kicker">THE CHALLENGE</div>
          <h3>Financial fraud detection</h3>
          <p>Fraudulent payments are rare compared with genuine transactions.
          This class imbalance makes identifying fraud a challenging task.</p>
          <div class="kicker">THE APPROACH</div>
          <h3>Intelligent risk estimation</h3>
          <p>A trained machine-learning model analyses transaction features
          and estimates the probability of fraud.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="glass">
          <div class="kicker">TECHNOLOGY STACK</div>
          <h3>Built with</h3>
          <div class="pipeline">Python & Streamlit</div>
          <div class="pipeline">Pandas & NumPy</div>
          <div class="pipeline">Scikit-learn & Imbalanced-learn</div>
          <div class="pipeline">Random Forest & XGBoost</div>
          <div class="pipeline">Plotly & Joblib</div>
        </div>
        """, unsafe_allow_html=True)

    st.info("FraudShield AI is an educational prototype, not a production banking security system.")


# -------------------- FOOTER --------------------

st.markdown("""
<div class="footer">
FRAUDSHIELD AI &nbsp; ✦ &nbsp; FINANCIAL INTELLIGENCE CENTER
<br><br>
POWERED BY MACHINE LEARNING
</div>
""", unsafe_allow_html=True)
