import streamlit as st
import requests
import sys
import os
import pandas as pd
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.explainability import SHAPExplainer

# ── API URL: env variable takes priority (production), config is fallback (local) ──
API_URL         = os.environ.get("API_URL", CONFIG["ui"]["api_url"])
API_EXPLAIN_URL = API_URL.replace("/predict", "/explain")
API_HEALTH_URL  = API_URL.replace("/predict", "")

# ── Caching Local Model & Explainer for Fallback Mode ──────────────────────
@st.cache_resource
def load_local_engine():
    try:
        pipeline = PredictionPipeline()
        explainer = SHAPExplainer(pipeline.model, pipeline.vectorizer)
        return pipeline, explainer
    except Exception as e:
        st.error(f"Failed to load local prediction engine: {e}")
        return None, None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Trust-Hire · Fake Job Detector",
    page_icon  = "🛡️",
    layout     = "wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 24px;
    }
    .result-fake {
        background: #ff4b4b22;
        border-left: 4px solid #ff4b4b;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ff4b4b;
    }
    .result-real {
        background: #21c55d22;
        border-left: 4px solid #21c55d;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 700;
        color: #21c55d;
    }
    .shap-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🛡️ Trust-Hire</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered Fake Job Detection · XGBoost + TF-IDF + SHAP Explainability</div>",
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    def check_api():
        try:
            res = requests.get(API_HEALTH_URL, timeout=1.5)
            return res.status_code == 200
        except:
            return False

    api_online = check_api()
    if api_online:
        st.success("✅ API Backend Online")
        st.caption("Running predictions via external FastAPI backend.")
    else:
        st.info("ℹ️ Standalone Mode Enabled")
        st.caption("No external backend needed. Model & SHAP explainer are running locally in the Streamlit container.")

    st.markdown("---")
    top_n = st.slider("Top SHAP features to show", min_value=5, max_value=15, value=10)
    st.markdown("---")
    st.info("Paste a job description or upload a `.txt` file, then click **Analyze**.")
    st.markdown("---")
    st.markdown("""
    **How SHAP works:**
    - 🔴 Red bars → words pushing towards **Fake**
    - 🟢 Green bars → words pushing towards **Real**
    - Longer bar = stronger influence
    """)

# ── Input Tabs ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📄 Manual Input", "📂 Upload File"])
text = ""

with tab1:
    text = st.text_area("Enter Job Description", height=220,
                         placeholder="e.g. We are hiring a Python developer with 3 years experience...")

with tab2:
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.success(f"File loaded: {uploaded_file.name}")
        st.text_area("Preview", text[:500] + ("..." if len(text) > 500 else ""), height=150, disabled=True)

# ── Analyze Button ────────────────────────────────────────────────────────────
analyze = st.button("🔍 Analyze Job Posting", use_container_width=True, type="primary")

if analyze:
    if not text.strip():
        st.warning("⚠️ Please provide a job description.")
    elif len(text.strip()) < 10:
        st.warning("⚠️ Job description is too short (min 10 characters).")
    else:
        with st.spinner("Analyzing with AI + computing SHAP explanations... 🤖"):
            prediction = None
            confidence = None
            top_features = []

            # ── Run using API if online, otherwise Fallback to Local Engine ──
            if api_online:
                try:
                    response = requests.post(
                        API_EXPLAIN_URL,
                        json={"description": text},
                        timeout=10
                    )
                    response.raise_for_status()
                    data = response.json()
                    prediction   = data["prediction"]
                    confidence   = data["confidence"]  # probability of class 1 (Fake)
                    top_features = data["top_features"]
                except Exception as api_err:
                    st.error(f"API Error: {api_err}. Falling back to local engine...")
                    api_online = False

            if not api_online:
                # Local execution
                pipeline, explainer = load_local_engine()
                if pipeline and explainer:
                    try:
                        # Vectorize and predict probability directly
                        vec = pipeline.vectorizer.transform([text])
                        prob = pipeline.model.predict_proba(vec)[0][1] # P(Fake)
                        
                        prediction = "Fake Job" if prob > 0.5 else "Real Job"
                        confidence = float(prob)
                        top_features = explainer.explain(text, top_n=15)
                    except Exception as local_err:
                        st.error(f"Local Engine Error: {local_err}")

            # ── Render UI Results if success ──────────────────────────────────
            if prediction is not None and confidence is not None:
                # ── Result Banner ─────────────────────────────────────────────
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    css_cls = "result-fake" if prediction == "Fake Job" else "result-real"
                    icon    = "🚨" if prediction == "Fake Job" else "✅"
                    st.markdown(f"<div class='{css_cls}'>{icon} {prediction}</div>",
                                unsafe_allow_html=True)

                with col2:
                    fake_pct = round(confidence * 100, 1)
                    st.metric("Fake Probability", f"{fake_pct}%")

                with col3:
                    real_pct = round((1 - confidence) * 100, 1)
                    st.metric("Real Probability", f"{real_pct}%")

                # Progress bar
                st.progress(int(confidence * 100),
                            text=f"Fake score: {fake_pct}%")

                # ── SHAP Explanation Chart ────────────────────────────────────
                if top_features:
                    st.markdown("---")
                    st.markdown("### 🧠 SHAP Explanation — Why did the model decide this?")
                    st.markdown(
                        "<div class='shap-label'>Top words influencing the prediction</div>",
                        unsafe_allow_html=True
                    )

                    # Trim to requested top_n
                    features = top_features[:top_n]

                    words  = [f["word"] for f in features]
                    values = [f["shap_value"] for f in features]
                    dirs   = [f["direction"] for f in features]

                    colors = ["#ff4b4b" if d == "fake" else "#21c55d" for d in dirs]

                    fig = go.Figure(go.Bar(
                        x           = values,
                        y           = words,
                        orientation = "h",
                        marker_color= colors,
                        text        = [f"{v:+.3f}" for v in values],
                        textposition= "outside",
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "SHAP value: %{x:.4f}<br>"
                            "<extra></extra>"
                        )
                    ))

                    fig.update_layout(
                        xaxis_title  = "SHAP Value  (+ = Fake, - = Real)",
                        yaxis_title  = "Word / Feature",
                        yaxis        = dict(autorange="reversed"),
                        plot_bgcolor = "rgba(0,0,0,0)",
                        paper_bgcolor= "rgba(0,0,0,0)",
                        font         = dict(size=13),
                        height       = max(350, top_n * 38),
                        margin       = dict(l=10, r=80, t=20, b=40),
                        xaxis        = dict(zeroline=True, zerolinecolor="#555", zerolinewidth=1.5),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # ── Feature table ─────────────────────────────────────────────
                    with st.expander("📊 Raw SHAP values (table)"):
                        df = pd.DataFrame(features)
                        df["direction"] = df["direction"].map(
                            {"fake": "🔴 Fake", "real": "🟢 Real"}
                        )
                        df.columns = ["Word", "SHAP Value", "Pushes Towards"]
                        st.dataframe(df, use_container_width=True, hide_index=True)

                # ── History ───────────────────────────────────────────────────
                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "text"      : text[:100],
                    "prediction": prediction,
                    "confidence": f"{fake_pct}%"
                })

# ── History ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📜 Prediction History")

if "history" in st.session_state and st.session_state.history:
    for item in reversed(st.session_state.history):
        icon = "🚨" if item["prediction"] == "Fake Job" else "✅"
        st.write(f"{icon} **{item['prediction']}** ({item['confidence']}) → {item['text']}...")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions yet — analyze a job description above.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
---
<center style='color:#666; font-size:0.85rem;'>
    Trust-Hire &nbsp;·&nbsp; Built with Streamlit + XGBoost + SHAP 🚀
</center>
""", unsafe_allow_html=True)