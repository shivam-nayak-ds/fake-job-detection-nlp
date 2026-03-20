import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Fake Job Detector", layout="wide")

# ================= HEADER =================
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>🧠 Fake Job Detection System</h1>
<p style='text-align:center; color:gray;'>AI-powered job fraud detection</p>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Settings")

# API status check
def check_api():
    try:
        res = requests.get("http://127.0.0.1:8000", timeout=2)
        return res.status_code == 200
    except:
        return False

if check_api():
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Down")

st.sidebar.markdown("---")
st.sidebar.info("Upload job description or paste manually.")

# ================= TABS =================
tab1, tab2 = st.tabs(["📄 Manual Input", "📂 Upload File"])

text = ""

# ================= TAB 1 =================
with tab1:
    text = st.text_area("Enter Job Description", height=250)

# ================= TAB 2 =================
with tab2:
    uploaded_file = st.file_uploader("Upload TXT file", type=["txt"])

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.success("File loaded successfully!")

# ================= ANALYZE =================
if st.button("🚀 Analyze", use_container_width=True):

    if not text.strip():
        st.warning("Please provide job description")

    else:
        with st.spinner("Analyzing... 🤖"):

            try:
                response = requests.post(
                    API_URL,
                    json={"description": text},
                    timeout=5
                )

                data = response.json()

                prediction = data["prediction"]
                confidence = data["confidence"]

                # result UI
                col1, col2 = st.columns(2)

                with col1:
                    if prediction == "Fake Job":
                        st.error(f"🚨 {prediction}")
                    else:
                        st.success(f"✅ {prediction}")

                with col2:
                    st.metric("Confidence", f"{confidence}")

                st.progress(min(int(confidence * 100), 100))

                # ================= HISTORY =================
                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "text": text[:100],
                    "prediction": prediction
                })

            except Exception as e:
                st.error(f"Error: {e}")

# ================= HISTORY =================
st.markdown("### 📜 Prediction History")

if "history" in st.session_state:
    for item in st.session_state.history[::-1]:
        st.write(f"**{item['prediction']}** → {item['text']}...")
else:
    st.info("No predictions yet")

# ================= FOOTER =================
st.markdown("""
---
<center style='color:gray;'>Built with FastAPI + Streamlit 🚀</center>
""", unsafe_allow_html=True)