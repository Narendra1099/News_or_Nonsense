"""
app.py
------
Streamlit web app for the Fake News Detector — playful, colorful UI.

Run with:
    streamlit run app.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from predict import FakeNewsPredictor
from explain import explain_linear

st.set_page_config(page_title="News or Nonsense?", page_icon="🦉", layout="centered")

# ---------------------------------------------------------------------------
# SAMPLE ARTICLES
# ---------------------------------------------------------------------------
SAMPLES = {
    "real": (
        "The Federal Reserve announced on Wednesday that it would hold interest "
        "rates steady, citing signs of slowing inflation and a stable labor "
        "market. Officials said they would continue monitoring economic data "
        "before making further adjustments."
    ),
    "fake": (
        "Scientists confirm the earth is flat, NASA admits massive decades-long "
        "cover-up in shocking leaked documents. Government officials refuse to "
        "comment as public outrage grows."
    ),
}

if "article_text" not in st.session_state:
    st.session_state.article_text = ""


def load_sample(kind: str):
    st.session_state.article_text = SAMPLES[kind]


# ---------------------------------------------------------------------------
# CUSTOM CSS — playful / Duolingo-style aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Nunito:wght@500;600;700;800&display=swap');

    :root {
        --green: #58CC02;
        --green-dark: #45A302;
        --blue: #1CB0F6;
        --blue-dark: #1899D6;
        --yellow: #FFC800;
        --yellow-dark: #E6B400;
        --red: #FF4B4B;
        --red-dark: #E63E3E;
        --purple: #CE82FF;
        --purple-dark: #B569E8;
        --bg: #F7F7FF;
        --text: #3C3C3C;
        --text-light: #777777;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 680px; }

    /* Floating decorative color blobs behind everything */
    .blob {
        position: fixed; border-radius: 50%; filter: blur(50px); opacity: 0.35; z-index: -1;
    }
    .blob1 { width: 260px; height: 260px; background: var(--blue); top: -60px; left: -80px; }
    .blob2 { width: 220px; height: 220px; background: var(--yellow); top: 30%; right: -90px; }
    .blob3 { width: 240px; height: 240px; background: var(--purple); bottom: -60px; left: 10%; }

    /* Header */
    .hero { text-align: center; margin-bottom: 6px; }
    .hero .badge-circle {
        width: 68px; height: 68px; border-radius: 50%; background: var(--yellow);
        display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto;
        font-size: 2.1rem; box-shadow: 0 5px 0 var(--yellow-dark);
    }
    .hero h1 {
        font-family: 'Baloo 2', sans-serif; font-weight: 800; font-size: 2.5rem;
        color: var(--text); margin: 0; letter-spacing: -0.5px;
    }
    .hero p {
        font-family: 'Nunito', sans-serif; font-weight: 600; font-size: 1.02rem;
        color: var(--text-light); margin: 8px auto 0 auto; max-width: 440px; line-height: 1.5;
    }

    /* Colorful stat pills */
    .stat-row { display: flex; justify-content: center; gap: 10px; margin: 22px 0 26px 0; flex-wrap: wrap; }
    .stat-pill {
        font-family: 'Nunito', sans-serif; font-weight: 800; font-size: 0.78rem; color: white;
        padding: 8px 16px; border-radius: 999px;
    }
    .stat-pill.p1 { background: var(--blue); }
    .stat-pill.p2 { background: var(--purple); }
    .stat-pill.p3 { background: var(--green); }

    /* Section headers */
    .step-label {
        font-family: 'Baloo 2', sans-serif; font-weight: 700; font-size: 1.05rem;
        color: var(--text); margin: 22px 0 10px 0; display: flex; align-items: center; gap: 8px;
    }
    .step-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 50%; background: var(--purple);
        color: white; font-size: 0.85rem; font-family: 'Nunito', sans-serif; font-weight: 800;
    }

    /* Card wrapper for the form area */
    .form-card {
        background: white; border-radius: 24px; padding: 26px 26px 22px 26px;
        box-shadow: 0 4px 0 rgba(0,0,0,0.06), 0 10px 30px rgba(0,0,0,0.06);
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #F5F5FF !important; border: 2px solid #E5E5F5 !important;
        border-radius: 14px !important; font-family: 'Nunito', sans-serif !important; font-weight: 700 !important;
    }

    .stTextArea textarea {
        background-color: #F5F5FF !important; border: 2px solid #E5E5F5 !important;
        border-radius: 14px !important; font-family: 'Nunito', sans-serif !important;
        font-size: 0.95rem !important; color: var(--text) !important;
    }
    .stTextArea textarea:focus { border-color: var(--blue) !important; box-shadow: none !important; }

    /* Chunky 3D primary button (Duolingo signature) */
    .stButton button[kind="primary"] {
        background-color: var(--green) !important; color: white !important;
        font-family: 'Baloo 2', sans-serif !important; font-weight: 700 !important;
        letter-spacing: 0.5px !important; text-transform: uppercase !important;
        font-size: 0.95rem !important; border-radius: 16px !important; border: none !important;
        padding: 0.75rem 1.6rem !important; width: 100%;
        box-shadow: 0 5px 0 var(--green-dark) !important;
        transition: transform 0.08s ease, box-shadow 0.08s ease;
    }
    .stButton button[kind="primary"]:hover { filter: brightness(1.04); }
    .stButton button[kind="primary"]:active {
        transform: translateY(4px) !important; box-shadow: 0 1px 0 var(--green-dark) !important;
    }

    /* Chunky secondary sample buttons */
    .stButton button[kind="secondary"] {
        background-color: white !important; color: var(--blue-dark) !important;
        font-family: 'Nunito', sans-serif !important; font-weight: 800 !important;
        font-size: 0.82rem !important; border: 2px solid var(--blue) !important;
        border-radius: 14px !important; padding: 0.5rem 0.9rem !important;
        box-shadow: 0 3px 0 var(--blue) !important;
        transition: transform 0.08s ease, box-shadow 0.08s ease;
    }
    .stButton button[kind="secondary"]:hover { background-color: #EAF7FF !important; }
    .stButton button[kind="secondary"]:active {
        transform: translateY(3px) !important; box-shadow: 0 0px 0 var(--blue) !important;
    }

    /* Verdict badge with bounce-in */
    @keyframes popIn {
        0% { transform: scale(0.3); opacity: 0; }
        60% { transform: scale(1.08); opacity: 1; }
        80% { transform: scale(0.96); }
        100% { transform: scale(1); }
    }
    .verdict-card {
        margin-top: 24px; border-radius: 22px; padding: 26px 20px; text-align: center;
        animation: popIn 0.5s cubic-bezier(.2,.9,.3,1.3) both;
    }
    .verdict-card.real { background: #EAFBDD; border: 3px solid var(--green); }
    .verdict-card.fake { background: #FFEAEA; border: 3px solid var(--red); }
    .verdict-emoji { font-size: 2.6rem; line-height: 1; margin-bottom: 6px; }
    .verdict-title {
        font-family: 'Baloo 2', sans-serif; font-weight: 800; font-size: 1.6rem;
    }
    .verdict-card.real .verdict-title { color: var(--green-dark); }
    .verdict-card.fake .verdict-title { color: var(--red-dark); }
    .verdict-sub {
        font-family: 'Nunito', sans-serif; font-weight: 700; font-size: 0.85rem;
        color: var(--text-light); margin-top: 4px;
    }

    .stProgress > div > div > div { background-color: #E5E5F5 !important; border-radius: 999px !important; }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--red), var(--yellow), var(--green)) !important;
        border-radius: 999px !important;
    }

    /* Influencing word chips */
    .chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .word-chip {
        font-family: 'Nunito', sans-serif; font-weight: 800; font-size: 0.78rem;
        padding: 7px 13px; border-radius: 999px; color: white;
    }
    .word-chip.real { background: var(--green); }
    .word-chip.fake { background: var(--red); }

    .footnote {
        font-family: 'Nunito', sans-serif; font-weight: 600; font-size: 0.78rem;
        color: var(--text-light); text-align: center; margin-top: 30px; line-height: 1.5;
    }
    </style>

    <div class="blob blob1"></div>
    <div class="blob blob2"></div>
    <div class="blob blob3"></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="badge-circle">🦉</div>
        <h1>News or Nonsense?</h1>
        <p>Paste a story below and let the checker guess whether it's the
        real deal or totally made up — no judgment, just vibes and data!</p>
    </div>
    <div class="stat-row">
        <span class="stat-pill p1">44K+ articles trained</span>
        <span class="stat-pill p2">TF-IDF + ML</span>
        <span class="stat-pill p3">Shows its work</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# FORM CARD
# ---------------------------------------------------------------------------
st.markdown('<div class="form-card">', unsafe_allow_html=True)

MODEL_OPTIONS = {
    "Logistic Regression (fast, solid baseline)": "logistic",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}

st.markdown('<div class="step-label"><span class="step-num">1</span> Pick a checker</div>', unsafe_allow_html=True)
model_label = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")
model_name = MODEL_OPTIONS[model_label]

st.markdown('<div class="step-label"><span class="step-num">2</span> Paste your story</div>', unsafe_allow_html=True)
article_text = st.text_area(
    "Article text",
    height=200,
    placeholder="Paste the article's title and/or body text here...",
    label_visibility="collapsed",
    key="article_text",
)

sample_col1, sample_col2 = st.columns(2)
with sample_col1:
    st.button("✅ Try a real one", type="secondary", on_click=load_sample, args=("real",))
with sample_col2:
    st.button("🚨 Try a fake one", type="secondary", on_click=load_sample, args=("fake",))

st.write("")
analyze_clicked = st.button("Check it!", type="primary")

st.markdown('</div>', unsafe_allow_html=True)  # close .form-card

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not article_text.strip():
        st.warning("Paste some text first — I need something to read! 👀")
    else:
        try:
            predictor = FakeNewsPredictor(model_name=model_name)
            result = predictor.predict(article_text)
            label = result["label"]
            confidence = result["confidence"]

            if label == "real":
                emoji, title, sub = "✅", "Looks Real!", "This reads like genuine reporting."
            else:
                emoji, title, sub = "🚨", "Smells Fake!", "This has the hallmarks of made-up news."

            st.markdown(
                f"""
                <div class="verdict-card {label}">
                    <div class="verdict-emoji">{emoji}</div>
                    <div class="verdict-title">{title}</div>
                    <div class="verdict-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence is not None:
                st.write("")
                st.markdown(
                    f'<p style="font-family:\'Nunito\',sans-serif; font-weight:800; text-align:center;">'
                    f"Confidence: {confidence * 100:.0f}%</p>",
                    unsafe_allow_html=True,
                )
                st.progress(confidence)

            if model_name in ("logistic", "naive_bayes"):
                pairs = explain_linear(article_text, model_name=model_name, top_n=10)
                if pairs:
                    st.markdown(
                        '<div class="step-label" style="margin-top:22px;">🔍 What tipped it off</div>',
                        unsafe_allow_html=True,
                    )
                    chips = '<div class="chip-wrap">'
                    for word, weight in pairs:
                        chip_class = "real" if weight > 0 else "fake"
                        chips += f'<span class="word-chip {chip_class}">{word}</span>'
                    chips += "</div>"
                    st.markdown(chips, unsafe_allow_html=True)

        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Train a model first: `python src/train.py --model logistic`")

st.markdown(
    """
    <div class="footnote">
        ⚠️ This is a statistical guess based on writing patterns, not a fact-checker.
        It can be wrong — always double-check important claims yourself!
    </div>
    """,
    unsafe_allow_html=True,
)
