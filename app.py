"""Streamlit Web Application for Spam Email Classifier.

A modern, futuristic interface for real-time email and SMS classification,
powered by TF-IDF Vectorization and Multinomial Naive Bayes.
"""

import json
from pathlib import Path
import re
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import METRICS_FILE_PATH, MODEL_FILE_PATH
from src.predict import load_model, predict_message, top_spam_words

# Example messages for instant testing
SPAM_EXAMPLE: str = (
    "WINNER!! As a valued customer, you have been selected to receive a $1,000 "
    "cash reward! Call 09061701461 now with claim code KL341 to claim your prize."
)
HAM_EXAMPLE: str = (
    "Hey! Are we still meeting for lunch at 12:30 today? Let me know if you need "
    "a ride from campus."
)

FUTURISTIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Base page aesthetics */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* App Background & glow effect */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(13, 22, 47, 0.95) 0%, rgba(8, 12, 22, 0.98) 90%);
    color: #e2e8f0;
}

/* Hero Badge */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(129, 140, 248, 0.3);
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Futuristic Hero Title */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 24px;
    line-height: 1.5;
}

/* Glassmorphism containers */
.glass-panel {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    transition: border-color 0.3s ease;
}

.glass-panel:hover {
    border-color: rgba(99, 102, 241, 0.3);
}

/* Analytics chips */
.stat-chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
}

.stat-chip {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.8rem;
    color: #cbd5e1;
    font-family: 'JetBrains Mono', monospace;
}

/* Result Alert Cards */
.result-card-spam {
    background: radial-gradient(circle at 10% 10%, rgba(239, 68, 68, 0.15) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(239, 68, 68, 0.5);
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
}

.result-card-ham {
    background: radial-gradient(circle at 10% 10%, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(16, 185, 129, 0.5);
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.verdict-tag {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

.tag-spam {
    color: #f87171;
    text-shadow: 0 0 12px rgba(248, 113, 113, 0.6);
}

.tag-ham {
    color: #34d399;
    text-shadow: 0 0 12px rgba(52, 211, 153, 0.6);
}

.confidence-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* Sidebar metric card */
.sidebar-metric-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
    text-align: center;
}

.sidebar-metric-title {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.sidebar-metric-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}

/* Keyword pill badge */
.keyword-badge {
    display: inline-flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 6px 10px;
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.85rem;
}

.keyword-name {
    color: #e2e8f0;
    font-weight: 500;
}

.keyword-score {
    color: #f43f5e;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}

/* Streamlit textarea & button customizations */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.35) !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #06b6d4 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}

div.stButton > button[kind="secondary"] {
    background: rgba(30, 41, 59, 0.7) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: #818cf8 !important;
    color: #ffffff !important;
    background: rgba(49, 46, 129, 0.3) !important;
}
</style>
"""


@st.cache_resource
def get_model() -> Pipeline:
    """Load and cache the trained pipeline so it is only loaded once into memory.

    Returns:
        Pipeline: Cached trained scikit-learn pipeline.
    """
    return load_model(MODEL_FILE_PATH)


def load_metrics() -> dict | None:
    """Load evaluation metrics from reports/metrics.json if present.

    Returns:
        dict | None: Dictionary of metric scores or None if report is missing.
    """
    if METRICS_FILE_PATH.exists():
        with open(METRICS_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def set_example_text(sample_text: str) -> None:
    """Callback function to populate the text area with an example message.

    Args:
        sample_text: Text string to insert into session state.
    """
    st.session_state["message_input"] = sample_text


def compute_text_stats(text: str) -> dict:
    """Calculate quick linguistic metrics for input inspection.

    Args:
        text: Raw user message.

    Returns:
        dict: Word count, character count, uppercase ratio, and link counts.
    """
    if not text:
        return {"words": 0, "chars": 0, "upper_pct": 0, "urls": 0}

    words = len(text.split())
    chars = len(text)
    letters = [c for c in text if c.isalpha()]
    upper_count = sum(1 for c in letters if c.isupper())
    upper_pct = round((upper_count / len(letters)) * 100, 1) if letters else 0
    urls = len(re.findall(r"https?://\S+|www\.\S+", text, re.IGNORECASE))

    return {
        "words": words,
        "chars": chars,
        "upper_pct": upper_pct,
        "urls": urls,
    }


def render_sidebar(pipeline: Pipeline) -> None:
    """Render the cyber-security sidebar with model metrics and top spam keywords.

    Args:
        pipeline: Trained pipeline to extract top spam terms from.
    """
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2rem;">🛡️</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #f8fafc; letter-spacing: 0.05em;">AI SENTINEL</div>
            <div style="font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">SECURITY ENGINE v1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = load_metrics()
    if metrics:
        st.sidebar.markdown(
            f"""
            <div style="padding: 8px 12px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; margin-bottom: 16px; font-size: 0.8rem; text-align: center;">
                <span style="color: #94a3b8;">Active Model:</span><br/>
                <strong style="color: #a5b4fc;">{metrics.get('model_name', 'Multinomial Naive Bayes')}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2x2 Metric Grid
        c1, c2 = st.sidebar.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="sidebar-metric-box">
                    <div class="sidebar-metric-title">Accuracy</div>
                    <div class="sidebar-metric-value">{metrics.get('accuracy', 0.0) * 100:.1f}%</div>
                </div>
                <div class="sidebar-metric-box">
                    <div class="sidebar-metric-title">Precision</div>
                    <div class="sidebar-metric-value" style="color: #10b981;">{metrics.get('precision', 0.0) * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="sidebar-metric-box">
                    <div class="sidebar-metric-title">Recall</div>
                    <div class="sidebar-metric-value">{metrics.get('recall', 0.0) * 100:.1f}%</div>
                </div>
                <div class="sidebar-metric-box">
                    <div class="sidebar-metric-title">F1-Score</div>
                    <div class="sidebar-metric-value" style="color: #818cf8;">{metrics.get('f1_score', 0.0) * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        cv_mean = metrics.get("cv_5fold_f1_mean")
        if cv_mean is not None:
            st.sidebar.markdown(
                f"""
                <div style="font-size: 0.75rem; color: #94a3b8; text-align: center; margin-top: 6px; font-family: 'JetBrains Mono', monospace;">
                    5-Fold CV F1: <strong style="color: #cbd5e1;">{cv_mean * 100:.2f}%</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.info("Run `python -m src.evaluate` to generate model benchmark metrics.")

    st.sidebar.markdown("<br/>", unsafe_allow_html=True)

    # Top Spam Keywords Expander
    with st.sidebar.expander("🔍 Top 15 Spam Triggers", expanded=False):
        st.caption("Keywords with highest learned log probability:")
        top_words = top_spam_words(n=15, pipeline=pipeline)
        for rank, (word, score) in enumerate(top_words, start=1):
            st.markdown(
                f"""
                <div class="keyword-badge">
                    <span class="keyword-name"><strong>{rank}.</strong> {word}</span>
                    <span class="keyword-score">{score:.2f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    """Render the application hero, controls, and predictions."""
    st.set_page_config(
        page_title="AI Spam Email Classifier",
        page_icon="🛡️",
        layout="centered",
    )

    # Inject custom styling
    st.markdown(FUTURISTIC_CSS, unsafe_allow_html=True)

    # Hero Banner
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div class="hero-badge">
                <span class="pulse-dot"></span>
                TF-IDF + NAIVE BAYES PIPELINE &bull; ONLINE
            </div>
            <div class="hero-title">Spam Email Classifier</div>
            <div class="hero-subtitle">
                Advanced machine learning filter to analyze text payloads and detect phishing, unsolicited ads, and malicious spam solicitations in real time.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Friendly check if model file has not been created yet
    if not MODEL_FILE_PATH.exists():
        st.error(
            "⚠️ Trained model file not found (`models/spam_classifier.joblib`).\n\n"
            "Please run `python -m src.train` in your terminal first."
        )
        st.stop()

    pipeline = get_model()
    render_sidebar(pipeline)

    # Initialize session state for user input
    if "message_input" not in st.session_state:
        st.session_state["message_input"] = ""

    # Quick Samples Container
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;'>⚡ Quick Test Presets</div>", unsafe_allow_html=True)
    col_btn1, col_btn2, _ = st.columns([1.1, 1.1, 1.8])
    with col_btn1:
        st.button(
            "🚨 Load Spam Preset",
            on_click=set_example_text,
            args=(SPAM_EXAMPLE,),
            use_container_width=True,
        )
    with col_btn2:
        st.button(
            "✉️ Load Clean Preset",
            on_click=set_example_text,
            args=(HAM_EXAMPLE,),
            use_container_width=True,
        )

    # Input message text area
    user_text = st.text_area(
        label="Message Payload:",
        key="message_input",
        height=140,
        placeholder="Type, paste, or select a preset message to analyze...",
        label_visibility="collapsed",
    )

    # Real-time text statistics bar
    stats = compute_text_stats(user_text)
    st.markdown(
        f"""
        <div class="stat-chip-container">
            <span class="stat-chip">📝 <strong>{stats['words']}</strong> Words</span>
            <span class="stat-chip">🔤 <strong>{stats['chars']}</strong> Characters</span>
            <span class="stat-chip">🔠 <strong>{stats['upper_pct']}%</strong> Uppercase</span>
            <span class="stat-chip">🔗 <strong>{stats['urls']}</strong> URLs</span>
        </div>
        <br/>
        """,
        unsafe_allow_html=True,
    )

    # Classify action button
    if st.button("🚀 Analyze & Classify Message", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Please provide a message or click one of the preset buttons above.")
        else:
            result = predict_message(user_text, pipeline=pipeline)
            label = result["label"]
            spam_prob = float(result["spam_probability"])
            spam_pct = spam_prob * 100

            if label == "Spam":
                st.markdown(
                    f"""
                    <div class="result-card-spam">
                        <div class="result-header">
                            <div>
                                <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #fca5a5;">Security Threat Level</span>
                                <div class="verdict-tag tag-spam">🚨 SPAM DETECTED</div>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 0.8rem; color: #94a3b8;">Spam Probability</span>
                                <div class="confidence-score" style="color: #f87171;">{spam_pct:.2f}%</div>
                            </div>
                        </div>
                        <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 14px;">
                            This message contains high-frequency spam triggers and patterns typically used in unsolicited promotional campaigns, phishing scams, or financial rewards.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(spam_prob)
            else:
                st.markdown(
                    f"""
                    <div class="result-card-ham">
                        <div class="result-header">
                            <div>
                                <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6ee7b7;">Verification Status</span>
                                <div class="verdict-tag tag-ham">✅ CLEAN / NOT SPAM</div>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 0.8rem; color: #94a3b8;">Spam Probability</span>
                                <div class="confidence-score" style="color: #34d399;">{spam_pct:.2f}%</div>
                            </div>
                        </div>
                        <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 14px;">
                            The text pattern matches standard legitimate communication with a very low spam probability. Safe to read.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(spam_prob)


if __name__ == "__main__":
    main()
