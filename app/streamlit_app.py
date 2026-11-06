"""Streamlit interface for VeriFinVerse."""

import html
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.database import get_history, initialize_database, save_prediction
from src.explain import explain_text
from src.inference import predict_one


def render_highlighted_text(text, signals):
    # Escape both source text and model-provided tokens before rendering HTML.
    ordered = sorted(signals, key=lambda signal: len(signal["word"]), reverse=True)
    directions = {}
    words = []
    for signal in ordered:
        word = str(signal["word"]).strip()
        if word:
            words.append(word)
            directions[word.casefold()] = signal["toward"]
    if not words:
        return html.escape(text)

    pattern = re.compile("|".join(re.escape(word) for word in words), re.IGNORECASE)

    def highlight(match):
        matched_text = match.group(0)
        color = (
            "#ffd1d1"
            if directions[matched_text.casefold()] == "FAKE"
            else "#c9f3d2"
        )
        return f'<mark style="background:{color};">{html.escape(matched_text)}</mark>'

    pieces = []
    previous_end = 0
    for match in pattern.finditer(text):
        pieces.append(html.escape(text[previous_end:match.start()]))
        pieces.append(highlight(match))
        previous_end = match.end()
    pieces.append(html.escape(text[previous_end:]))
    return "".join(pieces)


def main():
    st.set_page_config(page_title="VeriFinVerse", page_icon="📰", layout="wide")
    initialize_database()
    st.title("VeriFinVerse")
    st.caption("Explainable AI for financial fake news detection")

    page = st.sidebar.radio("Page", ["Verify article", "Prediction history"])
    if page == "Prediction history":
        st.subheader("Recent predictions")
        history = pd.DataFrame(get_history())
        if history.empty:
            st.info("No predictions have been saved yet.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)
        return

    model = st.sidebar.selectbox(
        "Model", ["finbert", "logistic", "xgboost"], index=0
    )
    text = st.text_area("Paste a financial news article or post", height=220)
    if st.button("Verify Now", type="primary", disabled=not text.strip()):
        try:
            with st.spinner("Classifying and calculating SHAP signals..."):
                result = predict_one(model, text)
                result["top_signals"] = explain_text(model, text)
                save_prediction(
                    text,
                    result["verdict"],
                    result["confidence"],
                    model,
                    result["latency_ms"],
                )
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            return

        st.markdown(f"## {result['verdict']}")
        st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.1%}")
        st.caption(f"Model: {model} | Inference: {result['latency_ms']:.1f} ms")
        risk = "High" if result["verdict"] == "FAKE" else "Lower"
        st.metric("Risk level", risk)
        st.info("Recommendation: verify financial claims with official sources such as SEBI or RBI.")
        st.caption(
            "The Verify evidence cross-check is planned future work; this prediction "
            "does not independently verify sources or factual claims."
        )

        signals = result["top_signals"]
        st.subheader("SHAP word signals")
        if signals:
            st.markdown(
                '<div style="line-height:2.2">'
                + render_highlighted_text(text, signals)
                + "</div>",
                unsafe_allow_html=True,
            )
            chart_data = pd.DataFrame(signals)
            chart = px.bar(
                chart_data.sort_values("score"),
                x="score",
                y="word",
                color="toward",
                orientation="h",
                color_discrete_map={"FAKE": "#c62828", "REAL": "#2e7d32"},
                labels={"score": "SHAP score (+ FAKE / - REAL)", "word": "Token"},
            )
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("No non-zero word signals were returned by SHAP.")


if __name__ == "__main__":
    main()
