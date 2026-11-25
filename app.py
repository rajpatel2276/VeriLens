import os

# Keep WSL2 safe from CUDA/OpenMP bridge crashes
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="VeriLens | Claim Credibility Scorer",
    page_icon="🛡️",
    layout="wide",
)

# Generous spacing and clean typography
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    tfidf_pipe = joblib.load("artifacts/tfidf_pipeline.joblib")
    embed_clf = joblib.load("artifacts/embed_classifier.joblib")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    with open("artifacts/metrics.json", "r") as f:
        metrics = json.load(f)
    return tfidf_pipe, embed_clf, embedder, metrics


tfidf_pipe, embed_clf, embedder, metrics = load_models()


def score_claims(texts):
    t0 = time.perf_counter()
    tfidf_probs = tfidf_pipe.predict_proba(texts)[:, 1]
    tfidf_ms = ((time.perf_counter() - t0) / max(len(texts), 1)) * 1000

    t0 = time.perf_counter()
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    minilm_probs = embed_clf.predict_proba(embeddings)[:, 1]
    minilm_ms = ((time.perf_counter() - t0) / max(len(texts), 1)) * 1000

    return tfidf_probs, minilm_probs, tfidf_ms, minilm_ms


def get_top_words(text):
    vec = tfidf_pipe.named_steps["tfidf"]
    clf = tfidf_pipe.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())
    X_vec = vec.transform([text])
    idx = X_vec.nonzero()[1]
    if len(idx) == 0:
        return pd.DataFrame()

    weights = clf.coef_[0][idx] * X_vec.data
    df = pd.DataFrame(
        {
            "Keyword / Phrase": feature_names[idx],
            "Impact Score": np.round(weights, 3),
            "Signal": ["Supports Credible (+)" if w > 0 else "Flags Misleading (-)" for w in weights],
        }
    ).sort_values(by="Impact Score", key=abs, ascending=False)
    return df.head(8)


# ---------------------------------------------------------
# Clean, Minimal Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛡️ VeriLens")
    st.caption("AI/ML Claim Credibility Evaluator")
    st.divider()

    st.markdown("**Dataset Used**")
    st.write(f"LIAR2 Benchmark ({metrics['dataset_train_size']:,} train / {metrics['dataset_test_size']:,} test)")

    st.divider()
    st.markdown("**Test Set Performance**")
    st.metric("Semantic Model (MiniLM) F1", f"{metrics['minilm']['macro_f1']:.3f}")
    st.metric("Keyword Model (TF-IDF) F1", f"{metrics['tfidf']['macro_f1']:.3f}")

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("Claim & News Credibility Scorer")
st.write("Analyze any factual claim or news headline using two NLP models: **Keyword Matching (TF-IDF)** and **Neural Meaning Embeddings (MiniLM)**.")
st.write("")

tab_analyze, tab_batch, tab_compare = st.tabs(
    ["Single Claim Check", "Batch CSV Check", "Model Benchmark"]
)

# =========================================================
# TAB 1: SINGLE CLAIM CHECK
# =========================================================
with tab_analyze:
    st.write("")
    presets = {
        "Select a sample claim (optional)...": "",
        "1. Economic Statistics Claim": "The unemployment rate in the United States dropped below 4 percent for the longest streak since the 1960s.",
        "2. Viral Health Hoax": "Drinking boiled lemon water every morning completely cures all forms of diabetes within two weeks.",
        "3. Public Policy Claim": "Federal spending on public education accounts for less than 10 percent of total K-12 school funding.",
    }

    choice = st.selectbox("Quick Examples", list(presets.keys()), label_visibility="collapsed")

    claim_text = st.text_area(
        "Enter claim or headline",
        value=presets[choice],
        height=110,
        placeholder="Type or paste a news headline or factual statement here...",
        label_visibility="collapsed",
    )

    with st.expander("⚙️ Calibration Settings (Optional)"):
        s_col1, s_col2 = st.columns(2, gap="large")
        with s_col1:
            threshold = st.slider("Credibility Cutoff Threshold", 0.30, 0.70, 0.50, 0.05)
        with s_col2:
            semantic_weight = st.slider("Semantic Model Weight", 0.0, 1.0, 0.70, 0.10)
        lexical_weight = round(1.0 - semantic_weight, 2)

    st.write("")
    analyze_clicked = st.button("Check Credibility", type="primary")

    if analyze_clicked:
        if not claim_text.strip():
            st.warning("Please enter a claim or pick an example from the dropdown.")
        else:
            p_tf_arr, p_ml_arr, ms_tf, ms_ml = score_claims([claim_text])
            p_tf = float(p_tf_arr[0])
            p_ml = float(p_ml_arr[0])
            p_final = (semantic_weight * p_ml) + (lexical_weight * p_tf)
            is_credible = p_final >= threshold

            st.divider()

            # 1. Clean Top Verdict Banner
            if is_credible:
                st.success(f"### Verdict: Likely Credible ({p_final * 100:.1f}% Score)")
            else:
                st.error(f"### Verdict: Low Credibility / Misleading ({p_final * 100:.1f}% Score)")

            if (p_tf >= threshold) != (p_ml >= threshold):
                st.info("Note: The Keyword model and Semantic model disagreed on this claim. Inspect the breakdown below.")

            st.write("")

            # 2. Two Spacious Model Cards
            col1, col2 = st.columns(2, gap="large")

            with col1:
                with st.container(border=True):
                    st.subheader("1. Keyword Model (TF-IDF)")
                    st.caption("Checks specific words and phrases")
                    st.write("")
                    st.metric("Credibility Score", f"{p_tf * 100:.1f}%")
                    st.progress(p_tf)
                    st.caption(f"Speed: {ms_tf:.2f} ms")

            with col2:
                with st.container(border=True):
                    st.subheader("2. Meaning Model (MiniLM)")
                    st.caption("Understands overall sentence context")
                    st.write("")
                    st.metric("Credibility Score", f"{p_ml * 100:.1f}%")
                    st.progress(p_ml)
                    st.caption(f"Speed: {ms_ml:.2f} ms")

            st.write("")

            # 3. Collapsible Word Breakdown
            with st.expander("🔍 See Which Words Influenced the Keyword Model"):
                words_df = get_top_words(claim_text)
                if not words_df.empty:
                    st.dataframe(words_df, hide_index=True, width="stretch")
                else:
                    st.write("None of the words in this claim matched the keyword vocabulary—the Semantic Model carried this prediction.")

# =========================================================
# TAB 2: BATCH CSV CHECK
# =========================================================
with tab_batch:
    st.write("")
    st.subheader("Evaluate Multiple Claims at Once")
    st.write("Paste several claims (one per line) or upload a CSV file to download scored results.")
    st.write("")

    mode = st.radio("Choose input method:", ["Paste Text Lines", "Upload CSV"], horizontal=True)
    lines = []

    if mode == "Paste Text Lines":
        raw = st.text_area(
            "Claims list",
            value=(
                "NASA confirmed that Earth will experience 15 days of complete darkness next month.\n"
                "India's renewable energy capacity crossed 200 gigawatts according to ministry data.\n"
                "5G cell towers directly spread airborne viruses across major cities."
            ),
            height=140,
            label_visibility="collapsed",
        )
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df_in = pd.read_csv(file)
            col = st.selectbox("Select text column:", df_in.columns)
            lines = df_in[col].dropna().astype(str).tolist()

    if st.button("Run Batch Scoring", type="primary"):
        if not lines:
            st.warning("Please provide at least one claim.")
        else:
            p_tf, p_ml, _, _ = score_claims(lines)
            p_ens = (0.7 * p_ml) + (0.3 * p_tf)
            out_df = pd.DataFrame(
                {
                    "Claim": lines,
                    "Keyword Score (%)": np.round(p_tf * 100, 1),
                    "Meaning Score (%)": np.round(p_ml * 100, 1),
                    "Final Score (%)": np.round(p_ens * 100, 1),
                    "Verdict": ["Likely Credible" if p >= 0.5 else "Low Credibility" for p in p_ens],
                }
            )
            st.write("")
            st.dataframe(out_df, hide_index=True, width="stretch")
            st.download_button(
                "Download Results CSV",
                out_df.to_csv(index=False).encode("utf-8"),
                "credibility_results.csv",
                "text/csv",
            )

# =========================================================
# TAB 3: MODEL BENCHMARK
# =========================================================
with tab_compare:
    st.write("")
    st.subheader("Why Compare Two Models? (Viva Summary)")
    st.write("")

    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        with st.container(border=True):
            st.metric("TF-IDF Accuracy", f"{metrics['tfidf']['accuracy']*100:.1f}%")
            st.caption(f"Latency: {metrics['tfidf']['latency_ms']:.3f} ms/claim")
    with m2:
        with st.container(border=True):
            st.metric("MiniLM Accuracy", f"{metrics['minilm']['accuracy']*100:.1f}%")
            st.caption(f"Latency: {metrics['minilm']['latency_ms']:.2f} ms/claim")
    with m3:
        with st.container(border=True):
            st.metric("Speed Advantage", "~200x Faster")
            st.caption("TF-IDF vs. Neural Transformer")

    st.write("")
    chart_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Macro-F1"],
            "Keyword Model (TF-IDF)": [metrics["tfidf"]["accuracy"], metrics["tfidf"]["macro_f1"]],
            "Meaning Model (MiniLM)": [metrics["minilm"]["accuracy"], metrics["minilm"]["macro_f1"]],
        }
    ).set_index("Metric")
    st.bar_chart(chart_df, stack=False)