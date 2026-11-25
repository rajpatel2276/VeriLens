import os

# Keep WSL2 safe from CUDA/OpenMP bridge crashes
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import time
from typing import List
import joblib
import numpy as np
import sklearn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer

app = FastAPI(title="VeriLens API")

# Load artifacts once at startup
tfidf_pipe = joblib.load("artifacts/tfidf_pipeline.joblib")
embed_clf = joblib.load("artifacts/embed_classifier.joblib")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ref_embeddings = np.load("artifacts/ref_embeddings.npy")

with open("artifacts/ref_records.json", "r") as f:
    ref_records = json.load(f)

with open("artifacts/metrics.json", "r") as f:
    metrics_data = json.load(f)


class AnalyzeRequest(BaseModel):
    claim: str
    threshold: float = 0.50
    semantic_weight: float = 0.70


class BatchRequest(BaseModel):
    claims: List[str]
    threshold: float = 0.50
    semantic_weight: float = 0.70


def build_explanation(claim: str, p_tfidf: float, p_minilm: float, p_final: float, threshold: float, top_terms: list, precedents: list):
    """Generate clear, human-readable reasoning on why the claim is flagged and what the actual point/context is."""
    lower = claim.lower()
    reasons = []
    takeaway = ""

    # 1. Linguistic & Rhetorical Pattern Checks
    absolutes = re.findall(r"\b(all|every|completely|always|never|100%|zero|cures|guaranteed|proven|secret|hoax)\b", lower)
    has_numbers = bool(re.search(r"\b(\d+(\.\d+)?%|\d+\s*(percent|million|billion|gigawatts|years|days|times))\b", lower))
    has_attribution = bool(re.search(r"\b(according to|data|report|study|ministry|department|bureau|survey|rate)\b", lower))
    sensational = re.findall(r"\b(miracle|shocking|banned|cover-up|exposed|darkness|viruses|destroy|instant)\b", lower)

    if absolutes:
        uniq = ", ".join(sorted(set(absolutes)))
        reasons.append({
            "type": "warning",
            "title": "Absolute or Extreme Framing",
            "detail": f"Uses sweeping terms ({uniq}). Real-world scientific and policy facts rarely operate in 100% absolutes, which is a strong marker of exaggerated or false claims."
        })

    if sensational:
        uniq = ", ".join(sorted(set(sensational)))
        reasons.append({
            "type": "warning",
            "title": "Sensational / Clickbait Vocabulary",
            "detail": f"Contains emotionally charged keywords ({uniq}) frequently associated with viral misinformation rather than neutral reporting."
        })

    if has_numbers and has_attribution:
        reasons.append({
            "type": "positive",
            "title": "Quantified & Attributed Framing",
            "detail": "Combines specific numerical metrics with institutional or data attribution—a structural pattern common in verifiable factual statements."
        })
    elif has_numbers and not has_attribution:
        reasons.append({
            "type": "neutral",
            "title": "Unattributed Statistic",
            "detail": "Cites specific numbers without naming a primary source or timeframe, making it vulnerable to cherry-picking or missing context."
        })

    # 2. Model Signal Explanation
    if abs(p_minilm - p_tfidf) >= 0.22:
        reasons.append({
            "type": "neutral",
            "title": "Lexical vs. Semantic Divergence",
            "detail": (
                f"The Keyword model scored {p_tfidf*100:.1f}% while the Meaning model scored {p_minilm*100:.1f}%. "
                "This happens when individual words look familiar to statistical filters, but the overall sentence meaning tells a different story."
            )
        })

    if top_terms:
        neg_terms = [t["term"] for t in top_terms if t["weight"] < -0.08][:3]
        pos_terms = [t["term"] for t in top_terms if t["weight"] > 0.08][:3]
        if neg_terms:
            reasons.append({
                "type": "warning",
                "title": "Historical Misinformation Triggers",
                "detail": f"Phrases like {', '.join(repr(x) for x in neg_terms)} statistically correlate with misleading or false claims in the LIAR2 dataset."
            })
        if pos_terms:
            reasons.append({
                "type": "positive",
                "title": "Credible Reporting Markers",
                "detail": f"Terms like {', '.join(repr(x) for x in pos_terms)} align with verified statements in past fact-checking records."
            })

    if not reasons:
        reasons.append({
            "type": "neutral",
            "title": "Semantic Embedding Classification",
            "detail": "No extreme clickbait triggers were detected; the score is driven primarily by how closely the sentence's 384-D meaning vector matches verified vs. false claims."
        })

    # 3. What is the actual point / reality check?
    top_match = precedents[0] if precedents else None
    if p_final >= threshold:
        takeaway = (
            "This claim is structured like a verifiable factual statement rather than an exaggerated rumor. "
            "However, numbers and policy thresholds can still be selectively framed—verify the exact date and primary dataset before citing."
        )
    else:
        takeaway = (
            "Claims like this are typically misleading because they oversimplify complex causality, omit critical context, "
            "or state unproven outcomes as settled fact. Always check primary scientific or official statistical sources rather than viral summaries."
        )

    if top_match and top_match["similarity"] >= 0.48 and top_match["justification"]:
        takeaway += f" Specifically, the closest historical fact-check on this topic ({top_match['rating']}) noted: \"{top_match['justification']}\""

    return {"reasons": reasons, "actual_point": takeaway}


@app.get("/api/metrics")
def get_metrics():
    return metrics_data


@app.post("/api/analyze")
def analyze_claim(req: AnalyzeRequest):
    claim = req.claim.strip()

    # Pipeline A: TF-IDF
    t0 = time.perf_counter()
    p_tfidf = float(tfidf_pipe.predict_proba([claim])[0][1])
    tfidf_ms = round((time.perf_counter() - t0) * 1000, 3)

    # Pipeline B: MiniLM
    t0 = time.perf_counter()
    emb = embedder.encode([claim], normalize_embeddings=True)[0]
    p_minilm = float(embed_clf.predict_proba([emb])[0][1])
    minilm_ms = round((time.perf_counter() - t0) * 1000, 2)

    lexical_weight = 1.0 - req.semantic_weight
    p_final = (req.semantic_weight * p_minilm) + (lexical_weight * p_tfidf)

    # Extract top active TF-IDF terms
    vec = tfidf_pipe.named_steps["tfidf"]
    clf = tfidf_pipe.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())
    X_vec = vec.transform([claim])
    idx = X_vec.nonzero()[1]
    top_terms = []
    if len(idx) > 0:
        weights = clf.coef_[0][idx] * X_vec.data
        order = np.argsort(np.abs(weights))[::-1][:8]
        for i in order:
            top_terms.append({
                "term": str(feature_names[idx[i]]),
                "weight": round(float(weights[i]), 3),
            })

    # Retrieve top 3 closest fact-checked precedents via dot product (cosine similarity)
    sims = np.dot(ref_embeddings, emb.astype(np.float32))
    top_k_idx = np.argsort(sims)[::-1][:3]
    precedents = []
    for k in top_k_idx:
        rec = ref_records[int(k)]
        precedents.append({
            "text": rec["text"],
            "rating": rec["rating"],
            "is_credible": rec["is_credible"],
            "subject": rec["subject"],
            "speaker": rec["speaker"],
            "justification": rec["justification"],
            "similarity": round(float(sims[k]), 3),
        })

    explanation = build_explanation(
        claim, p_tfidf, p_minilm, p_final, req.threshold, top_terms, precedents
    )

    return {
        "claim": claim,
        "verdict": "Likely Credible" if p_final >= req.threshold else "Low Credibility / Misleading",
        "is_credible": bool(p_final >= req.threshold),
        "ensemble_score": round(p_final * 100, 1),
        "tfidf": {
            "score": round(p_tfidf * 100, 1),
            "is_credible": bool(p_tfidf >= req.threshold),
            "latency_ms": tfidf_ms,
        },
        "minilm": {
            "score": round(p_minilm * 100, 1),
            "is_credible": bool(p_minilm >= req.threshold),
            "latency_ms": minilm_ms,
        },
        "models_agree": bool((p_tfidf >= req.threshold) == (p_minilm >= req.threshold)),
        "top_terms": top_terms,
        "precedents": precedents,
        "explanation": explanation,
    }


@app.post("/api/batch")
def analyze_batch(req: BatchRequest):
    claims = [c.strip() for c in req.claims if c.strip()]
    if not claims:
        return {"results": []}

    p_tf = tfidf_pipe.predict_proba(claims)[:, 1]
    embs = embedder.encode(claims, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    p_ml = embed_clf.predict_proba(embs)[:, 1]
    lexical_weight = 1.0 - req.semantic_weight
    p_ens = (req.semantic_weight * p_ml) + (lexical_weight * p_tf)

    results = []
    for c, tf_s, ml_s, en_s in zip(claims, p_tf, p_ml, p_ens):
        results.append({
            "claim": c,
            "tfidf_score": round(float(tf_s) * 100, 1),
            "minilm_score": round(float(ml_s) * 100, 1),
            "final_score": round(float(en_s) * 100, 1),
            "verdict": "Likely Credible" if en_s >= req.threshold else "Low Credibility",
            "is_credible": bool(en_s >= req.threshold),
        })
    return {"results": results}


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_index():
    return FileResponse("static/index.html")