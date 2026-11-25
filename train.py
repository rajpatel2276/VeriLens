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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from datasets import load_dataset
import torch
from sentence_transformers import SentenceTransformer

os.makedirs("artifacts", exist_ok=True)

LABEL_NAMES = {
    0: "Pants on Fire (Completely False)",
    1: "False",
    2: "Barely True",
    3: "Half True (Missing Context)",
    4: "Mostly True",
    5: "True",
}


def load_liar_binary():
    print("Loading LIAR2 dataset from Hugging Face...")
    ds = load_dataset("chengxuphd/liar2")
    cols = ds["train"].column_names

    def to_df(split):
        raw_df = ds[split].to_pandas()
        df = pd.DataFrame()
        df["text"] = raw_df["statement"].astype(str)
        df["raw_label"] = raw_df["label"].astype(int)
        df["label"] = df["raw_label"].apply(lambda x: 1 if x in [4, 5] else 0)
        df["subject"] = raw_df["subject"].fillna("General").astype(str) if "subject" in cols else "General"
        df["speaker"] = raw_df["speaker"].fillna("Unknown").astype(str) if "speaker" in cols else "Unknown"
        df["justification"] = (
            raw_df["justification"].fillna("").astype(str) if "justification" in cols else ""
        )
        return df[df["text"].str.strip().astype(bool)].reset_index(drop=True)

    train_df = to_df("train")
    test_df = to_df("test")
    print(f"Loaded {len(train_df)} train samples and {len(test_df)} test samples.")
    return train_df, test_df


def main():
    train_df, test_df = load_liar_binary()
    X_train, y_train = train_df["text"].tolist(), train_df["label"].values
    X_test, y_test = test_df["text"].tolist(), test_df["label"].values

    # 1. Pipeline A: TF-IDF + Logistic Regression
    print("\n[1/2] Training Pipeline A: TF-IDF + Logistic Regression...")
    t0 = time.perf_counter()
    tfidf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    tfidf_pipe.fit(X_train, y_train)
    tfidf_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_tfidf = tfidf_pipe.predict(X_test)
    tfidf_infer_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000

    tfidf_acc = accuracy_score(y_test, y_pred_tfidf)
    tfidf_f1 = f1_score(y_test, y_pred_tfidf, average="macro")
    print(f"TF-IDF -> Accuracy: {tfidf_acc:.4f} | Macro-F1: {tfidf_f1:.4f} | Latency: {tfidf_infer_ms:.3f} ms")

    # 2. Pipeline B: MiniLM Dense Embeddings + Logistic Regression
    print("\n[2/2] Encoding texts with all-MiniLM-L6-v2 on CPU...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    t0 = time.perf_counter()
    X_train_emb = embedder.encode(X_train, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    embed_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    embed_clf.fit(X_train_emb, y_train)
    embed_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_test_emb = embedder.encode(X_test, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    y_pred_embed = embed_clf.predict(X_test_emb)
    embed_infer_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000

    embed_acc = accuracy_score(y_test, y_pred_embed)
    embed_f1 = f1_score(y_test, y_pred_embed, average="macro")
    print(f"MiniLM -> Accuracy: {embed_acc:.4f} | Macro-F1: {embed_f1:.4f} | Latency: {embed_infer_ms:.3f} ms")

    # Save models, metrics, and semantic reference index
    joblib.dump(tfidf_pipe, "artifacts/tfidf_pipeline.joblib")
    joblib.dump(embed_clf, "artifacts/embed_classifier.joblib")
    np.save("artifacts/ref_embeddings.npy", X_train_emb.astype(np.float32))

    ref_records = []
    for _, row in train_df.iterrows():
        ref_records.append({
            "text": row["text"],
            "rating": LABEL_NAMES.get(int(row["raw_label"]), "Unknown"),
            "is_credible": bool(row["label"] == 1),
            "subject": row["subject"],
            "speaker": row["speaker"],
            "justification": row["justification"],
        })

    with open("artifacts/ref_records.json", "w") as f:
        json.dump(ref_records, f)

    metrics = {
        "dataset_train_size": len(X_train),
        "dataset_test_size": len(X_test),
        "tfidf": {
            "accuracy": round(float(tfidf_acc), 4),
            "macro_f1": round(float(tfidf_f1), 4),
            "train_sec": round(float(tfidf_train_time), 2),
            "latency_ms": round(float(tfidf_infer_ms), 3),
        },
        "minilm": {
            "accuracy": round(float(embed_acc), 4),
            "macro_f1": round(float(embed_f1), 4),
            "train_sec": round(float(embed_train_time), 2),
            "latency_ms": round(float(embed_infer_ms), 3),
        },
    }
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll models and fact-check reference index saved to ./artifacts/!")


if __name__ == "__main__":
    main()