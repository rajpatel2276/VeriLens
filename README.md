# VeriLens — Dual-Pipeline Claim Credibility & Fact-Check Workbench

An end-to-end NLP triage system that evaluates the linguistic credibility of news headlines and factual claims by combining **statistical bag-of-words classification (TF-IDF)** with **dense neural semantic embeddings (`all-MiniLM-L6-v2`)** and **semantic precedent retrieval**.

## System Architecture
1. **Lexical Pipeline (TF-IDF + Logistic Regression):** Extracts 10,000 unigram/bigram features with balanced class weights. Provides sub-millisecond inference (`~0.02 ms/claim`) and transparent token-level attribution.
2. **Semantic Pipeline (`all-MiniLM-L6-v2` + Logistic Regression):** Encodes claims into 384-dimensional L2-normalized dense embeddings to capture contextual meaning and paraphrasing (`0.620 Macro-F1` on LIAR2).
3. **Semantic Precedent Retrieval:** Computes cosine similarity against 18,369 embedded LIAR2 statements to retrieve the top-3 closest historical fact-checks and their human justifications.
4. **Explainability Engine:** Combines rhetorical pattern analysis (sweeping absolutes, sensational framing, source attribution) with active TF-IDF weights to explain *why* a claim was flagged.

## Benchmark Results (LIAR2 Test Set — 2,296 Claims)
| Pipeline | Accuracy | Macro-F1 | Avg Latency |
| :--- | :---: | :---: | :---: |
| TF-IDF + Logistic Regression | 66.38% | 0.6176 | 0.020 ms |
| MiniLM-L6-v2 + Logistic Regression | 65.51% | 0.6199 | 4.122 ms |

## Quickstart
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn server:app --host 0.0.0.0 --port 8000


#### 2. Optional: `Dockerfile` (For Hugging Face Spaces or Local Container Demo)
If you want to host it live on **Hugging Face Spaces** (or run it as a clean container), create a `Dockerfile` in the root folder:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt fastapi uvicorn

COPY . .

# Pre-cache MiniLM weights inside the image so startup is instant
RUN python -c "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', device='cpu')"

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]# VERILENS
# random
# random
