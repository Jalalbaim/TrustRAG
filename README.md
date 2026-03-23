# TrustRAG

A Trustworthy Retrieval-Augmented Generation system for the **medical domain**, built on a hybrid dense+sparse retrieval pipeline with cross-encoder reranking.

## Architecture

```
Query
  └─► Hybrid Retrieval (FAISS dense + BM25 sparse)
          └─► Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
                  └─► Generator (Ollama / gemma3:4b)
                          └─► Answer
```

**Data:** ~230K Wikipedia medicine passages (`wiki_snippets`, filtered by medical keywords)

**Indexes:**
- Dense: `intfloat/e5-small-v2` (384-dim) → FAISS `IndexFlatL2` (~354 MB)
- Sparse: BM25Okapi → `bm25.pkl` (~166 MB)

**Fusion:** MinMax-normalized scores with configurable `alpha` (dense) and `beta` (BM25), both default 0.5. BM25 scores < 0.01 penalized to -1.0.

**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — re-scores top candidates after fusion, lazy-loaded.

---

## Setup

```bash
pip install -r requirements.txt
```

Build indexes (run once):

```bash
python data/processing.py
python index/build_index.py
python index/build_bm25.py
```

---

## Usage

```bash
# Hybrid retrieval demo (with before/after reranking output)
python rag/hybrid_retriever.py

# Full RAG pipeline (retrieval + reranking + generation)
python rag/generator.py
```

---

## Evaluation

### Retrieval metrics (Recall, NDCG, MRR) — no API key needed

```bash
python eval/eval_retrieval.py --dev ./eval/dev.jsonl --k 10

# With reranking — reports both hybrid and reranked scores side-by-side
python eval/eval_retrieval.py --dev ./eval/dev.jsonl --k 10 --rerank
```

### RAGAS metrics (Faithfulness, AnswerRelevancy, ContextPrecision) — requires Ollama

```bash
ollama serve && ollama pull gemma3:4b

python eval/eval_ragas.py --dev ./eval/dev.jsonl --k 5
python eval/eval_ragas.py --dev ./eval/dev.jsonl --k 5 --rerank
python eval/eval_ragas.py --dev ./eval/dev.jsonl --k 5 --answerable-only
```

---

## Results

### RAGAS — `k=2`, no reranking (baseline)

| Metric | Score |
|---|---|
| Faithfulness | 0.7144 |
| Answer Relevancy | 0.1631 |
| Context Precision | 0.5508 |

> Low answer relevancy at `k=2` is expected — too few passages for the generator to produce on-target answers. Run with `--k 5` or higher and `--rerank` for better results.

---

## Key Parameters

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `alpha` | `hybrid_retriever.py` | 0.5 | Dense retrieval weight |
| `beta` | `hybrid_retriever.py` | 0.5 | BM25 weight |
| `dense_k` | `hybrid_retriever.py` | 100 | Dense candidates before fusion |
| `bm25_k` | `hybrid_retriever.py` | 200 | BM25 candidates before fusion |
| `reranker_top_k` | `retrieve_hybrid()` | 5 | Passages kept after reranking |
| `_DEFAULT_MODEL` | `reranker.py` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
