# TrustRAG

A Trustworthy Retrieval-Augmented Generation system for the **medical domain**, built on a hybrid dense+sparse retrieval pipeline with cross-encoder reranking and NLI-based faithfulness verification.

## Architecture

```
Query
  └─► Hybrid Retrieval (FAISS dense + BM25 sparse)
          └─► Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
                  └─► Generator (Ollama / gemma3:4b)
                          └─► NLI Faithfulness Checker (DeBERTa-v3)
                                  └─► Verified Answer  (or refusal if hallucinated)
```

**Data:** ~230K Wikipedia medicine passages (`wiki_snippets`, filtered by medical keywords)

**Indexes:**
- Dense: `intfloat/e5-small-v2` (384-dim) → FAISS `IndexFlatL2` (~354 MB)
- Sparse: BM25Okapi → `bm25.pkl` (~166 MB)

**Fusion:** MinMax-normalized scores with configurable `alpha` (dense) and `beta` (BM25), both default 0.5. BM25 scores < 0.01 penalized to -1.0.

**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — re-scores top candidates after fusion, lazy-loaded.

**Faithfulness checker:** `cross-encoder/nli-deberta-v3-base` — checks every sentence in the generated answer against the retrieved passages. Answers that fall below the entailment threshold are withheld and replaced with an explanatory refusal message.

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

Requires `ollama serve` and `ollama pull gemma3:4b` for generation.

---

## Interactive CLI — `main.py`

The main entry point. Ask medical questions interactively and get grounded, faithfulness-verified answers with per-query RAGAS metrics.

```bash
# Full pipeline — retrieval + reranking + generation + NLI faithfulness + RAGAS metrics
python main.py

# Retrieve more passages
python main.py --k 10

# Skip cross-encoder reranking (faster)
python main.py --no-rerank

# Skip RAGAS metrics (no Ollama evaluator needed)
python main.py --no-metrics

# Custom Ollama URL / model
python main.py --ollama-url http://localhost:11434 --ragas-model gemma3:4b
```

**Output per query:**
1. Retrieved passages table — rank, pid, score, article/section title, latency
2. Generated answer — or a faithfulness refusal if the answer is not grounded
3. NLI faithfulness log — per-claim entailment scores printed to stdout
4. RAGAS metrics — Faithfulness, Answer Relevancy, Context Precision for that query

---

## Dev / Debug Scripts

```bash
# Hybrid retrieval demo — prints before/after reranking for a sample query
python rag/hybrid_retriever.py

# Generator demo — single query end-to-end (includes faithfulness check)
python rag/generator.py

# Faithfulness checker demo — runs NLI on a hardcoded 3-sentence answer
# and prints per-claim verdicts so you can verify the model is working
python rag/faithfulness.py
```

---

## Evaluation

### Retrieval metrics (Recall, NDCG, MRR) — no API key needed

```bash
python eval/eval_retrieval.py --dev ./eval/dev.jsonl --k 10

# With reranking — reports both hybrid and reranked scores side-by-side
python eval/eval_retrieval.py --dev ./eval/dev.jsonl --k 10 --rerank
```

### RAGAS + NLI faithfulness metrics — requires Ollama

```bash
ollama serve && ollama pull gemma3:4b

# Default — NLI faithfulness check is ON alongside RAGAS metrics
python eval/eval_ragas.py --dev ./eval/dev.jsonl --k 5

# With cross-encoder reranking before generation
python eval/eval_ragas.py --dev ./eval/dev.jsonl --k 5 --rerank

# Answerable queries only
python eval/eval_ragas.py --dev ./eval/dev.jsonl --k 5 --answerable-only

# Disable NLI faithfulness check (faster, skips DeBERTa model)
python eval/eval_ragas.py --dev ./eval/dev.jsonl --k 5 --no-faithfulness
```

The results JSON includes `nli_faithfulness_avg` — the average NLI entailment score across all queries — alongside the standard RAGAS aggregate metrics.

---

## NLI Faithfulness Checker — `rag/faithfulness.py`

The faithfulness checker runs automatically after every generation step. It:

1. Splits the answer into sentences using `nltk.sent_tokenize`
2. For each sentence (claim), runs `cross-encoder/nli-deberta-v3-base` against every retrieved passage with `apply_softmax=True` to get entailment probabilities
3. Labels each claim `entailed` if its maximum entailment score across passages is ≥ `ENTAILMENT_MIN_SCORE` (0.5)
4. Computes an overall faithfulness score (fraction of entailed claims)
5. If the score is below `FAITHFULNESS_THRESHOLD` (0.7), the answer is **replaced with a refusal message** that lists the un-grounded sentences — the hallucinated answer is never returned to the user

**Refusal format example:**

```
[FAITHFULNESS WARNING] This answer was withheld because its faithfulness score (33.33%)
is below the required threshold (70%).

The following claims could not be verified against the retrieved passages:
  • It was first synthesized in ancient Egypt thousands of years ago.
  • Aspirin strengthens platelet aggregation and thereby reduces bleeding risk.
```

**Key constants** (in `rag/faithfulness.py`):

| Constant | Default | Effect |
|---|---|---|
| `FAITHFULNESS_THRESHOLD` | `0.7` | Minimum fraction of entailed claims to pass |
| `ENTAILMENT_MIN_SCORE` | `0.5` | Minimum per-claim entailment probability |
| `NLI_MODEL_NAME` | `cross-encoder/nli-deberta-v3-base` | NLI cross-encoder model |

---

## Results

### RAGAS — `k=2`, no reranking (baseline)

| Metric | Score |
|---|---|
| Faithfulness | 0.7144 |
| Answer Relevancy | 0.1631 |
| Context Precision | 0.5508 |

> Low answer relevancy at `k=2` is expected — too few passages for the generator to produce on-topic answers. Run with `--k 5` or higher and `--rerank` for better results.

---

## Key Parameters

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `alpha` | `hybrid_retriever.py` | 0.5 | Dense retrieval weight in score fusion |
| `beta` | `hybrid_retriever.py` | 0.5 | BM25 weight in score fusion |
| `dense_k` | `hybrid_retriever.py` | 100 | Dense candidates fetched before fusion |
| `bm25_k` | `hybrid_retriever.py` | 200 | BM25 candidates fetched before fusion |
| `reranker_top_k` | `retrieve_hybrid()` | 5 | Passages kept after reranking |
| `FAITHFULNESS_THRESHOLD` | `faithfulness.py` | 0.7 | Fraction of claims that must be entailed |
| `ENTAILMENT_MIN_SCORE` | `faithfulness.py` | 0.5 | Min entailment probability per claim |
