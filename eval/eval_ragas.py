"""
RAGAS evaluation for TrustRAG.

Metrics (no ground-truth answers required):
  - Faithfulness               : is the answer grounded in the retrieved context?
  - AnswerRelevancy            : is the answer relevant to the question?
  - LLMContextPrecisionWithoutReference : are retrieved passages useful for answering?

Usage:
  python eval/eval_ragas.py --dev ./eval/dev.jsonl --k 5
  python eval/eval_ragas.py --dev ./eval/dev.jsonl --k 5 --answerable-only
"""

import os
import sys
import json
import argparse

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.hybrid_retriever import load_all, retrieve_hybrid
from rag.generator import generate_answer


def build_ragas_components(model: str, base_url: str):
    """Return (ragas_llm, ragas_embeddings) backed by a local Ollama model."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings

    llm = LangchainLLMWrapper(ChatOllama(model=model, base_url=base_url))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    )
    return llm, embeddings


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for TrustRAG")
    parser.add_argument("--dev", default="./eval/dev.jsonl", help="Dev set JSONL path")
    parser.add_argument("--k", type=int, default=5, help="Number of passages to retrieve")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    parser.add_argument(
        "--ragas-model", default="gemma3:4b", help="Ollama model used by RAGAS evaluators"
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama server base URL"
    )
    parser.add_argument("--out", default="./eval/results_ragas.json", help="Output JSON path")
    parser.add_argument(
        "--answerable-only",
        action="store_true",
        help="Skip unanswerable queries (gold_pids == [])",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking after hybrid retrieval.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Override the default cross-encoder model (optional).",
    )
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Disable NLI-based faithfulness checking (faster, no DeBERTa model needed).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dev):
        raise FileNotFoundError(f"Dev set not found: {args.dev}")

    # ── Load retrieval stack ──────────────────────────────────────────────────
    print("Loading retrieval indexes...")
    load_all(device=args.device)

    if args.rerank:
        from rag.reranker import load_reranker, _DEFAULT_MODEL
        model_name = args.reranker_model or _DEFAULT_MODEL
        load_reranker(model_name)

    # ── Load queries ──────────────────────────────────────────────────────────
    items = []
    with open(args.dev, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if args.answerable_only:
        items = [it for it in items if it.get("gold_pids")]
        print(f"Filtered to {len(items)} answerable queries.")

    # ── Retrieve + Generate ───────────────────────────────────────────────────
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    per_q_meta = []

    run_faithfulness = not args.no_faithfulness
    nli_scores: list = []

    print(f"Retrieving k={args.k} passages and generating answers for {len(items)} queries...")
    for it in tqdm(items):
        q = it["question"]
        if args.rerank:
            from rag.reranker import rerank as _rerank
            candidate_k = max(args.k, 20)
            hits = retrieve_hybrid(q, k=candidate_k)
            hits = _rerank(q, hits, top_k=args.k)
        else:
            hits = retrieve_hybrid(q, k=args.k)
        contexts = [h["passage_text"] for h in hits]
        context_str = "\n\n---\n\n".join(contexts)
        gen_result = generate_answer(q, context_str, passages=hits,
                                     run_faithfulness_check=run_faithfulness)
        answer = gen_result["answer"]
        faith = gen_result["faithfulness"]

        if faith is not None:
            nli_scores.append(faith["score"])
            print(
                f"  [NLI faithfulness] qid={it['qid']}  "
                f"score={faith['score']:.2%}  "
                f"({'PASS' if faith['faithful'] else 'FAIL'})"
            )

        samples.append(
            SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
            )
        )
        per_q_meta.append(
            {
                "qid": it["qid"],
                "question": q,
                "answer": answer,
                "retrieved_pids": [h["pid"] for h in hits],
                "nli_faithfulness": faith,
            }
        )

    dataset = EvaluationDataset(samples=samples)

    # ── Build RAGAS LLM / embeddings wrappers ─────────────────────────────────
    print("Initialising RAGAS evaluators (Ollama)...")
    ragas_llm, ragas_embeddings = build_ragas_components(args.ragas_model, args.ollama_url)

    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithoutReference,
    )
    from ragas import evaluate

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        LLMContextPrecisionWithoutReference(llm=ragas_llm),
    ]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Running RAGAS evaluation (local Ollama inference)...")
    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()

    for i, meta in enumerate(per_q_meta):
        df.loc[i, "qid"] = meta["qid"]
        df.loc[i, "answer"] = meta["answer"]
        df.loc[i, "retrieved_pids"] = str(meta["retrieved_pids"])

    metric_cols = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"]
    aggregate = {
        col: round(float(df[col].mean()), 4)
        for col in metric_cols
        if col in df.columns
    }

    nli_avg = round(sum(nli_scores) / len(nli_scores), 4) if nli_scores else None
    if nli_avg is not None:
        aggregate["nli_faithfulness_avg"] = nli_avg

    output = {
        "k": args.k,
        "rerank": args.rerank,
        "ragas_model": args.ragas_model,
        "ollama_url": args.ollama_url,
        "n_samples": len(samples),
        "answerable_only": args.answerable_only,
        "nli_faithfulness_avg": nli_avg,
        "aggregate": aggregate,
        "per_query": df.to_dict(orient="records"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.out}")
    print(f"\n{'Metric':<45} {'Score':>6}")
    print("-" * 52)
    for metric, val in aggregate.items():
        print(f"  {metric:<43} {val:.4f}")


if __name__ == "__main__":
    main()
