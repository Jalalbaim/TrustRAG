"""
TrustRAG interactive CLI.

Ask any medical question and get an answer grounded in Wikipedia medicine passages,
with retrieval details and RAGAS quality metrics shown for every query.

Usage:
    python main.py                        # default: k=5, reranking on, metrics on
    python main.py --k 10                 # retrieve more passages
    python main.py --no-rerank            # skip cross-encoder reranking
    python main.py --no-metrics           # skip RAGAS metrics (faster)
"""

import argparse
import sys
import os
import time

# ── make project root importable regardless of cwd ────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag.hybrid_retriever import load_all, retrieve_hybrid
from rag.generator import generate_answer


# ── RAGAS single-sample evaluation ────────────────────────────────────────────

def _build_ragas_components(ollama_model: str, ollama_url: str):
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings

    llm = LangchainLLMWrapper(ChatOllama(model=ollama_model, base_url=ollama_url))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    )
    return llm, embeddings


def _compute_ragas(query: str, answer: str, contexts: list, llm, embeddings) -> dict:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithoutReference,
    )

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
        )
    ])
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
    ]
    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()
    return {
        "faithfulness":                          round(float(df["faithfulness"].iloc[0]), 4),
        "answer_relevancy":                      round(float(df["answer_relevancy"].iloc[0]), 4),
        "context_precision":                     round(float(df["llm_context_precision_without_reference"].iloc[0]), 4),
    }


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_passages(hits: list, use_rerank: bool) -> None:
    score_field = "reranker_score" if use_rerank else "score"
    score_label = "reranker" if use_rerank else "hybrid "
    print(f"\n  {'#':<3} {'pid':<10} {score_label+' score':>14}  article / section")
    print("  " + "-" * 72)
    for i, h in enumerate(hits, 1):
        score = h.get(score_field, h.get("score", 0.0))
        title = h["article_title"]
        section = h["section_title"]
        label = f"{title} — {section}"
        if len(label) > 52:
            label = label[:49] + "..."
        print(f"  #{i:<2} {str(h['pid']):<10} {score:>14.4f}  {label}")


def _print_metrics(m: dict) -> None:
    print()
    print(f"  {'Metric':<46} {'Score':>6}")
    print("  " + "-" * 54)
    print(f"  {'Faithfulness':<46} {m['faithfulness']:>6.4f}")
    print(f"  {'Answer Relevancy':<46} {m['answer_relevancy']:>6.4f}")
    print(f"  {'Context Precision':<46} {m['context_precision']:>6.4f}")


def _hr(char="─", width=72):
    print("  " + char * width)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrustRAG interactive CLI")
    parser.add_argument("--k",            type=int,  default=5,
                        help="Passages to retrieve (default 5)")
    parser.add_argument("--device",       type=str,  default="cpu",
                        help="Device for embedding model (default cpu)")
    parser.add_argument("--no-rerank",    action="store_true",
                        help="Disable cross-encoder reranking")
    parser.add_argument("--no-metrics",   action="store_true",
                        help="Skip RAGAS metric computation (faster)")
    parser.add_argument("--ragas-model",  type=str,  default="gemma3:4b",
                        help="Ollama model used by RAGAS evaluators")
    parser.add_argument("--ollama-url",   type=str,  default="http://localhost:11434",
                        help="Ollama server URL")
    args = parser.parse_args()

    use_rerank   = not args.no_rerank
    use_metrics  = not args.no_metrics
    candidate_k  = max(args.k, 20) if use_rerank else args.k

    # ── Startup ────────────────────────────────────────────────────────────────
    print("\n  TrustRAG — Medical Q&A")
    _hr("═")
    print(f"  retrieval  : hybrid (FAISS + BM25), k={args.k}")
    print(f"  reranking  : {'on  (cross-encoder/ms-marco-MiniLM-L-6-v2)' if use_rerank else 'off'}")
    print(f"  metrics    : {'on  (RAGAS via Ollama ' + args.ragas_model + ')' if use_metrics else 'off'}")
    _hr("═")
    print("  Loading indexes…", end=" ", flush=True)
    load_all(device=args.device)
    print("done.")

    if use_rerank:
        from rag.reranker import load_reranker
        print("  Loading reranker…", end=" ", flush=True)
        load_reranker()
        print("done.")

    ragas_llm = ragas_emb = None
    if use_metrics:
        print("  Loading RAGAS evaluators…", end=" ", flush=True)
        try:
            ragas_llm, ragas_emb = _build_ragas_components(args.ragas_model, args.ollama_url)
            print("done.")
        except Exception as e:
            print(f"failed ({e})\n  Metrics will be skipped.")
            use_metrics = False

    print("\n  Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    # ── Query loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            query = input("  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("\n  Goodbye.")
            break

        print()
        _hr()

        # ── Retrieval ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if use_rerank:
            from rag.reranker import rerank as _rerank
            hits = retrieve_hybrid(query, k=candidate_k)
            hits = _rerank(query, hits, top_k=args.k)
        else:
            hits = retrieve_hybrid(query, k=args.k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        print(f"  Retrieved {len(hits)} passage(s)  [{retrieval_ms:.0f} ms]")
        _print_passages(hits, use_rerank)

        # ── Generation ─────────────────────────────────────────────────────────
        print()
        context_str = "\n\n---\n\n".join(h["passage_text"] for h in hits)
        t1 = time.perf_counter()
        gen_result = generate_answer(query, context_str, passages=hits)
        gen_ms = (time.perf_counter() - t1) * 1000
        answer = gen_result["answer"]

        print(f"  Answer  [{gen_ms:.0f} ms]")
        _hr("·")
        # indent answer lines
        for line in answer.strip().splitlines():
            print(f"  {line}")
        _hr("·")

        # ── RAGAS metrics ──────────────────────────────────────────────────────
        if use_metrics:
            print(f"\n  Computing RAGAS metrics…", end=" ", flush=True)
            t2 = time.perf_counter()
            try:
                contexts = [h["passage_text"] for h in hits]
                metrics = _compute_ragas(query, answer, contexts, ragas_llm, ragas_emb)
                ragas_ms = (time.perf_counter() - t2) * 1000
                print(f"done  [{ragas_ms:.0f} ms]")
                _print_metrics(metrics)
            except Exception as e:
                print(f"failed ({e})")

        _hr()
        print()


if __name__ == "__main__":
    main()
