import os
import json
import math
import argparse
import sys
from typing import List, Any

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.hybrid_retriever import load_all, retrieve_hybrid


# python eval/eval_retrieval.py --dev ./eval/dev.jsonl --k 10


def dcg(relevances: List[int]) -> float:
    s = 0.0
    for i, rel in enumerate(relevances, start=1):
        if rel:
            s += 1.0 / math.log2(i + 1)
    return s


def ndcg_at_k(ranked_pids: List[Any], gold_set: set, k: int) -> float:
    top = ranked_pids[:k]
    rel = [1 if pid in gold_set else 0 for pid in top]
    dcg_val = dcg(rel)

    ideal_rel = [1] * min(len(gold_set), k) + [0] * max(0, k - min(len(gold_set), k))
    idcg = dcg(ideal_rel)
    if idcg == 0.0:
        return 0.0
    return dcg_val / idcg


def recall_at_k(ranked_pids: List[Any], gold_set: set, k: int) -> float:
    if len(gold_set) == 0:
        return 1.0 
    top = set(ranked_pids[:k])
    return len(top.intersection(gold_set)) / float(len(gold_set))


def mrr_at_k(ranked_pids: List[Any], gold_set: set, k: int) -> float:
    if len(gold_set) == 0:
        return 1.0
    for i, pid in enumerate(ranked_pids[:k], start=1):
        if pid in gold_set:
            return 1.0 / float(i)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=str, default="./eval/dev.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="./eval/results_retrieval.json")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking; reports metrics both with and without reranking.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Override the default cross-encoder model (optional).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dev):
        raise FileNotFoundError(f"Missing dev set: {args.dev}")

    load_all(device=args.device)
    print("Loaded all models")

    if args.rerank:
        from rag.reranker import load_reranker, _DEFAULT_MODEL
        model_name = args.reranker_model or _DEFAULT_MODEL
        load_reranker(model_name)

    items = []
    with open(args.dev, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    k = int(args.k)

    total = 0
    ans_count = 0
    unans_count = 0

    recall_sum = 0.0
    ndcg_sum = 0.0
    mrr_sum = 0.0

    # rerank accumulators (only used when --rerank is set)
    rr_recall_sum = 0.0
    rr_ndcg_sum = 0.0
    rr_mrr_sum = 0.0

    per_q = []

    print("Processing items...")
    for it in tqdm(items):
        qid = it["qid"]
        q = it["question"]
        gold = it.get("gold_pids", [])
        gold_set = set(gold)

        # Always retrieve a larger candidate set so the reranker has enough to work with
        candidate_k = max(k, 20) if args.rerank else k
        hits = retrieve_hybrid(q, k=candidate_k, dense_k=50, bm25_k=50)
        ranked_pids = [int(h["pid"]) if hasattr(h["pid"], 'item') else h["pid"] for h in hits]

        r = recall_at_k(ranked_pids, gold_set, k)
        n = ndcg_at_k(ranked_pids, gold_set, k)
        m = mrr_at_k(ranked_pids, gold_set, k)

        total += 1
        recall_sum += r
        ndcg_sum += n
        mrr_sum += m

        if len(gold_set) == 0:
            unans_count += 1
        else:
            ans_count += 1

        entry: dict = {
            "qid": qid,
            "recall@k": r,
            "ndcg@k": n,
            "mrr@k": m,
            "gold_size": len(gold_set),
            "top_pids": ranked_pids,
        }

        if args.rerank:
            from rag.reranker import rerank as _rerank
            rr_hits = _rerank(q, hits, top_k=k)
            rr_pids = [int(h["pid"]) if hasattr(h["pid"], 'item') else h["pid"] for h in rr_hits]

            rr_r = recall_at_k(rr_pids, gold_set, k)
            rr_n = ndcg_at_k(rr_pids, gold_set, k)
            rr_m = mrr_at_k(rr_pids, gold_set, k)

            rr_recall_sum += rr_r
            rr_ndcg_sum += rr_n
            rr_mrr_sum += rr_m

            entry["reranked_recall@k"] = rr_r
            entry["reranked_ndcg@k"] = rr_n
            entry["reranked_mrr@k"] = rr_m
            entry["reranked_pids"] = rr_pids

        per_q.append(entry)

    results = {
        "k": k,
        "rerank": args.rerank,
        "n_total": total,
        "n_answerable": ans_count,
        "n_unanswerable": unans_count,
        "mean_recall@k": recall_sum / max(total, 1),
        "mean_ndcg@k": ndcg_sum / max(total, 1),
        "mean_mrr@k": mrr_sum / max(total, 1),
        "per_query": per_q,
    }

    if args.rerank:
        results["reranked_mean_recall@k"] = rr_recall_sum / max(total, 1)
        results["reranked_mean_ndcg@k"] = rr_ndcg_sum / max(total, 1)
        results["reranked_mean_mrr@k"] = rr_mrr_sum / max(total, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.out}")
    print(f"\n{'Metric':<12} {'Hybrid':>8}{'Reranked':>10}")
    print("-" * 30)
    rr_r_str = f"{results['reranked_mean_recall@k']:.4f}" if args.rerank else "  n/a"
    rr_n_str = f"{results['reranked_mean_ndcg@k']:.4f}" if args.rerank else "  n/a"
    rr_m_str = f"{results['reranked_mean_mrr@k']:.4f}" if args.rerank else "  n/a"
    print(f"Recall@{k:<5} {results['mean_recall@k']:>8.4f}{rr_r_str:>10}")
    print(f"NDCG@{k:<7} {results['mean_ndcg@k']:>8.4f}{rr_n_str:>10}")
    print(f"MRR@{k:<8} {results['mean_mrr@k']:>8.4f}{rr_m_str:>10}")
    print(f"\nAnswerable: {ans_count} | Unanswerable: {unans_count}")


if __name__ == "__main__":
    main()


