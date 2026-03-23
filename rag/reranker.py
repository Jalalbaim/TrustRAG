"""
Cross-encoder reranker for TrustRAG.

Takes passage dicts from hybrid_retriever and reorders them by true relevance
using a cross-encoder model (query, passage) scoring.
"""

from typing import List, Dict

from sentence_transformers import CrossEncoder

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_STATE: Dict = {
    "loaded": False,
    "model": None,
    "model_name": None,
}


def load_reranker(model_name: str = _DEFAULT_MODEL) -> None:
    global _STATE
    if _STATE["loaded"]:
        return
    print(f"Loading cross-encoder reranker: {model_name}")
    _STATE["model"] = CrossEncoder(model_name)
    _STATE["model_name"] = model_name
    _STATE["loaded"] = True


def rerank(query: str, passages: List[Dict], top_k: int = 5) -> List[Dict]:
    """Score each (query, passage_text) pair with the cross-encoder and return
    the top_k passages sorted by reranker_score descending.

    Args:
        query:    The user query string.
        passages: List of passage dicts (must contain 'passage_text').
                  Accepts the exact format output by retrieve_hybrid().
        top_k:    Number of passages to return after reranking.

    Returns:
        A new list of passage dicts (copies) with an added 'reranker_score'
        field, sorted by that score descending, truncated to top_k.
    """
    if not _STATE["loaded"]:
        load_reranker()

    if not passages:
        return []

    model: CrossEncoder = _STATE["model"]

    pairs = [(query, p["passage_text"]) for p in passages]
    scores = model.predict(pairs)  # numpy array, shape (len(passages),)

    scored = []
    for passage, score in zip(passages, scores):
        p = dict(passage)
        p["reranker_score"] = float(score)
        scored.append(p)

    scored.sort(key=lambda x: x["reranker_score"], reverse=True)
    return scored[:top_k]
