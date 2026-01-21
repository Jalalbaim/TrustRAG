import os
import re
import json
import pickle
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


_DEFAULT_INDEX_DIR = "./index"
_STATE = {
    "loaded": False,
    "faiss": None,
    "meta": None,
    "cfg": None,
    "model": None,
    "bm25": None,
}

_token_re = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str):
    return _token_re.findall((text or "").lower())


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def load_all(index_dir: str = _DEFAULT_INDEX_DIR, device: str = "cpu") -> None:
    global _STATE
    if _STATE["loaded"]:
        return

    faiss_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.parquet")
    cfg_path = os.path.join(index_dir, "config.json")
    bm25_path = os.path.join(index_dir, "bm25.pkl")

    for p in [faiss_path, meta_path, cfg_path, bm25_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    index = faiss.read_index(faiss_path)
    meta = pd.read_parquet(meta_path)

    model_name = cfg.get("embedding_model")
    if not model_name:
        raise ValueError("config.json missing 'embedding_model'")
    model = SentenceTransformer(model_name, device=device)

    with open(bm25_path, "rb") as f:
        bm25_obj = pickle.load(f)["bm25"]

    _STATE.update(
        {
            "loaded": True,
            "faiss": index,
            "meta": meta,
            "cfg": cfg,
            "model": model,
            "bm25": bm25_obj,
        }
    )


def _embed_query(q: str) -> np.ndarray:
    cfg = _STATE["cfg"]
    model = _STATE["model"]
    prefix = cfg.get("query_prefix", "query:")
    q2 = f"{prefix} {q}".strip()
    emb = model.encode([q2], normalize_embeddings=False).astype(np.float32)
    return _l2_normalize(emb).astype(np.float32)


def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def retrieve_hybrid(
    query: str,
    k: int = 10,
    dense_k: int = 100,
    bm25_k: int = 200,
    alpha: float = 0.5,   # weight for dense
    beta: float = 0.5,    # weight for bm25
    min_score: Optional[float] = None,
) -> List[Dict]:
    
    if not _STATE["loaded"]:
        load_all()

    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    faiss_index = _STATE["faiss"]
    meta = _STATE["meta"]
    bm25 = _STATE["bm25"]

    # Dense
    qvec = _embed_query(query)
    dense_scores, dense_idxs = faiss_index.search(qvec, int(dense_k))
    dense_scores = dense_scores[0].astype(np.float32)
    dense_idxs = dense_idxs[0].astype(np.int64)
    # top dense_k
    mask = dense_idxs >= 0
    dense_scores = dense_scores[mask]
    dense_idxs = dense_idxs[mask]

    dense_norm = _minmax_norm(dense_scores)

    # BM25
    q_tokens = tokenize(query)
    bm25_scores_all = np.array(bm25.get_scores(q_tokens), dtype=np.float32)  # size = n_docs
    # top bm25_k
    if bm25_k < len(bm25_scores_all):
        top_b = np.argpartition(-bm25_scores_all, bm25_k)[:bm25_k]
    else:
        top_b = np.arange(len(bm25_scores_all), dtype=np.int64)

    bm25_scores = bm25_scores_all[top_b]
    bm25_idxs = top_b.astype(np.int64)
    bm25_norm = _minmax_norm(bm25_scores)

    # Fusion
    fused = {}
    for rid, s in zip(dense_idxs.tolist(), dense_norm.tolist()):
        fused[rid] = [s, 0.0]
    for rid, s in zip(bm25_idxs.tolist(), bm25_norm.tolist()):
        if rid in fused:
            fused[rid][1] = s
        else:
            fused[rid] = [0.0, s]

    # final score
    items = []
    for rid, (sd, sb) in fused.items():
        if sb < 0.01 : # penalise for low BM25 score
            sb = -1.0
        score = alpha * sd + beta * sb
        items.append((rid, float(score)))

    items.sort(key=lambda x: x[1], reverse=True)
    items = items[: int(k)]

    results = []
    for rid, score in items:
        row = meta.iloc[int(rid)]
        if (min_score is not None) and (score < float(min_score)):
            continue
        results.append(
            {
                "pid": row["pid"],
                "score": float(score),
                "article_title": row["article_title"],
                "section_title": row["section_title"],
                "passage_text": row["passage_text"],
            }
        )
    return results


if __name__ == "__main__":
    load_all(device="cpu")
    q = "What is diabetes and how is it diagnosed?"
    hits = retrieve_hybrid(q, k=5)
    for i, h in enumerate(hits, 1):
        print(f"\n#{i} score={h['score']:.4f} pid={h['pid']}")
        print(f"{h['article_title']} — {h['section_title']}")
        print(h["passage_text"][:350], "...")
