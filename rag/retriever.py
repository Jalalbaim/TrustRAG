import os
import sys
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from index.build_index import l2_normalize

_DEFAULT_INDEX_DIR = "./index"
_STATE = {
    "loaded": False,
    "index": None,
    "meta": None,
    "cfg": None,
    "model": None,
}

def load_index(index_dir = _DEFAULT_INDEX_DIR, device = "cpu"):

    global _STATE
    if _STATE["loaded"]:
        return

    faiss_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.parquet")
    cfg_path = os.path.join(index_dir, "config.json")

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"Missing FAISS index: {faiss_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.parquet: {meta_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    index = faiss.read_index(faiss_path)

    meta = pd.read_parquet(meta_path)
    required_cols = {"row_id", "pid", "article_title", "section_title", "passage_text"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"meta.parquet missing columns: {sorted(missing)}")

    model_name = cfg.get("embedding_model")
    if not model_name:
        raise ValueError("config.json missing 'embedding_model'")

    model = SentenceTransformer(model_name, device=device)

    _STATE.update(
        {
            "loaded": True,
            "index": index,
            "meta": meta,
            "cfg": cfg,
            "model": model,
        }
    )

def _embed_query(query: str):
    cfg = _STATE["cfg"]
    model = _STATE["model"]
    if cfg is None or model is None:
        raise RuntimeError("Index not loaded. Call load_index() first.")

    query_prefix = cfg.get("query_prefix", "query:")
    q = f"{query_prefix} {query}".strip()

    emb = model.encode([q], normalize_embeddings=False)
    emb = emb.astype(np.float32)
    emb = l2_normalize(emb).astype(np.float32)
    return emb


def retrieve(query: str, k: int = 10) -> List[Dict]:

    if not _STATE["loaded"]:
        load_index()

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    index = _STATE["index"]
    meta = _STATE["meta"]

    qvec = _embed_query(query)

    scores, idxs = index.search(qvec, int(k))

    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    results: List[Dict] = []
    for score, row_id in zip(scores, idxs):
        if row_id == -1:
            continue

        row = meta.iloc[int(row_id)]
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


def main():
    load_index(device="cpu")
    q = "What is diabetes and how is it diagnosed?"
    hits = retrieve(q, k=5)
    for i, h in enumerate(hits, 1):
        print(f"\n#{i} score={h['score']:.4f} pid={h['pid']}")
        print(f"{h['article_title']} — {h['section_title']}")
        print(h["passage_text"][:400], "...")

if __name__ == "__main__":
    main()