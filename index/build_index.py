import os
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import faiss
from sentence_transformers import SentenceTransformer


def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def l2_normalize(x, eps: float=1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)

def main():

    batch_Size = 256
    device = "gpu" if torch.cuda.is_available() else "cpu"
    max_rows = 10
    out_dir = "./index"

    # data 
    df = pd.read_parquet("./processed/wiki_snippets_subset/wiki_snippets_230610_medicine_seed42.parquet")
    
    required_cols = ["pid", "article_title", "section_title", "passage_text"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")
    
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    texts = df["passage_text"].astype(str).tolist()
    print("Data loaded successfully.")

    # embedder 
    model_name = "intfloat/e5-small-v2"
    model = SentenceTransformer(model_name)
    passage_prefix = "passage: "
    print("Embedding passages...")
    t0 = time.time()

    test_emb = model.encode(["passage: test"], normalize_embeddings=False)
    dim = int(test_emb.shape[1])

    embs = np.empty((len(texts), dim), dtype=np.float32)

    for start in tqdm(range(0, len(texts), batch_Size), desc="Embedding"):
        end = min(start + batch_Size, len(texts))
        batch_texts = [passage_prefix + t for t in texts[start:end]]
        batch_emb = model.encode(
            batch_texts,
            batch_size= batch_Size,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype(np.float32)
        embs[start:end] = batch_emb
    
    embs = l2_normalize(embs).astype(np.float32)
    print(f"Embedding completed in {time.time() - t0:.2f} seconds.")

    # FAISS 
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    print("FAISS index created and saved.")

    out_meta = os.path.join(out_dir, "meta.parquet")
    meta_cols = ["row_id", "pid", "article_title", "section_title", "passage_text"]
    meta_df = df[meta_cols].copy()
    meta_df.to_parquet(out_meta, index=False)
    print("Metadata saved successfully.")

    out_cfg = os.path.join(out_dir, "config.json")
    cfg = {
        "created_utc": now_iso(),
        "corpus_parquet": os.path.abspath("./processed/wiki_snippets_subset/wiki_snippets_230610_medicine_seed42.parquet"),
        "out_dir": os.path.abspath(out_dir),
        "embedding_model": model_name,
        "device": device,
        "batch_size": batch_Size,
        "dim": dim,
        "faiss_index_type": "IndexFlatIP",
        "similarity": "L2-normalized embeddings",
        "passage_prefix": passage_prefix.strip(),
        "query_prefix": "query:",
        "n_rows_indexed": int(len(df)),
    }

    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    print("Done")

if __name__ == "__main__":
    main()