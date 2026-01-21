import os
import re
import json
import pickle
import argparse
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi


def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


_token_re = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str):
    return _token_re.findall((text or "").lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, default="./index")
    parser.add_argument("--meta", type=str, default="", help="Override meta.parquet path")
    args = parser.parse_args()

    index_dir = args.index_dir
    meta_path = args.meta or os.path.join(index_dir, "meta.parquet")
    out_pkl = os.path.join(index_dir, "bm25.pkl")
    out_cfg = os.path.join(index_dir, "bm25_config.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.parquet: {meta_path}")

    df = pd.read_parquet(meta_path)

    if "passage_text" not in df.columns:
        raise ValueError("meta.parquet must contain 'passage_text'")

    texts = df["passage_text"].astype(str).tolist()

    tokenized = []
    for t in tqdm(texts, desc="Tokenize"):
        tokenized.append(tokenize(t))

    bm25 = BM25Okapi(tokenized)

    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "tokenizer": "simple_alnum_lower",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    cfg = {
        "created_utc": now_iso(),
        "meta_parquet": os.path.abspath(meta_path),
        "n_docs": int(len(texts)),
        "bm25_class": "BM25Okapi",
        "tokenizer": "simple_alnum_lower",
    }
    print(f"[SAVE] {out_cfg}")
    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
