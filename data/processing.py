import os
import re
import json
from typing import List, Dict

from datasets import load_dataset

CACHE_DIR = "./data/wiki_snippets"
OUT_DIR = "./processed/wiki_snippets_subset"
CONFIG_NAME = "wikipedia_en_100_0"
SPLIT = "train"

SEED = 42
N_PASSAGES = 300_000

THEMES: Dict[str, List[str]] = {
    "ai_ml": ["machine learning", "artificial intelligence", "deep learning", "neural", "algorithm", "computer vision", "nlp"],
    "medicine": ["medicine", "medical", "disease", "clinical", "hospital", "diagnosis", "surgery", "pharmacology"],
    "law": ["law", "legal", "court", "constitution", "treaty", "legislation", "jurisprudence"],
    "history": ["history", "historical", "empire", "war", "revolution", "dynasty", "ancient", "medieval"],
    "physics": ["physics", "quantum", "relativity", "particle", "thermodynamics", "optics", "electromagnetism"],
    "finance": ["finance", "bank", "economics", "market", "stock", "inflation", "interest rate", "trade"],
    "sports": ["football", "soccer", "basketball", "tennis", "olympics", "athlete", "league", "tournament"],
    "music": ["music", "album", "song", "composer", "band", "singer", "orchestra", "genre"],
}

THEME_KEY = "medicine"

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_theme_regex(keywords: List[str]) -> re.Pattern:
    escaped = [re.escape(k.lower()) for k in keywords]
    pat = "(" + "|".join(escaped) + ")"
    return re.compile(pat, flags=re.IGNORECASE)

def is_theme_match(article_title: str, section_title: str, regex: re.Pattern) -> bool:
    at = (article_title or "")
    st = (section_title or "")
    text = f"{at} {st}"
    return bool(regex.search(text))

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if THEME_KEY not in THEMES:
        raise ValueError(f"Unknown THEME_KEY={THEME_KEY}. Available: {list(THEMES.keys())}")

    theme_regex = build_theme_regex(THEMES[THEME_KEY])

    print("[1/4] Loading dataset (this should be instant if already cached)...")
    ds = load_dataset(
        "wiki_snippets",
        CONFIG_NAME,
        split=SPLIT,
        cache_dir=CACHE_DIR,
    )

    print("[2/4] Filtering by theme keywords (article_title / section_title)...")
    themed = ds.filter(
        lambda ex: is_theme_match(ex.get("article_title", ""), ex.get("section_title", ""), theme_regex),
        num_proc=os.cpu_count() or 1,
        desc=f"Filtering theme={THEME_KEY}",
    )

    themed_count = len(themed)
    print(f"Theme '{THEME_KEY}' matched passages: {themed_count}")

    if themed_count == 0:
        raise RuntimeError("No passages matched your theme keywords. Add/adjust keywords.")

    print("[3/4] Reproducible sampling...")
    themed_shuffled = themed.shuffle(seed=SEED)

    n = min(N_PASSAGES, len(themed_shuffled))
    subset = themed_shuffled.select(range(n))
    print(f"Selected subset size: {len(subset)} (requested {N_PASSAGES})")

    print("[4/4] Cleaning + exporting to Parquet and JSONL...")
    out_base = f"wiki_snippets_{n}_{THEME_KEY}_seed{SEED}"
    parquet_path = os.path.join(OUT_DIR, out_base + ".parquet")
    jsonl_path = os.path.join(OUT_DIR, out_base + ".jsonl")

    def to_clean_example(ex):
        pid = ex.get("datasets_id", None)
        if pid is None:
            pid = ex.get("_id")
        return {
            "pid": pid,
            "wiki_id": ex.get("wiki_id"),
            "article_title": normalize_text(ex.get("article_title", "")),
            "section_title": normalize_text(ex.get("section_title", "")),
            "passage_text": normalize_text(ex.get("passage_text", "")),
            "source": "wiki_snippets:wikipedia_en_100_0",
            "theme": THEME_KEY,
        }

    subset_clean = subset.map(to_clean_example, remove_columns=subset.column_names)

    subset_clean.to_parquet(parquet_path)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in subset_clean:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("Parquet:", parquet_path)
    print("JSONL  :", jsonl_path)


if __name__ == "__main__":
    main()
