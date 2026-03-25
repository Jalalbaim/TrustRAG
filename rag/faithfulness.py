"""
NLI-based faithfulness checker for TrustRAG.

After generation, checks every claim (sentence) in the answer against the
retrieved passages using a cross-encoder NLI model.  Returns a faithfulness
score and per-claim verdicts so callers can refuse or warn on hallucinations.
"""

import json
from typing import List, Dict

import nltk
from sentence_transformers import CrossEncoder

# ── Constants ─────────────────────────────────────────────────────────────────

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Fraction of claims that must be entailed for the answer to be "faithful"
FAITHFULNESS_THRESHOLD = 0.7

# Minimum entailment probability for a single claim to count as entailed
ENTAILMENT_MIN_SCORE = 0.5

# ── Lazy global ───────────────────────────────────────────────────────────────

_nli_model = None


def _load_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        print(f"Loading NLI model: {NLI_MODEL_NAME}")
        _nli_model = CrossEncoder(NLI_MODEL_NAME)
    return _nli_model


def _ensure_nltk() -> None:
    """Download punkt tokenizer data quietly on first use."""
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
        except LookupError:
            token = resource.split("/")[-1]
            nltk.download(token, quiet=True)


# ── Public API ────────────────────────────────────────────────────────────────

def check_faithfulness(answer: str, passages: List[Dict]) -> Dict:
    """Check every claim (sentence) in *answer* against *passages* via NLI.

    Args:
        answer:   The generated answer string.
        passages: List of passage dicts — must contain ``passage_text`` and
                  ``pid`` fields (same format as reranker output).

    Returns:
        A dict with keys:

        * ``score``    (float) – fraction of claims labelled *entailed*
        * ``faithful`` (bool)  – ``score >= FAITHFULNESS_THRESHOLD``
        * ``claims``   (list)  – per-claim dicts, each with:

          - ``text``             – the claim sentence
          - ``verdict``          – ``"entailed"`` or ``"not_entailed"``
          - ``entail_score``     – max entailment probability across passages
          - ``best_passage_pid`` – pid of the passage that best supports the claim
    """
    _ensure_nltk()
    model = _load_model()

    claims = nltk.sent_tokenize(answer)
    if not claims or not passages:
        return {"score": 0.0, "faithful": False, "claims": []}

    claim_results = []
    for claim_text in claims:
        # Premise = passage text, Hypothesis = claim (standard NLI order)
        pairs = [(p["passage_text"], claim_text) for p in passages]
        # probs shape: (num_passages, 3) — columns: [contradiction, neutral, entailment]
        probs = model.predict(pairs, apply_softmax=True)

        best_idx = int(probs[:, 2].argmax())
        best_entail_score = float(probs[best_idx, 2])
        best_pid = passages[best_idx].get("pid", "unknown")

        verdict = "entailed" if best_entail_score >= ENTAILMENT_MIN_SCORE else "not_entailed"
        claim_results.append(
            {
                "text": claim_text,
                "verdict": verdict,
                "entail_score": round(best_entail_score, 4),
                "best_passage_pid": best_pid,
            }
        )

    n_entailed = sum(1 for c in claim_results if c["verdict"] == "entailed")
    score = n_entailed / len(claim_results)

    return {
        "score": round(score, 4),
        "faithful": score >= FAITHFULNESS_THRESHOLD,
        "claims": claim_results,
    }


def faithfulness_summary(result: Dict) -> str:
    """Format a ``check_faithfulness`` result as a human-readable string."""
    status = "PASS" if result["faithful"] else "FAIL"
    lines = [
        f"Faithfulness score: {result['score']:.2%}  "
        f"({status}, threshold={FAITHFULNESS_THRESHOLD:.0%})"
    ]
    for i, c in enumerate(result["claims"], 1):
        icon = "✓" if c["verdict"] == "entailed" else "✗"
        lines.append(
            f"  [{icon}] Claim {i} (entail={c['entail_score']:.3f}, "
            f"best_pid={c['best_passage_pid']}): {c['text']}"
        )
    return "\n".join(lines)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sentence 1: clearly entailed by p001
    # Sentence 2: unsupported / neutral — not present in either passage
    # Sentence 3: contradiction of p002 (aspirin *increases* bleeding risk)
    _answer = (
        "Aspirin is a nonsteroidal anti-inflammatory drug used to reduce fever and pain. "
        "It was first synthesized in ancient Egypt thousands of years ago. "
        "Aspirin strengthens platelet aggregation and thereby reduces bleeding risk."
    )

    _passages = [
        {
            "pid": "p001",
            "passage_text": (
                "Aspirin, also known as acetylsalicylic acid (ASA), is a nonsteroidal "
                "anti-inflammatory drug (NSAID) used to reduce fever, pain, and inflammation. "
                "It works by inhibiting cyclooxygenase enzymes."
            ),
        },
        {
            "pid": "p002",
            "passage_text": (
                "Aspirin irreversibly inhibits platelet aggregation, which increases bleeding "
                "risk. It is commonly used as an antiplatelet agent to prevent cardiovascular "
                "events such as heart attacks and strokes."
            ),
        },
    ]

    result = check_faithfulness(_answer, _passages)

    print("\n=== Faithfulness Check Demo ===")
    print(faithfulness_summary(result))
    print("\nFull result:")
    print(json.dumps(result, indent=2))
