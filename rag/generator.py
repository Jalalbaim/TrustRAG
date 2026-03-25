import requests
import ollama
from typing import List, Optional

from rag.faithfulness import (
    check_faithfulness,
    faithfulness_summary,
    FAITHFULNESS_THRESHOLD,
)

_DEFAULT_MODEL = "gemma3:4b"


def generate_answer(
    query: str,
    context: str,
    passages: Optional[List[dict]] = None,
    run_faithfulness_check: bool = True,
) -> dict:
    """Generate an answer grounded in *context* and optionally verify faithfulness.

    Args:
        query:                  The user question.
        context:                Pre-built context string passed to the LLM.
        passages:               Retrieved passage dicts (same format as reranker
                                output).  Required for faithfulness checking.
        run_faithfulness_check: When True (default) and *passages* is provided,
                                run NLI-based faithfulness verification.

    Returns:
        A dict with:

        * ``answer``       (str)        – the generated (or refusal) text
        * ``faithfulness`` (dict|None)  – full result from ``check_faithfulness``,
                                          or ``None`` when the check was skipped
    """
    system_prompt = (
        "You are an expert Cyber Threat Intelligence Analyst. "
        "Based ONLY on the provided real-time data from the last 60 minutes, "
        "provide a concise summary, a list of IoCs, and a recommended patch/mitigation priority."
    )

    full_prompt = f"{system_prompt}\n\nUSER QUERY: {query}\n\nDATA CONTEXT:\n{context}"

    try:
        response = ollama.generate(model=_DEFAULT_MODEL, prompt=full_prompt)
        answer = response["response"]
    except Exception as e:
        answer = f"Error generating report: {str(e)}"

    # ── Faithfulness check ────────────────────────────────────────────────────
    faith_result: Optional[dict] = None

    if run_faithfulness_check and passages:
        faith_result = check_faithfulness(answer, passages)
        print(f"[faithfulness] {faithfulness_summary(faith_result)}")

        if not faith_result["faithful"]:
            not_entailed = [
                c for c in faith_result["claims"] if c["verdict"] == "not_entailed"
            ]
            bullet_lines = "\n".join(f"  • {c['text']}" for c in not_entailed)
            answer = (
                f"[FAITHFULNESS WARNING] This answer was withheld because its "
                f"faithfulness score ({faith_result['score']:.2%}) is below the "
                f"required threshold ({FAITHFULNESS_THRESHOLD:.0%}).\n\n"
                f"The following claims could not be verified against the retrieved passages:\n"
                f"{bullet_lines}"
            )

    return {"answer": answer, "faithfulness": faith_result}


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from rag.hybrid_retriever import load_all, retrieve_hybrid

    load_all(device="cpu")
    q = "What is diabetes and how is it diagnosed?"
    hits = retrieve_hybrid(q, k=10, rerank=True, reranker_top_k=3)
    context_str = "\n\n---\n\n".join(h["passage_text"] for h in hits)
    result = generate_answer(q, context_str, passages=hits)
    print(f"Q: {q}\n\nA: {result['answer']}")
