import requests
import ollama
from typing import List

_DEFAULT_MODEL = "gemma3:4b"

def generate_answer(query, context):
    system_prompt = (
        "You are an expert Cyber Threat Intelligence Analyst. "
        "Based ONLY on the provided real-time data from the last 60 minutes, "
        "provide a concise summary, a list of IoCs, and a recommended patch/mitigation priority."
    )
    
    full_prompt = f"{system_prompt}\n\nUSER QUERY: {query}\n\nDATA CONTEXT:\n{context}"
    
    try:
        response = ollama.generate(
            model="gemma3:4b",
            prompt=full_prompt
        )
        return response['response']
    except Exception as e:
        return f"Error generating report: {str(e)}"
    

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from rag.hybrid_retriever import load_all, retrieve_hybrid

    load_all(device="cpu")
    q = "What is diabetes and how is it diagnosed?"
    hits = retrieve_hybrid(q, k=10, rerank=True, reranker_top_k=3)
    context_str = "\n\n---\n\n".join(h["passage_text"] for h in hits)
    answer = generate_answer(q, context_str)
    print(f"Q: {q}\n\nA: {answer}")
