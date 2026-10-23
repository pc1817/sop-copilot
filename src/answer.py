"""Stage 7: grounded answers with citations, plus the refusal gate."""
import os
import re
import time

import requests

from src.retrieve import retrieve_all

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MIN_DENSE_SCORE = 0.665  # calibrate.py: in-scope >= 0.700, everything else <= 0.633
REFUSAL = "I can't find that in the GitLab Finance handbook pages I have indexed, so I won't guess."
SENTINEL = "NOT_IN_DOCS"

SYSTEM = (
    "You answer questions about GitLab's Finance handbook using ONLY the numbered excerpts provided.\n"
    "Rules:\n"
    "1. Use only facts stated in the excerpts. Never use outside knowledge.\n"
    "2. Cite each claim with its excerpt number in square brackets, placed at the end of the sentence or step it supports, like: Attach the receipt in Navan. [2] Never group citations at the end of the answer.\n"
    f"3. If the excerpts do not contain the answer, including when they only tell the reader to see "
    f"another page or the internal handbook, reply with exactly {SENTINEL} and nothing else.\n"
    "4. Be concise (under 150 words). Use short steps when the excerpts describe a procedure."
)


def _call_groq(question: str, hits: list[dict]) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set. Put it in .env")
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    excerpts = "\n\n".join(
        f"[{i}] ({h['heading_path']})\n{h['text']}" for i, h in enumerate(hits, start=1)
    )
    body = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": 1500,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
        ],
    }
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(1, 5):
        r = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code in (429, 500, 502, 503):
            try:
                wait = float(r.headers.get("retry-after", ""))
            except ValueError:
                wait = 20 * attempt
            print(f"  Groq HTTP {r.status_code}, waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Groq: too many retries")


def answer_question(index, question: str) -> dict:
    r = retrieve_all(index, question)
    dense_top = r["dense_top"]
    hits = r["hybrid"]
    base = {"question": question, "dense_top": dense_top, "retrieved": [h["id"] for h in hits], "hits": hits}

    if dense_top < MIN_DENSE_SCORE:  # gate: no LLM call at all
        return {**base, "answer": REFUSAL, "refused": True, "reason": "low_similarity",
                "uncited": False, "sources": []}

    text = _call_groq(question, hits)
    if SENTINEL in text:
        return {**base, "answer": REFUSAL, "refused": True, "reason": "model_declined",
                "uncited": False, "sources": []}

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    sources = [
        {"n": n, "url": hits[n - 1]["url"], "heading_path": hits[n - 1]["heading_path"]}
        for n in cited
    ]
    return {**base, "answer": text, "refused": False, "reason": None,
            "uncited": not cited, "sources": sources}

