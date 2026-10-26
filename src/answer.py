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

SYSTEM_V1 = (
    "You answer questions about GitLab's Finance handbook using ONLY the numbered excerpts provided.\n"
    "Rules:\n"
    "1. Use only facts stated in the excerpts. Never use outside knowledge.\n"
    "2. Cite each claim with its excerpt number in square brackets, placed at the end of the sentence "
    "or step it supports, like: Attach the receipt in Navan. [2] Never group citations at the end of the answer.\n"
    f"3. If the excerpts do not contain the answer, including when they only tell the reader to see "
    f"another page or the internal handbook, reply with exactly {SENTINEL} and nothing else.\n"
    "4. Be concise (under 150 words). Use short steps when the excerpts describe a procedure."
)

SYSTEM_V2 = (
    "You answer questions about GitLab's Finance handbook using ONLY the numbered excerpts provided.\n"
    "Rules:\n"
    "1. Use only facts stated in the excerpts. Never use outside knowledge.\n"
    "2. Cite each claim with its excerpt number in square brackets, placed at the end of the sentence "
    "or step it supports, like: Attach the receipt in Navan. [2] Never group citations at the end of the answer.\n"
    "3. Every sentence and every step must carry a citation. If you cannot cite it, leave it out.\n"
    f"4. If the excerpts do not contain the answer, including when they only tell the reader to see "
    f"another page or the internal handbook, reply with exactly {SENTINEL} and nothing else.\n"
    "5. If the excerpts answer only part of the question, answer that part and say plainly what is missing.\n"
    "6. Say who must do each step (the team member, the manager, Accounts Payable, and so on) exactly as the "
    "excerpts state it.\n"
    "7. If the excerpts describe different cases (for example different countries, legal entities, or employee "
    "types), say which case each statement applies to. Do not merge them into one general procedure. "
    "If sources conflict, say so.\n"
    "8. If the question names a specific item or situation that the excerpts do not mention, say that the "
    "excerpts do not mention it. Do not assume it is covered.\n"
    "9. Be concise (under 150 words). Use short steps when the excerpts describe a procedure."
)


def _system_prompt() -> str:
    return SYSTEM_V2 if os.environ.get("PROMPT_VERSION", "v2") == "v2" else SYSTEM_V1


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
            {"role": "system", "content": _system_prompt()},
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
    hits = r[os.environ.get("RETRIEVAL_MODE", "hybrid")]
    base = {"question": question, "dense_top": dense_top,
            "retrieved": [h["id"] for h in hits], "hits": hits}

    if dense_top < MIN_DENSE_SCORE:  # gate: no LLM call at all
        return {**base, "answer": REFUSAL, "refused": True, "reason": "low_similarity",
                "uncited": False, "sources": []}

    text = _call_groq(question, hits)
    if SENTINEL in text:
        return {**base, "answer": REFUSAL, "refused": True, "reason": "model_declined",
                "uncited": False, "sources": []}

    cited = sorted({int(n) for n in re.findall(r"[\[\u3010](\d+)[\]\u3011]", text) if 1 <= int(n) <= len(hits)})
    sources = [
        {"n": n, "url": hits[n - 1]["url"], "heading_path": hits[n - 1]["heading_path"]}
        for n in cited
    ]
    return {**base, "answer": text, "refused": False, "reason": None,
            "uncited": not cited, "sources": sources}