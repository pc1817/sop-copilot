"""Stage 4: embed chunks with Gemini (REST), with a resumable disk cache."""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import requests

MODEL = "gemini-embedding-001"
DIM = 768
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchEmbedContents"
CACHE = Path(".cache/embeddings.jsonl")
BATCH = 20
PAUSE = 11  # seconds between calls, keeps us under the free-tier tokens-per-minute limit


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def embed_text_for(chunk: dict) -> str:
    """What we embed: the heading path gives each chunk its context."""
    return f"{chunk['heading_path']}\n\n{chunk['text']}"


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cache_key(text: str, task_type: str) -> str:
    return hashlib.sha1(f"{MODEL}|{DIM}|{task_type}|{text}".encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["k"]] = row["v"]
    return cache


def _call(texts: list[str], task_type: str) -> list[list[float]]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set. Put it in .env")
    body = {"requests": [{
        "model": f"models/{MODEL}",
        "content": {"parts": [{"text": t}]},
        "taskType": task_type,
        "outputDimensionality": DIM,
    } for t in texts]}
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    for attempt in range(1, 6):
        r = requests.post(URL, headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            return [_normalize(e["values"]) for e in r.json()["embeddings"]]
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 30 * attempt
            print(f"  HTTP {r.status_code}, waiting {wait}s (attempt {attempt}/5)")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Gemini: too many retries")


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    cache = _load_cache()
    keys = [_cache_key(t, task_type) for t in texts]
    todo = list({k: t for k, t in zip(keys, texts) if k not in cache}.items())
    if todo:
        CACHE.parent.mkdir(exist_ok=True)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        vecs = _call([t for _, t in batch], task_type)
        with CACHE.open("a", encoding="utf-8") as f:
            for (k, _), v in zip(batch, vecs):
                cache[k] = v
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        print(f"  embedded {min(i + BATCH, len(todo))}/{len(todo)}")
        if i + BATCH < len(todo):
            time.sleep(PAUSE)
    return [cache[k] for k in keys]
