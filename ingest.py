import json
import time
from pathlib import Path

from src.parse import parse_all
from src.clean import clean_pages
from src.chunk import chunk_pages
from src.embed import load_env, embed_texts, embed_text_for, MODEL, DIM
from src.index import upsert_chunks, INDEX_NAME, NAMESPACE

load_env()
pages = parse_all(Path("docs"))
kept, dropped = clean_pages(pages)
chunks = chunk_pages(kept)
print(f"parsed {len(pages)} | kept {len(kept)} | dropped {len(dropped)} | chunks {len(chunks)}")

vecs = embed_texts([embed_text_for(c) for c in chunks])  # served from cache if already embedded
print(f"embedded {len(vecs)} chunks ({MODEL}, dim {DIM})")

index = upsert_chunks(chunks, vecs)
time.sleep(10)  # documents are indexed asynchronously

report = {
    "corpus_commit": "1d017e2417c7510ed270a22e03f038e968fdb679",
    "pages_parsed": len(pages),
    "pages_kept": len(kept),
    "pages_dropped": len(dropped),
    "dropped_paths": dropped,
    "chunks": len(chunks),
    "embedded": len(vecs),
    "upserted": len(chunks),
    "embedding_model": MODEL,
    "dimensions": DIM,
    "index": INDEX_NAME,
    "namespace": NAMESPACE,
    "search": "pinecone document index: BM25 (search_text) + dense (embedding)",
}
Path("data").mkdir(exist_ok=True)
Path("data/ingest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# smoke test: one question through the whole pipeline, both retrievers
q = "How do I submit an expense report?"
qv = embed_texts([q], task_type="RETRIEVAL_QUERY")[0]
dense = index.documents.search(
    namespace=NAMESPACE, top_k=3, include_fields=["*"],
    score_by=[{"type": "dense_vector", "fields": ["embedding"], "values": qv}],
)
bm25 = index.documents.search(
    namespace=NAMESPACE, top_k=3, include_fields=["*"],
    score_by=[{"type": "text", "field": "search_text", "query": q}],
)
print("\nTest query:", q)
for name, resp in (("dense", dense), ("bm25", bm25)):
    print(f"\n{name}:")
    for m in resp.matches:
        print(f"  {m._score:.3f} | {getattr(m, 'heading_path', '')} | {getattr(m, 'url', '')}")
