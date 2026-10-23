"""Retrieval: BM25, dense, and hybrid (reciprocal rank fusion) over one Pinecone index."""
from src.embed import embed_texts
from src.index import NAMESPACE


def _hit(m) -> dict:
    return {
        "id": m._id,
        "score": m._score,
        "text": getattr(m, "text", ""),
        "url": getattr(m, "url", ""),
        "title": getattr(m, "title", ""),
        "heading_path": getattr(m, "heading_path", ""),
    }


def search_dense(index, question: str, k: int = 20) -> list[dict]:
    qv = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    resp = index.documents.search(
        namespace=NAMESPACE, top_k=k, include_fields=["*"],
        score_by=[{"type": "dense_vector", "fields": ["embedding"], "values": qv}],
    )
    return [_hit(m) for m in resp.matches]


def search_bm25(index, question: str, k: int = 20) -> list[dict]:
    resp = index.documents.search(
        namespace=NAMESPACE, top_k=k, include_fields=["*"],
        score_by=[{"type": "text", "field": "search_text", "query": question}],
    )
    return [_hit(m) for m in resp.matches]


def rrf(result_lists: list[list[dict]], k_const: int = 60, top_k: int = 5) -> list[dict]:
    """Reciprocal rank fusion: score = sum of 1/(k_const + rank) across the lists."""
    scores, docs = {}, {}
    for results in result_lists:
        for rank, h in enumerate(results, start=1):
            scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (k_const + rank)
            docs.setdefault(h["id"], h)
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{**docs[i], "score": scores[i]} for i in ordered]


def retrieve(index, question: str, mode: str = "hybrid", top_k: int = 5) -> list[dict]:
    if mode == "dense":
        return search_dense(index, question, top_k)
    if mode == "bm25":
        return search_bm25(index, question, top_k)
    if mode == "hybrid":
        return rrf([search_dense(index, question, 20), search_bm25(index, question, 20)], top_k=top_k)
    raise ValueError(f"unknown mode: {mode}")
