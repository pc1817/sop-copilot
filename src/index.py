"""Stage 5: create the Pinecone document index (BM25 + dense) and upsert chunks."""
import os
import time

from pinecone import Pinecone, SchemaBuilder

from src.embed import embed_text_for, DIM

INDEX_NAME = "sop-copilot"
NAMESPACE = "__default__"
BATCH = 100


def get_client() -> Pinecone:
    key = os.environ.get("PINECONE_API_KEY")
    if not key:
        raise SystemExit("PINECONE_API_KEY is not set. Put it in .env")
    return Pinecone(api_key=key)


def ensure_index(pc: Pinecone):
    """Create the index from a schema if it doesn't exist (schemas can't be edited later)."""
    if not pc.indexes.exists(name=INDEX_NAME):
        schema = (
            SchemaBuilder()
            .add_string_field("search_text", full_text_search={"language": "en"})
            .add_dense_vector_field("embedding", dimension=DIM, metric="cosine")
            .build()
        )
        pc.indexes.create(name=INDEX_NAME, schema=schema)
        print(f"created index {INDEX_NAME}")
    while not pc.indexes.describe(name=INDEX_NAME).status.ready:
        time.sleep(2)
    return pc.Index(name=INDEX_NAME)


def get_index():
    return ensure_index(get_client())


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]):
    index = get_index()
    for i in range(0, len(chunks), BATCH):
        docs = [
            {
                "_id": c["id"],
                "embedding": v,
                "search_text": embed_text_for(c),  # what BM25 and the embedding both see
                "text": c["text"],                 # what we show in answers
                "url": c["url"],
                "title": c["title"],
                "heading_path": c["heading_path"],
            }
            for c, v in zip(chunks[i:i + BATCH], vectors[i:i + BATCH])
        ]
        index.documents.upsert(namespace=NAMESPACE, documents=docs)
        print(f"  upserted {min(i + BATCH, len(chunks))}/{len(chunks)}")
    return index
