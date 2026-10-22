import sys
from pathlib import Path

from src.parse import parse_all
from src.clean import clean_pages
from src.chunk import chunk_pages
from src.embed import load_env, embed_texts, embed_text_for, MODEL, DIM

load_env()
pages = parse_all(Path("docs"))
kept, dropped = clean_pages(pages)
chunks = chunk_pages(kept)
print(f"parsed {len(pages)} | kept {len(kept)} | dropped {len(dropped)} | chunks {len(chunks)}")

limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(chunks)
vecs = embed_texts([embed_text_for(c) for c in chunks[:limit]])
print(f"embedded {len(vecs)} chunks with {MODEL}, dim={len(vecs[0])} (expected {DIM})")
print("norm of first vector:", round(sum(x * x for x in vecs[0]) ** 0.5, 4))
