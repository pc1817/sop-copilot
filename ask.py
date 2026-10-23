import sys

from src.embed import load_env
from src.index import get_index
from src.retrieve import retrieve

load_env()
if len(sys.argv) < 2:
    raise SystemExit('usage: python ask.py "your question" [dense|bm25|hybrid]')
question = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid"

index = get_index()
print(f"Q: {question}  [{mode}]\n")
for i, h in enumerate(retrieve(index, question, mode), start=1):
    print(f"{i}. [{h['score']:.3f}] {h['heading_path'][:110]}")
    print(f"   {h['url']}")
    print(f"   {h['text'][:220].replace(chr(10), ' ')}")
    print()
