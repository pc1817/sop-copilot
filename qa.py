import sys

from src.embed import load_env
from src.index import get_index
from src.answer import answer_question

load_env()
show = "--show" in sys.argv
args = [a for a in sys.argv[1:] if a != "--show"]
if not args:
    raise SystemExit('usage: python qa.py "your question" [--show]')
res = answer_question(get_index(), args[0])
print(f"Q: {res['question']}")
print(f"[top dense score {res['dense_top']:.3f} | refused: {res['refused']} ({res['reason']})]\n")
print(res["answer"])
if res["sources"]:
    print("\nSources:")
    for s in res["sources"]:
        print(f"  [{s['n']}] {s['heading_path'][:90]}\n      {s['url']}")
if show:
    print("\nExcerpts given to the model:")
    for i, h in enumerate(res["hits"], start=1):
        print(f"\n[{i}] {h['heading_path']}\n{h['text']}")
