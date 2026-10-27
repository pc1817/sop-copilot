import json
import re
from pathlib import Path

CITE = re.compile(r"[\[\u3010]\d+[\]\u3011]")
STEP = re.compile(r"^(\d+[.)]|[-*])\s")
TEN = {"q02", "q03", "q04", "q05", "q11", "q12", "q13", "q18", "q21", "q22"}


def uncited_steps(text: str) -> int:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return sum(1 for ln in lines if STEP.match(ln) and not CITE.search(ln))


def stats(rows: list) -> str:
    if not rows:
        return "n=0"
    steps = sum(uncited_steps(r["answer"]) for r in rows)
    words = sum(len(r["answer"].split()) for r in rows) / len(rows)
    return f"n={len(rows)} uncited_steps={steps} avg_words={words:.0f}"


for path in sorted(Path("eval/results").glob("answers_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data.get("config", {"prompt": "v1", "retrieval": "dense"})
    rows = [r for r in data["rows"] if r["type"] == "in_scope" and not r["refused"]]
    ten = [r for r in rows if r["id"] in TEN]
    print(f"{path.name}\n  prompt={cfg['prompt']} retrieval={cfg['retrieval']}")
    print(f"  all answered: {stats(rows)}")
    print(f"  the 10 shared: {stats(ten)}")