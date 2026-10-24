"""Retrieval-only eval: BM25 vs dense vs hybrid, plus the refusal gate. Uses no Groq tokens."""
import json
import time
from datetime import datetime
from pathlib import Path

from src.embed import load_env
from src.index import get_index
from src.parse import BASE_URL
from src.retrieve import retrieve_all
from src.answer import MIN_DENSE_SCORE

MODES = ("bm25", "dense", "hybrid")


def load(path: str) -> list:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else []


def is_gold(hit: dict, gold_items: list) -> bool:
    for item in gold_items:
        page, head = item[0], item[1]
        text = item[2] if len(item) > 2 else ""
        if (hit["url"] == BASE_URL + page
                and head.lower() in hit["heading_path"].lower()
                and text.lower() in hit["text"].lower()):
            return True
    return False


def first_rank(hits: list, gold_items: list):
    for i, h in enumerate(hits, start=1):
        if is_gold(h, gold_items):
            return i
    return None


def summarize(subset: list, label: str) -> None:
    n = len(subset)
    if n == 0:
        return
    print(f"\n{label} (n={n})        hit@1   hit@5   MRR")
    for m in MODES:
        ranks = [x[m + "_rank"] for x in subset]
        h1 = sum(r == 1 for r in ranks)
        h5 = sum(r is not None for r in ranks)
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"  {m:7s}  {h1:3d}/{n}  {h5:3d}/{n}   {mrr:.3f}")


def main() -> None:
    load_env()
    own = load("eval/own_questions.json")
    own_ids = {q["id"] for q in own}
    questions = load("eval/questions.json") + own
    index = get_index()

    rows = []
    for n, q in enumerate(questions, start=1):
        r = retrieve_all(index, q["q"], top_k=5)
        row = {
            "id": q["id"], "type": q["type"], "own": q["id"] in own_ids, "q": q["q"],
            "dense_top": r["dense_top"], "gate_pass": r["dense_top"] >= MIN_DENSE_SCORE,
        }
        for m in MODES:
            row[m] = [{"id": h["id"], "heading": h["heading_path"], "score": h["score"]} for h in r[m]]
            if q["type"] == "in_scope":
                row[m + "_rank"] = first_rank(r[m], q["gold"])
        rows.append(row)
        print(f"  {n}/{len(questions)} {q['id']}")
        time.sleep(1)

    ins = [x for x in rows if x["type"] == "in_scope"]
    summarize(ins, "IN-SCOPE, all")
    summarize([x for x in ins if not x["own"]], "IN-SCOPE, drafted with AI")
    summarize([x for x in ins if x["own"]], "IN-SCOPE, your own words")

    def rk(v):
        return "-" if v is None else str(v)

    print("\nPer question: rank of first correct chunk in the top 5 ('-' = missed)")
    for x in ins:
        flag = "   <- gate would refuse" if not x["gate_pass"] else ""
        print(f"  {x['id']}  bm25 {rk(x['bm25_rank']):>2}  dense {rk(x['dense_rank']):>2}  "
              f"hybrid {rk(x['hybrid_rank']):>2}   top dense {x['dense_top']:.3f}{flag}")

    print(f"\nRefusal gate at {MIN_DENSE_SCORE}")
    for t in ("in_scope", "out_of_scope", "near_miss"):
        sub = [x for x in rows if x["type"] == t]
        passed = sum(x["gate_pass"] for x in sub)
        print(f"  {t:13s} passed the gate {passed}/{len(sub)}   refused {len(sub) - passed}/{len(sub)}")
    leaked = [(x["id"], round(x["dense_top"], 3)) for x in rows
              if x["type"] != "in_scope" and x["gate_pass"]]
    print("  out-of-scope or near-miss questions that got past the gate:", leaked or "none")

    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"retrieval_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"threshold": MIN_DENSE_SCORE, "rows": rows}, indent=2), encoding="utf-8")
    print("\nsaved", path)


if __name__ == "__main__":
    main()
