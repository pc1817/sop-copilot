"""Answer-level eval: every question goes through the full pipeline (gate + Groq). Uses Groq tokens."""
import json
import random
import time
from datetime import datetime
from pathlib import Path

from src.embed import load_env
from src.index import get_index
from src.answer import answer_question, MIN_DENSE_SCORE
from run_eval import load, is_gold

PAUSE = 22  # seconds after each Groq call, keeps us under the 8K tokens/min limit


def main() -> None:
    load_env()
    own = load("eval/own_questions.json")
    own_ids = {q["id"] for q in own}
    questions = load("eval/questions.json") + own
    index = get_index()

    rows, hits_by_id = [], {}
    for n, q in enumerate(questions, start=1):
        res = answer_question(index, q["q"])
        used_llm = res["reason"] != "low_similarity"
        row = {
            "id": q["id"], "type": q["type"], "own": q["id"] in own_ids, "q": q["q"],
            "refused": res["refused"], "reason": res["reason"], "dense_top": res["dense_top"],
            "uncited": res["uncited"], "answer": res["answer"],
            "cited": [s["heading_path"] for s in res["sources"]],
            "retrieved": res["retrieved"], "used_llm": used_llm,
        }
        if q["type"] == "in_scope":
            row["gold_in_context"] = any(is_gold(h, q["gold"]) for h in res["hits"])
            if q.get("must") and not res["refused"]:
                row["must_ok"] = all(t.lower() in res["answer"].lower() for t in q["must"])
        rows.append(row)
        hits_by_id[q["id"]] = res["hits"]
        print(f"  {n}/{len(questions)} {q['id']}  {'LLM ' if used_llm else 'gate'}  refused={res['refused']}")
        if used_llm:
            time.sleep(PAUSE)

    ins = [r for r in rows if r["type"] == "in_scope"]
    answered = [r for r in ins if not r["refused"]]
    print(f"\nGate at {MIN_DENSE_SCORE}")
    print(f"IN-SCOPE: answered {len(answered)}/{len(ins)}, refused {len(ins) - len(answered)}")
    print(f"  answers with no citation: {sum(r['uncited'] for r in answered)}")
    print(f"  correct chunk was in the model's context: "
          f"{sum(r['gold_in_context'] for r in answered)}/{len(answered)}")
    must = [r for r in answered if "must_ok" in r]
    if must:
        print(f"  required terms present: {sum(r['must_ok'] for r in must)}/{len(must)}")
    print("  answered WITHOUT the correct chunk in context:",
          [r["id"] for r in answered if not r["gold_in_context"]] or "none")

    for t in ("out_of_scope", "near_miss"):
        sub = [r for r in rows if r["type"] == t]
        by_gate = sum(r["reason"] == "low_similarity" for r in sub)
        by_model = sum(r["reason"] == "model_declined" for r in sub)
        leaked = [r for r in sub if not r["refused"]]
        print(f"\n{t.upper()}: {len(sub)} questions | refused by gate {by_gate} | "
              f"refused by model {by_model} | ANSWERED {len(leaked)}")
        for r in leaked:
            print(f"  {r['id']} ({r['dense_top']:.3f}) {r['q']}\n    -> {r['answer'][:300]}")

    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    (out / f"answers_{stamp}.json").write_text(
        json.dumps({"threshold": MIN_DENSE_SCORE, "rows": rows}, indent=2), encoding="utf-8")

    sample = random.Random(7).sample(answered, min(10, len(answered)))
    review_ids = {r["id"] for r in sample} | {r["id"] for r in rows if r["type"] != "in_scope" and not r["refused"]}
    lines = []
    for r in rows:
        if r["id"] in review_ids:
            lines.append(f"=== {r['id']} ({r['type']})\nQUESTION: {r['q']}\n\nANSWER:\n{r['answer']}\n\nEXCERPTS THE MODEL SAW:")
            for i, h in enumerate(hits_by_id[r["id"]], start=1):
                lines.append(f"[{i}] {h['heading_path']}\n{h['text']}\n")
            lines.append("")
    (out / f"review_{stamp}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved eval/results/answers_{stamp}.json and review_{stamp}.txt")


if __name__ == "__main__":
    main()
