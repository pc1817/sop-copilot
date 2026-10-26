"""Answer-level eval: every question goes through the full pipeline (gate + Groq). Uses Groq tokens.

  python run_answers.py                  all 40 questions
  python run_answers.py --ids q02,q03    only these questions
  python run_answers.py --tag v2-dense   label for the output files
Config comes from the environment: PROMPT_VERSION (v1 or v2) and RETRIEVAL_MODE (dense or hybrid).
"""
import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

from src.embed import load_env
from src.index import get_index
from src.answer import answer_question, MIN_DENSE_SCORE
from run_eval import load, is_gold

PAUSE = 25  # seconds after each Groq call, keeps us under the 8K tokens/min limit
REVIEW_IDS = ["q02", "q03", "q04", "q05", "q11", "q12", "q13", "q18", "q21", "q22"]  # the v1 review sample
CITE = re.compile(r"[\[\u3010]\d+[\]\u3011]")
STEP = re.compile(r"^(\d+[.)]|[-*])\s")


def uncited_steps(text: str) -> int:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return sum(1 for ln in lines if STEP.match(ln) and not CITE.search(ln))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    load_env()
    prompt = os.environ.get("PROMPT_VERSION", "v2")
    mode = os.environ.get("RETRIEVAL_MODE", "hybrid")
    own = load("eval/own_questions.json")
    own_ids = {q["id"] for q in own}
    questions = load("eval/questions.json") + own
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    tag = args.tag or f"{prompt}-{mode}"
    print(f"config: prompt={prompt} retrieval={mode} questions={len(questions)}")
    index = get_index()

    rows, hits_by_id = [], {}
    for n, q in enumerate(questions, start=1):
        res = answer_question(index, q["q"])
        used_llm = res["reason"] != "low_similarity"
        row = {
            "id": q["id"], "type": q["type"], "own": q["id"] in own_ids, "q": q["q"],
            "refused": res["refused"], "reason": res["reason"], "dense_top": res["dense_top"],
            "uncited": res["uncited"], "answer": res["answer"],
            "uncited_steps": 0 if res["refused"] else uncited_steps(res["answer"]),
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
    print(f"\nconfig: prompt={prompt} retrieval={mode} | gate at {MIN_DENSE_SCORE}")
    if ins:
        print(f"IN-SCOPE: answered {len(answered)}/{len(ins)}, refused {len(ins) - len(answered)}")
        print(f"  answers with no citation at all: {sum(r['uncited'] for r in answered)}")
        print(f"  list steps without a citation: {sum(r['uncited_steps'] for r in answered)} "
              f"(in {sum(r['uncited_steps'] > 0 for r in answered)} answers)")
        print(f"  correct chunk was in the model's context: "
              f"{sum(r['gold_in_context'] for r in answered)}/{len(answered)}")
        must = [r for r in answered if "must_ok" in r]
        if must:
            print(f"  required terms present: {sum(r['must_ok'] for r in must)}/{len(must)}")
        print("  refused in-scope questions:", [r["id"] for r in ins if r["refused"]] or "none")
        print("  answered WITHOUT the correct chunk in context:",
              [r["id"] for r in answered if not r["gold_in_context"]] or "none")

    for t in ("out_of_scope", "near_miss"):
        sub = [r for r in rows if r["type"] == t]
        if not sub:
            continue
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
    config = {"prompt": prompt, "retrieval": mode, "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")}
    (out / f"answers_{tag}_{stamp}.json").write_text(
        json.dumps({"config": config, "threshold": MIN_DENSE_SCORE, "rows": rows}, indent=2), encoding="utf-8")

    if args.ids:
        review_ids = {r["id"] for r in rows}
    else:
        pool = [r["id"] for r in answered if r["id"] not in REVIEW_IDS]
        extra = random.Random(11).sample(pool, min(5, len(pool)))
        leaked_ids = {r["id"] for r in rows if r["type"] != "in_scope" and not r["refused"]}
        review_ids = set(REVIEW_IDS) | set(extra) | leaked_ids
    lines = []
    for r in rows:
        if r["id"] in review_ids and r["id"] in hits_by_id:
            lines.append(f"=== {r['id']} ({r['type']}) config {prompt}/{mode}\nQUESTION: {r['q']}\n\n"
                         f"ANSWER:\n{r['answer']}\n\nEXCERPTS THE MODEL SAW:")
            for i, h in enumerate(hits_by_id[r["id"]], start=1):
                lines.append(f"[{i}] {h['heading_path']}\n{h['text']}\n")
            lines.append("")
    (out / f"review_{tag}_{stamp}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved eval/results/answers_{tag}_{stamp}.json and review_{tag}_{stamp}.txt")


if __name__ == "__main__":
    main()