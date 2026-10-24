"""Sanity-check the eval set against the corpus before running any retriever."""
import json
from pathlib import Path

from src.parse import parse_all, BASE_URL
from src.clean import clean_pages
from src.chunk import chunk_pages


def load(path: str) -> list:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else []


questions = load("eval/questions.json") + load("eval/own_questions.json")
chunks = chunk_pages(clean_pages(parse_all(Path("docs")))[0])

ids = [q["id"] for q in questions]
assert len(ids) == len(set(ids)), "duplicate question ids"
counts = {t: sum(q["type"] == t for q in questions) for t in ("in_scope", "out_of_scope", "near_miss")}
print("questions:", len(questions), counts)

problems = 0
for q in questions:
    if "YOUR QUESTION" in q["q"] or not q["q"].strip():
        print(f"TODO  {q['id']}  still a placeholder")
        problems += 1
        continue
    if q["type"] == "in_scope":
        for item in q["gold"]:
            page, head = item[0], item[1]
            text = item[2] if len(item) > 2 else ""
            n = sum(1 for c in chunks
                    if c["url"] == BASE_URL + page
                    and head.lower() in c["heading_path"].lower()
                    and text.lower() in c["text"].lower())
            label = f"{page} | {head}" + (f" | text: {text}" if text else "")
            if n == 0:
                print(f"NO MATCH  {q['id']}  gold = {label}")
                problems += 1
            else:
                print(f"ok  {q['id']}  {n} chunk(s)  {label}")
    else:
        for term in q.get("absent", []):
            n = sum(1 for c in chunks if term.lower() in c["text"].lower())
            print(f"{q['type']}  {q['id']}  '{term}' appears in {n} chunk(s)")
print("problems:", problems)
