from pathlib import Path

from src.parse import parse_all
from src.clean import clean_pages
from src.chunk import chunk_pages

kept, _ = clean_pages(parse_all(Path("docs")))
seen, lines = set(), []
for c in chunk_pages(kept):
    key = (c["url"].replace("https://handbook.gitlab.com/handbook/finance/", ""),
           " > ".join(c["heading_path"].split(" > ")[:3]))
    if key not in seen:
        seen.add(key)
        lines.append(f"{key[0]} | {key[1]}")
Path("data").mkdir(exist_ok=True)
Path("data/outline.txt").write_text("\n".join(lines), encoding="utf-8")
print(len(lines), "outline lines written to data/outline.txt")
