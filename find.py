import sys
from pathlib import Path

from src.parse import parse_all, BASE_URL
from src.clean import clean_pages
from src.chunk import chunk_pages

if len(sys.argv) < 2:
    raise SystemExit('usage: python find.py "keyword" [page-path-prefix]')
word = sys.argv[1].lower()
prefix = sys.argv[2] if len(sys.argv) > 2 else ""
chunks = chunk_pages(clean_pages(parse_all(Path("docs")))[0])
n = 0
for c in chunks:
    page = c["url"].replace(BASE_URL, "")
    low = c["text"].lower()
    if page.startswith(prefix) and word in low:
        n += 1
        i = low.index(word)
        snippet = c["text"][max(0, i - 80): i + 120].replace("\n", " ")
        print(f"{page} | {c['heading_path'].split(' > ', 1)[-1][:90]}")
        print(f"    ...{snippet}...")
print(n, "chunk(s)")
