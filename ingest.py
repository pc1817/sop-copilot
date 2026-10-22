from pathlib import Path
from src.parse import parse_all
from src.clean import clean_pages

pages = parse_all(Path("docs"))
kept, dropped = clean_pages(pages)
print("parsed:", len(pages), "kept:", len(kept), "dropped:", len(dropped))
for d in dropped:
    print("  dropped:", d)
print("total chars kept:", sum(len(p["text"]) for p in kept))
for p in kept[:3]:
    print(p["url"], "|", p["title"])
