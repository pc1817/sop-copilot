from pathlib import Path
from src.parse import parse_all
from src.clean import clean_pages
from src.chunk import chunk_pages, MAX_CHARS

pages = parse_all(Path("docs"))
kept, dropped = clean_pages(pages)
chunks = chunk_pages(kept)

print("parsed:", len(pages), "kept:", len(kept), "dropped:", len(dropped))
sizes = [len(c["text"]) for c in chunks]
print("chunks:", len(chunks), "| avg chars:", sum(sizes) // len(sizes),
      "| min:", min(sizes), "| max:", max(sizes), "| over soft max:", sum(s > MAX_CHARS for s in sizes))
ids = [c["id"] for c in chunks]
print("unique ids:", len(set(ids)) == len(ids))
for c in chunks[:2] + chunks[len(chunks) // 2:len(chunks) // 2 + 1]:
    print("---", c["id"], "|", c["heading_path"])
    print(c["text"][:200].replace("\n", " "))
