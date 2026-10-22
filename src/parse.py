"""Stage 1: read a handbook .md file into a dict."""
import re
from pathlib import Path

BASE_URL = "https://handbook.gitlab.com/handbook/finance/"
FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
TITLE = re.compile(r'^title:\s*["\']?(.*?)["\']?\s*$', re.MULTILINE)


def page_url(rel: Path) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "_index":
        parts = parts[:-1]
    return BASE_URL + "/".join(parts) + ("/" if parts else "")


def parse_file(path: Path, docs_root: Path) -> dict:
    rel = path.relative_to(docs_root)
    text = path.read_text(encoding="utf-8-sig")
    title = rel.stem
    m = FRONT.match(text)
    if m:
        t = TITLE.search(m.group(1))
        if t and t.group(1):
            title = t.group(1)
        text = text[m.end():]
    return {"path": rel.as_posix(), "url": page_url(rel), "title": title, "body": text}


def parse_all(docs_root: Path) -> list[dict]:
    return [parse_file(p, docs_root) for p in sorted(docs_root.rglob("*.md"))]
