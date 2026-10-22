"""Stage 3: split each page into heading-aware chunks."""
import re

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
MAX_CHARS = 1800    # soft ceiling per chunk
SOFT_CHARS = 1000   # prefer to break at a blank line after this size
OVERLAP = 200       # characters carried into the next chunk when a section is split
MIN_BODY = 40       # drop chunks shorter than this


def split_sections(text: str, page_title: str):
    """Return [(heading_path, body)] following # headings (ignores # inside code fences)."""
    sections, stack, current = [], [], []
    in_code = False

    def flush():
        body = "\n".join(current).strip()
        if body:
            path = " > ".join([page_title] + [t for _, t in stack])
            sections.append((path, body))

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            flush()
            current.clear()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            current.append(line)
    flush()
    return sections


def split_long(body: str):
    if len(body) <= MAX_CHARS:
        return [body]
    chunks, buf, size, fresh = [], [], 0, False
    for line in body.split("\n"):
        at_break = not line.strip()
        if buf and (size + len(line) > MAX_CHARS or (at_break and size >= SOFT_CHARS)):
            chunks.append("\n".join(buf).strip())
            tail, tsize = [], 0
            for prev in reversed(buf):
                if tsize + len(prev) > OVERLAP:
                    break
                tail.insert(0, prev)
                tsize += len(prev) + 1
            buf, size, fresh = tail, tsize, False
        buf.append(line)
        size += len(line) + 1
        if line.strip():
            fresh = True
    if fresh:
        chunks.append("\n".join(buf).strip())
    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    out = []
    for p in pages:
        n = 0
        for path, body in split_sections(p["text"], p["title"]):
            for piece in split_long(body):
                if len(piece) < MIN_BODY:
                    continue
                out.append({
                    "id": f"{p['path'].replace('/', '__')}__{n:03d}",
                    "url": p["url"],
                    "title": p["title"],
                    "heading_path": path,
                    "text": piece,
                })
                n += 1
    return out
