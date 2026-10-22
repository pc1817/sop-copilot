"""Stage 2: strip Hugo/HTML noise, keep readable markdown."""
import re

SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
AUTOLINK = re.compile(r"<((?:https?://|mailto:)[^>\s]+|[^\s<>@]+@[^\s<>]+)>")
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
BLANKS = re.compile(r"\n{3,}")

MIN_CHARS = 400  # pages shorter than this are mostly link lists


def clean_text(body: str) -> str:
    t = HTML_COMMENT.sub("", body)
    t = SHORTCODE.sub("", t)
    t = IMAGE.sub("", t)
    t = LINK.sub(r"\1", t)
    t = AUTOLINK.sub(r"\1", t)   # keep emails/URLs such as ar@gitlab.com
    t = HTML_TAG.sub("", t)
    t = "\n".join(line.rstrip() for line in t.splitlines())
    return BLANKS.sub("\n\n", t).strip()


def clean_pages(pages: list[dict]):
    kept, dropped = [], []
    for p in pages:
        text = clean_text(p["body"])
        if len(text) < MIN_CHARS:
            dropped.append(p["path"])
        else:
            kept.append({**p, "text": text})
    return kept, dropped
