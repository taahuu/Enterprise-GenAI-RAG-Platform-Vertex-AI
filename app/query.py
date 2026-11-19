"""Small deterministic query rewrite helpers for identity-style questions."""
from __future__ import annotations

import re

_FILLER = re.compile(
    r"\b(give me|tell me|which is|that is|in this document|in the document|mention(?:ed)?|mentioning)\b",
    re.IGNORECASE,
)
_SPACE = re.compile(r"\s+")

_SYNONYMS = {
    "name": "full name candidate applicant author",
    "email": "email address contact",
    "phone": "phone number telephone contact",
    "date": "date time deadline",
    "address": "address location",
    "title": "title heading subject",
}
_IDENTITY_TERMS = {"name", "person", "who", "email", "phone", "date", "address", "title", "author", "candidate", "applicant"}


def rewrite_query(question: str) -> tuple[str, bool]:
    """Remove filler, add identity synonyms once, and identify identity questions."""
    text = _FILLER.sub(" ", question.strip())
    text = _SPACE.sub(" ", text).strip()
    if not text:
        text = question.strip()

    terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    is_identity = bool(terms & _IDENTITY_TERMS)
    if not is_identity:
        return text, False

    parts = set(re.findall(r"[a-z0-9]+", text.lower()))
    additions = []
    for term, expansion in _SYNONYMS.items():
        if term in parts and expansion not in text.lower():
            additions.append(expansion)
    rewritten = _SPACE.sub(" ", f"{text} {' '.join(additions)}".strip())
    return rewritten, True


def needs_rewrite(original: str, rewritten: str) -> bool:
    """Do not embed a rewrite that differs only by trailing punctuation."""
    strip = lambda s: s.strip().strip(".,;:!? ")
    return strip(original).casefold() != strip(rewritten).casefold()
