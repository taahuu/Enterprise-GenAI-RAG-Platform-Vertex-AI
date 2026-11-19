"""Small, dependency-free text helpers shared by retrieval, hashing embeddings and the extractive LLM."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being it its this
    that these those do does did can could should would will shall may might must i you he she we they them
    our your their my me what which who whom whose when where why how about into over under than then so such
    not no yes per any all each""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "edly", "ed", "ies", "es", "s")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def stem(token: str) -> str:
    """Very light suffix stripper (good enough for lexical matching)."""
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)] + ("y" if suf == "ies" else "")
    return token


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks]


def split_sentences(text: str, max_unit: int = 240) -> list[str]:
    """Split into answer-sized units: one per line; long lines are split into sentences.

    Short lines such as "SEV1: outage. Respond within 15 minutes." stay together. Headings and
    questions (units with fewer than 4 words, or ending in '?') are dropped.
    """
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip(" -*#\t")
        if not line:
            continue
        units.extend(_SENTENCE.split(line) if len(line) > max_unit else [line])
    return [u.strip() for u in units if len(u.split()) >= 4 and not u.rstrip().endswith("?")]
