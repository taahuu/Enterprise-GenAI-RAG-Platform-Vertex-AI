"""Response post-processing: validate citations against the retrieved context."""
from __future__ import annotations

import re

from .prompt import REFUSAL

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def finalize(answer: str, n_blocks: int) -> tuple[str, list[int]]:
    """Strip citations that point at non-existent context blocks; return (clean_answer, cited_numbers)."""
    cited: set[int] = set()

    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = [n for n in nums if 1 <= n <= n_blocks]
        cited.update(valid)
        return f"[{', '.join(map(str, valid))}]" if valid else ""

    cleaned = _CITATION.sub(fix, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    if cleaned.startswith(REFUSAL[:40]):
        return REFUSAL, []
    return cleaned, sorted(cited)
