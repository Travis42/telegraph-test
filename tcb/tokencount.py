"""Heuristic approximate token counter.

This is a word/subword approximation (BPE-ish), NOT the model tokenizer.
Every output is labeled approx=true. It is used only for encoder economy
checks and design-time feedback; savings reports always come from API
`usage` fields in the A/B runner.
"""

import re

_CHUNK_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


def approx_count(text):
    """Approximate token count: alphanumeric runs split into <=4-char subword
    pieces (ceil), each punctuation character counts as one token."""
    if not text:
        return 0
    total = 0
    for match in _CHUNK_RE.finditer(text):
        chunk = match.group(0)
        total += max(1, -(-len(chunk) // 4))
    return total


def approx_count_labeled(text):
    return {"count": approx_count(text), "approx": True}
