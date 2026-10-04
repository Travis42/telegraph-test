"""SPEC-VALIDATION: grader v2 — passage-anchored deterministic grading.

The truth path contains no LLM (knockout principle, Theory directive
2026-10-03). Correctness = normalized-token containment of the model
answer in the expected answer (same |A∩B|/min(|A|,|B|) formula as the
existing lexical machinery in ab/baseline.py), PLUS an anchor check:
the expected answer must itself be extractable from the source passage
under the same containment. Pure functions only — no network, no model
calls, fixed normalization (case, punctuation, digit canonicalization).
"""

import re

ANCHOR_THRESHOLD = 0.8
MATCH_THRESHOLD = 0.8

_ORDINAL_RE = re.compile(r"(\d)(?:st|nd|rd|th)\b")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_NONWORD_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text):
    """Fixed deterministic normalization: casefold, digit
    canonicalization (1,000 -> 1000; 3rd -> 3), punctuation to spaces,
    whitespace collapse."""
    s = text.casefold().strip()
    s = _THOUSANDS_RE.sub("", s)
    s = _ORDINAL_RE.sub(r"\1", s)
    s = _NONWORD_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def tokens(text):
    return frozenset(normalize(text).split())


def containment(tokens_a, tokens_b):
    """|A∩B| / min(|A|,|B|) — the ab.baseline.token_overlap formula,
    applied to pre-normalized token sets (0.0 on either side empty)."""
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def anchor_score(passage, expected_answer):
    return containment(tokens(expected_answer), tokens(passage))


def match_score(expected_answer, model_answer):
    return containment(tokens(model_answer), tokens(expected_answer))


def anchored(passage, expected_answer, threshold=ANCHOR_THRESHOLD):
    """True iff expected_answer is extractable from passage under the
    normalized-token containment (>= threshold)."""
    return anchor_score(passage, expected_answer) >= threshold


def grade_v2(passage, question, expected_answer, model_answer):
    """Pure grader: (passage, question, expected_answer, model_answer)
    -> {correct, anchored, diagnostics}. correct requires BOTH the
    anchor check and the answer-vs-expected containment; an unanchored
    item is invalid and never correct. `question` is carried for
    provenance only (grading never inspects it)."""
    a_score = anchor_score(passage, expected_answer)
    m_score = match_score(expected_answer, model_answer)
    is_anchored = a_score >= ANCHOR_THRESHOLD
    return {
        "correct": is_anchored and m_score >= MATCH_THRESHOLD,
        "anchored": is_anchored,
        "diagnostics": {
            "anchor_score": round(a_score, 6),
            "match_score": round(m_score, 6),
            "anchor_threshold": ANCHOR_THRESHOLD,
            "match_threshold": MATCH_THRESHOLD,
        },
    }
