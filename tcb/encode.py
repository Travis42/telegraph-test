"""Dictionary and register encoders plus the composite router.

Determinism: all entry iteration is sorted (longest expansion first, then
casefolded expansion, then code); regex alternation therefore prefers the
longest match at any position.

Round-trip guarantee (dictionary layer): a span is substituted only when it
is byte-identical to the canonical table expansion, so table-driven decode
restores exactly the original bytes.
"""

import re

from . import tokencount
from .__init__ import PAYLOAD_VERSION


def _sorted_entries(tables):
    entries = []
    for _, entry in sorted(
        ((name, entry) for name, entry in _iter_table_entries(tables)),
        key=lambda pair: (
            -len(pair[1]["expansion"]),
            pair[1]["expansion"].casefold(),
            pair[1]["code"].casefold(),
        ),
    ):
        entries.append(entry)
    return entries


def _iter_table_entries(tables):
    for table in sorted(tables, key=lambda t: t["name"]):
        for entry in table["entries"]:
            yield table["name"], entry


def _expansion_pattern(entries):
    if not entries:
        return None
    parts = [
        r"(?<!\w)" + re.escape(entry["expansion"]) + r"(?!\w)" for entry in entries
    ]
    return re.compile("|".join(parts), re.IGNORECASE)


def encode_dictionary(text, tables):
    """Longest-match-first whole-word/phrase replacement of expansions by codes.

    Economy guard: an entry is applied only when the approximate token cost
    of its code is strictly lower than that of the matched span.

    Returns (coded_text, report) where report = {"applied": [...], "skipped": [...]}.
    """
    entries = _sorted_entries(tables)
    lookup = {entry["expansion"].casefold(): entry for entry in entries}
    pattern = _expansion_pattern(entries)
    coded_parts = []
    applied, skipped = [], []
    pos = 0
    if pattern is not None:
        for match in pattern.finditer(text):
            span = match.group(0)
            entry = lookup[span.casefold()]
            if span != entry["expansion"]:
                # Case variant: skipping keeps decode(encode(x)) == x.
                continue
            if tokencount.approx_count(entry["code"]) >= tokencount.approx_count(
                entry["expansion"]
            ):
                skipped.append(
                    {
                        "code": entry["code"],
                        "expansion": entry["expansion"],
                        "reason": "economy",
                    }
                )
                continue
            coded_parts.append(text[pos : match.start()])
            coded_parts.append(entry["code"])
            pos = match.end()
            applied.append(
                {
                    "code": entry["code"],
                    "expansion": entry["expansion"],
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    coded_parts.append(text[pos:])
    return "".join(coded_parts), {"applied": applied, "skipped": skipped}


_WORD_RE = re.compile(r"\w+")


def _looks_verbish(word, rules):
    folded = word.casefold()
    if folded in rules["copula_next_verb_stoplist"]:
        return True
    return any(folded.endswith(suf) for suf in rules["copula_next_verb_suffixes"])


def encode_register(text, rules):
    """Deterministic register rules: drop articles; drop copulas only when
    conservatively followed by a non-verb-looking token; apply documented
    vowel-drop derivations. Punctuation and contractions preserved.

    Returns (coded_text, loss_report).
    """
    articles = {a.casefold() for a in rules["drop_articles"]}
    copulas = {c.casefold() for c in rules["drop_copulas"]}
    derivations = {d["word"].casefold(): d for d in rules["derivations"]}
    words = list(_WORD_RE.finditer(text))
    drop_spans = []
    derive_spans = []
    for i, match in enumerate(words):
        word = match.group(0)
        folded = word.casefold()
        if word in articles:
            drop_spans.append((match.start(), match.end(), word, "article"))
            continue
        if word in copulas:
            nxt = words[i + 1].group(0) if i + 1 < len(words) else None
            if nxt is not None and not _looks_verbish(nxt, rules):
                drop_spans.append((match.start(), match.end(), word, "copula"))
                continue
            # Uncertain: keep the word (conservative).
            continue
        if folded in derivations:
            d = derivations[folded]
            if tokencount.approx_count(d["derived"]) < tokencount.approx_count(word):
                derive_spans.append((match.start(), match.end(), word, d["derived"]))
                continue
    out_parts = []
    pos = 0
    replacements = sorted(drop_spans + derive_spans, key=lambda s: s[0])
    for start, end, word, replacement in replacements:
        out_parts.append(text[pos:start])
        if replacement in ("article", "copula"):
            out_parts.append("")  # drop; collapse doubled spaces below
        else:
            out_parts.append(replacement)
        pos = end
    out_parts.append(text[pos:])
    coded = "".join(out_parts)
    coded = re.sub(r"[ \t]{2,}", " ", coded)
    dropped = [
        {"token": w, "kind": k, "start": s, "end": e}
        for s, e, w, k in sorted(drop_spans, key=lambda s: s[0])
    ]
    derived = [
        {"from": w, "to": r, "start": s, "end": e}
        for s, e, w, r in sorted(derive_spans, key=lambda s: s[0])
    ]
    loss_report = {
        "dropped": dropped,
        "derived": derived,
        "flags": {
            "articles_dropped": sum(1 for d in dropped if d["kind"] == "article"),
            "copulas_dropped": sum(1 for d in dropped if d["kind"] == "copula"),
            "words_derived": len(derived),
            "lossy": bool(dropped),
            "derivations_recoverable": True,
            "drops_recoverable": False,
        },
    }
    return coded, loss_report


def encode_router(text, tables=None, rules=None):
    """Dictionary layer first, then register layer.

    Returns a composite payload dict with version stamp and per-layer reports.
    """
    if tables is None or rules is None:
        from . import tables as tables_mod

        if tables is None:
            tables = tables_mod.load_tables()
        if rules is None:
            rules = tables_mod.load_rules()
    dict_coded, dict_report = encode_dictionary(text, tables)
    reg_coded, reg_report = encode_register(dict_coded, rules)
    return {
        "version": PAYLOAD_VERSION,
        "original": text,
        "coded": reg_coded,
        "layers": {
            "dictionary": dict_report,
            "register": reg_report,
        },
    }


def render_payload(payload):
    """Text serialization with the version stamp header line."""
    import json

    reports = json.dumps(payload["layers"], sort_keys=True)
    return (
        f"#{payload['version']}\n{payload['coded']}\n#tcb-report\n{reports}\n"
    )
