"""Deterministic script-side decoder.

The dictionary layer is inverted table-driven (longest code first). Codes
that look like codes but resolve to no table entry raise UnknownCodeError —
they are never silently dropped or passed through.

The register layer is lossy and not byte-inverted; describe_register_loss()
summarizes what the loss report says, for consumption by verify.
"""

import re

from .__init__ import PAYLOAD_VERSION


class UnknownCodeError(Exception):
    def __init__(self, token):
        self.token = token
        super().__init__(f"unknown code token: {token!r}")


_CODE_LIKE_RE = re.compile(r"^[A-Z]{2,5}$|^\d{2,3}$")


def _split_payload(coded):
    if coded.startswith(f"#{PAYLOAD_VERSION}\n"):
        rest = coded[len(f"#{PAYLOAD_VERSION}\n"):]
        lines = rest.split("\n", 1)
        if len(lines) == 2 and lines[1].startswith("#tcb-report"):
            return lines[0]
        return rest
    return coded


def decode_dictionary(coded, tables, strict=True):
    """Inverse of encode_dictionary. strict=True raises UnknownCodeError for
    standalone code-shaped tokens that appear in no table (never silent)."""
    coded = _split_payload(coded)
    entries = []
    for table in sorted(tables, key=lambda t: t["name"]):
        for entry in table["entries"]:
            entries.append(entry)
    entries.sort(key=lambda e: (-len(e["code"]), e["code"].casefold()))
    if not entries:
        return coded
    parts = [r"(?<!\w)" + re.escape(e["code"]) + r"(?!\w)" for e in entries]
    pattern = re.compile("|".join(parts))
    out_parts = []
    pos = 0
    for match in pattern.finditer(coded):
        span = match.group(0)
        resolved = None
        for entry in entries:
            if entry["code"] == span or (
                not entry["case_sensitive"]
                and entry["code"].casefold() == span.casefold()
            ):
                resolved = entry
                break
        if resolved is None:
            if strict:
                raise UnknownCodeError(span)
            continue
        out_parts.append(coded[pos : match.start()])
        out_parts.append(resolved["expansion"])
        pos = match.end()
    out_parts.append(coded[pos:])
    result = "".join(out_parts)
    if strict:
        for word in _word_tokens(result):
            stripped = word.strip(".,;:!?\"'()[]")
            if _CODE_LIKE_RE.match(stripped):
                known = any(
                    e["code"].casefold() == stripped.casefold()
                    or e["expansion"].casefold() == stripped.casefold()
                    for e in entries
                )
                if not known:
                    raise UnknownCodeError(stripped)
    return result


def _word_tokens(text):
    return re.findall(r"\S+", text)


def describe_register_loss(loss_report):
    """Human-readable flags-and-loss summary of the register layer."""
    flags = loss_report["flags"]
    dropped = loss_report["dropped"]
    derived = loss_report["derived"]
    lines = [
        "register layer loss report:",
        f"  articles dropped: {flags['articles_dropped']}",
        f"  copulas dropped: {flags['copulas_dropped']}",
        f"  words derived (recoverable via rules.json table): {flags['words_derived']}",
        f"  lossy: {flags['lossy']}",
    ]
    if dropped:
        toks = ", ".join(f"{d['token']}@{d['start']}" for d in dropped)
        lines.append(f"  dropped tokens: {toks}")
    if derived:
        toks = ", ".join(f"{d['from']}->{d['to']}@{d['start']}" for d in derived)
        lines.append(f"  derived tokens: {toks}")
    return "\n".join(lines)
