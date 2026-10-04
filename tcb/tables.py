"""Codebook table loading and validation.

All iteration over table entries is sorted so that no dict-order leaks into
encoder/decoder behavior.
"""

import json
import os

CODEBOOK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "codebook"
)
TABLE_FILES = ["morse_abbr.json", "phillips.json", "qcode.json", "wu92.json"]
RULES_FILE = "rules.json"


class TableError(Exception):
    """Structural problem in a codebook table."""


def _validate_entry(entry, table_name):
    if not isinstance(entry, dict):
        raise TableError(f"{table_name}: entry is not an object: {entry!r}")
    for field in ("code", "expansion", "case_sensitive"):
        if field not in entry:
            raise TableError(f"{table_name}: entry missing field {field!r}: {entry!r}")
    if not isinstance(entry["code"], str) or not entry["code"]:
        raise TableError(f"{table_name}: bad code: {entry!r}")
    if not isinstance(entry["expansion"], str) or not entry["expansion"]:
        raise TableError(f"{table_name}: bad expansion: {entry!r}")
    if not isinstance(entry["case_sensitive"], bool):
        raise TableError(f"{table_name}: case_sensitive must be bool: {entry!r}")


def load_table(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    name = os.path.splitext(os.path.basename(path))[0]
    if not isinstance(raw.get("provenance"), str) or not raw["provenance"].strip():
        raise TableError(f"{name}: missing provenance string")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise TableError(f"{name}: missing or empty entries list")
    for entry in entries:
        _validate_entry(entry, name)
    seen = {}
    for entry in entries:
        key = entry["code"].casefold()
        if key in seen:
            raise TableError(f"{name}: duplicate code {entry['code']!r}")
        seen[key] = entry
    return {
        "name": name,
        "provenance": raw["provenance"],
        "entries": sorted(entries, key=lambda e: (e["code"].casefold(), e["expansion"])),
    }


def load_tables(directory=CODEBOOK_DIR):
    """Load and cross-validate all codebook tables (sorted, deterministic)."""
    tables = [load_table(os.path.join(directory, fn)) for fn in sorted(TABLE_FILES)]
    codes = {}      # casefolded code -> table name
    expansions = {} # casefolded expansion -> table name
    for table in tables:
        for entry in table["entries"]:
            code_key = entry["code"].casefold()
            exp_key = entry["expansion"].casefold()
            if code_key in codes:
                raise TableError(
                    f"code {entry['code']!r} appears in both {codes[code_key]} and {table['name']}"
                )
            codes[code_key] = table["name"]
            expansions.setdefault(exp_key, table["name"])
    for code_key, table_name in sorted(codes.items()):
        if code_key in expansions and expansions[code_key] != table_name:
            raise TableError(
                f"code {code_key!r} in {table_name} collides with an expansion in "
                f"{expansions[code_key]}"
            )
    return tables


def load_rules(directory=CODEBOOK_DIR):
    with open(os.path.join(directory, RULES_FILE), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw.get("provenance"), str) or not raw["provenance"].strip():
        raise TableError("rules: missing provenance string")
    for field in ("drop_articles", "drop_copulas", "derivations"):
        if field not in raw:
            raise TableError(f"rules: missing field {field!r}")
    raw["drop_articles"] = sorted(raw["drop_articles"])
    raw["drop_copulas"] = sorted(raw["drop_copulas"])
    raw["copula_next_verb_stoplist"] = sorted(raw.get("copula_next_verb_stoplist", []))
    raw["copula_next_verb_suffixes"] = sorted(raw.get("copula_next_verb_suffixes", []))
    raw["derivations"] = sorted(raw["derivations"], key=lambda d: d["word"])
    derived = [d["derived"].casefold() for d in raw["derivations"]]
    if len(derived) != len(set(derived)):
        raise TableError("rules: duplicate derived form")
    return raw


def all_entries(tables):
    """All entries from all tables in a deterministic global order."""
    entries = []
    for table in sorted(tables, key=lambda t: t["name"]):
        for entry in table["entries"]:
            entries.append((table["name"], entry))
    return entries
