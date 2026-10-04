"""Round-trip verification and exit codes.

Exit codes:
  0 — dictionary layer round-trips; register loss report emitted.
  1 — round-trip failure (or --strict economy-skip failure).
  2 — structural error (e.g., UnknownCodeError while decoding).
"""

import difflib
import json
import os

from . import tables as tables_mod
from .decode import UnknownCodeError, decode_dictionary, describe_register_loss
from .encode import encode_dictionary, encode_register


def _diff_summary(original, restored):
    diff = list(
        difflib.unified_diff(
            original.splitlines() or [original],
            restored.splitlines() or [restored],
            lineterm="",
            n=0,
        )
    )
    return "\n".join(diff[:12])


def verify_text(text, tables=None, rules=None, strict=False):
    """Returns (exit_code, message, loss_report)."""
    if tables is None:
        tables = tables_mod.load_tables()
    if rules is None:
        rules = tables_mod.load_rules()
    try:
        dict_coded, dict_report = encode_dictionary(text, tables)
        restored = decode_dictionary(dict_coded, tables, strict=True)
    except UnknownCodeError as exc:
        return 2, f"structural error: {exc}", None
    if restored != text:
        return (
            1,
            "round-trip failure (dictionary layer):\n"
            + _diff_summary(text, restored),
            None,
        )
    if strict and dict_report["skipped"]:
        skipped = ", ".join(
            f"{s['expansion']}->{s['code']}" for s in dict_report["skipped"]
        )
        return 1, f"strict mode: dictionary substitutions skipped for economy: {skipped}", None
    reg_coded, loss_report = encode_register(text, rules)
    msg = "dictionary layer round-trip: OK\n" + describe_register_loss(loss_report)
    return 0, msg, loss_report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m tcb verify", description="round-trip verification"
    )
    parser.add_argument("--text", help="input text (default: read --file or stdin)")
    parser.add_argument("--file", help="input file")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--loss-report", metavar="PATH", help="write loss report JSON here")
    args = parser.parse_args(argv)

    if args.text is not None:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        import sys

        text = sys.stdin.read()

    code, message, loss_report = verify_text(text, strict=args.strict)
    print(message)
    if args.loss_report and loss_report is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.loss_report)), exist_ok=True)
        with open(args.loss_report, "w", encoding="utf-8") as fh:
            json.dump(loss_report, fh, sort_keys=True, indent=2)
        print(f"loss report written to {args.loss_report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
