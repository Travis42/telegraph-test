"""argparse CLI entry: encode / decode / verify / tables / demo."""

import argparse
import json
import sys

from . import tables as tables_mod
from . import tokencount
from .decode import UnknownCodeError, decode_dictionary, describe_register_loss
from .encode import encode_router, render_payload


def _read_text(args):
    if args.text is not None:
        return args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read()
    return sys.stdin.read()


def _cmd_encode(args):
    text = _read_text(args)
    payload = encode_router(text)
    if args.flat:
        print(payload["coded"])
    else:
        print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


def _cmd_decode(args):
    text = _read_text(args)
    tables = tables_mod.load_tables()
    try:
        print(decode_dictionary(text, tables, strict=True))
    except UnknownCodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_verify(args):
    from .verify import main as verify_main

    return verify_main(
        ["--text", _read_text(args)]
        + (["--strict"] if args.strict else [])
        + (["--loss-report", args.loss_report] if args.loss_report else [])
    )


def _cmd_tables(args):
    tables = tables_mod.load_tables()
    rules = tables_mod.load_rules()
    for table in tables:
        print(f"{table['name']}: {len(table['entries'])} entries")
        print(f"  provenance: {table['provenance']}")
    print(f"rules: {len(rules['derivations'])} derivations, "
          f"{len(rules['drop_articles'])} article drops, "
          f"{len(rules['drop_copulas'])} copula drops")
    return 0


def _cmd_demo(args):
    text = _read_text(args)
    payload = encode_router(text)
    restored = decode_dictionary(payload["coded"], tables_mod.load_tables())
    orig_count = tokencount.approx_count_labeled(text)
    coded_count = tokencount.approx_count_labeled(payload["coded"])
    print(render_payload(payload))
    print(describe_register_loss(payload["layers"]["register"]))
    print(f"approx tokens original: {orig_count['count']} (approx={orig_count['approx']})")
    print(f"approx tokens coded:    {coded_count['count']} (approx={coded_count['approx']})")
    print(f"script decode == original: {restored == text}")
    print("note: savings reported by ab/run_ab.py use API usage fields, not this counter")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tcb", description="Token Codebook MVP"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("encode", help="encode text (dictionary + register)")
    p.add_argument("--text"); p.add_argument("--file")
    p.add_argument("--flat", action="store_true", help="print coded text only")
    p.set_defaults(func=_cmd_encode)

    p = sub.add_parser("decode", help="deterministic decode (dictionary layer)")
    p.add_argument("--text"); p.add_argument("--file")
    p.set_defaults(func=_cmd_decode)

    p = sub.add_parser("verify", help="round-trip verification (exit 0/1/2)")
    p.add_argument("--text"); p.add_argument("--file")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--loss-report", metavar="PATH")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("tables", help="list codebook tables")
    p.set_defaults(func=_cmd_tables)

    p = sub.add_parser("demo", help="end-to-end demo transcript for a passage")
    p.add_argument("--text"); p.add_argument("--file")
    p.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
