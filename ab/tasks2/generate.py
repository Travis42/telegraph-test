#!/usr/bin/env python3
"""Task bank v2 — fact-shaped class regeneration (SPEC-V21 §2).

Generation-verification loop, per passage candidate:
  1. generate passage + candidate fact list (one model call, JSON output);
  2. verify every fact is stated in the passage (fact_check, raw logged);
  3. accept ONLY if the candidate facts parse AND >= MIN_VERIFY_RATIO of
     facts verify against the generated passage.

No silent fallbacks: every model answer is written VERBATIM to a JSONL
generation log before any parsing; rejected candidates are logged with
their reason (unparseable / verify-ratio-below-threshold), never silently
dropped or repaired by hand.

The shipped bank files were hand-authored offline (see README.md); this
script regenerates/replaces them with model-generated material when a real
API key is available. Requires ZAI_API_KEY.
"""

import argparse
import datetime
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ab.instrument import (  # noqa: E402,E501
    call_model,
    default_client,
    fact_check,
    parse_json_block,
)

MIN_PER_CLASS = 6
MIN_VERIFY_RATIO = 0.60
MAX_CANDIDATES_PER_CLASS = 20
LENGTH_SPEC = {
    "medium": "~800 characters",
    "long": "~1500-2000 characters",
}

GEN_SYSTEM = (
    "You write 19th-century-style factual reports (shipping, railways, "
    "lighthouses, mining, telegraphy) and a matching fact list. Output ONLY "
    "a JSON object with keys \"passage\" (string) and \"facts\" (array of "
    "short English sentences, each a discrete fact literally stated in the "
    "passage). 10-15 facts for a medium passage, 20-30 for a long one."
)

_PREFIX = {"medium": "med", "long": "long"}

SEEDS = {
    "medium": [
        "a steamship coal run",
        "a lighthouse inspection",
        "a copper shipment dispute",
        "a railway bridge repair",
        "a tea clipper race",
        "a mine flooding incident",
        "a telegraph cable fault",
        "a pilot station inquiry",
    ],
    "long": [
        "a wreck commission inquiry",
        "a merchant house correspondence",
        "an annual company report",
        "a railway survey dispatch",
        "a consular trade report",
        "a whaling voyage log",
        "a harbour works contract",
        "an emigration passage account",
    ],
}


def generate_candidate(client, model, len_class, seed, max_tokens, gen_log):
    user = (
        f"Write a {LENGTH_SPEC[len_class]} passage about {seed}, then the "
        "fact list. Remember: output only the JSON object with keys "
        "\"passage\" and \"facts\"."
    )
    result = call_model(
        client, model, GEN_SYSTEM, user, max_tokens, parse=parse_json_block
    )
    record = {
        "type": "generation_call",
        "len_class": len_class,
        "seed": seed,
        "raw": result["raw"],
        "usage": result["usage"],
        "latency_ms": result["latency_ms"],
    }
    _append(gen_log, record)
    obj = result["parsed"]
    if result["parse_error"] is not None:
        _append(
            gen_log,
            {"type": "candidate_rejected", "len_class": len_class,
             "seed": seed, "reason": f"parse: {result['parse_error']}"},
        )
        return None
    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("passage"), str)
        or not isinstance(obj.get("facts"), list)
        or not obj["facts"]
        or not all(isinstance(f, str) for f in obj["facts"])
    ):
        _append(
            gen_log,
            {"type": "candidate_rejected", "len_class": len_class,
             "seed": seed,
             "reason": "schema: expected {passage: str, facts: [str]}"},
        )
        return None
    return obj


def verify_candidate(client, model, candidate, len_class, seed, max_tokens, gen_log):
    check = fact_check(
        client, model, candidate["passage"], candidate["facts"], max_tokens
    )
    _append(
        gen_log,
        {"type": "verification_call", "len_class": len_class, "seed": seed,
         "raw": check["raw"], "facts_total": check["facts_total"],
         "parse_ok": check["parse_ok"], "parse_error": check.get("parse_error"),
         "facts_matched": check["facts_matched"]},
    )
    if not check["parse_ok"]:
        _append(
            gen_log,
            {"type": "candidate_rejected", "len_class": len_class,
             "seed": seed, "reason": f"verification parse: {check['parse_error']}"},
        )
        return None, check
    ratio = len(check["facts_matched"]) / check["facts_total"]
    if ratio < MIN_VERIFY_RATIO:
        _append(
            gen_log,
            {"type": "candidate_rejected",
             "len_class": len_class, "seed": seed,
             "reason": f"verify ratio {ratio:.2f} < {MIN_VERIFY_RATIO}"},
        )
        return None, check
    _append(
        gen_log,
        {"type": "candidate_accepted", "len_class": len_class,
         "seed": seed, "verify_ratio": round(ratio, 3)},
    )
    return ratio, check


def _append(path, record):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def run_class(client, model, len_class, max_tokens, gen_log):
    accepted = []
    for seed in SEEDS[len_class][:MAX_CANDIDATES_PER_CLASS]:
        if len(accepted) >= MIN_PER_CLASS:
            break
        candidate = generate_candidate(
            client, model, len_class, seed, max_tokens, gen_log
        )
        if candidate is None:
            continue
        ratio, _ = verify_candidate(
            client, model, candidate, len_class, seed, max_tokens, gen_log
        )
        if ratio is None:
            continue
        accepted.append((seed, candidate, ratio))
    return accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("TCB_MODEL", "zai/glm-5.3-flash"))
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--out-dir", default=_HERE)
    parser.add_argument(
        "--results-dir",
        default=os.path.join(_ROOT, "results"),
    )
    args = parser.parse_args(argv)

    if not os.environ.get("ZAI_API_KEY"):
        print(
            "error: ZAI_API_KEY is not set; generation needs the real API.",
            file=sys.stderr,
        )
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_log = os.path.join(args.results_dir, f"tasks2_gen_{stamp}.jsonl")
    os.makedirs(args.results_dir, exist_ok=True)

    counts = {}
    for len_class in ("medium", "long"):
        accepted = run_class(
            default_client, args.model, len_class, args.max_tokens, gen_log
        )
        counts[len_class] = len(accepted)
        if len(accepted) < MIN_PER_CLASS:
            print(
                f"error: only {len(accepted)} accepted for {len_class} "
                f"(need {MIN_PER_CLASS}); see {gen_log}",
                file=sys.stderr,
            )
            return 1
        for i, (seed, candidate, ratio) in enumerate(accepted, 1):
            task = {
                "id": f"fact_{_PREFIX[len_class]}_{i:02d}",
                "register": "fact",
                "len_class": len_class,
                "passage": candidate["passage"],
                "question": f"According to the passage about {seed}, state the main fact.",  # placeholder — reviewed before use
                "expected_answer": candidate["facts"][0],
                "facts": candidate["facts"],
            }
            path = os.path.join(
                args.out_dir, f"fact_{_PREFIX[len_class]}_{i:02d}.json"
            )
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(task, fh, sort_keys=True, indent=2, ensure_ascii=False)
                fh.write("\n")
    print(f"generation log: {gen_log}")
    print(f"accepted per class: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
