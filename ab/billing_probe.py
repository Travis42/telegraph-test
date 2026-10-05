#!/usr/bin/env python3
"""Records-only billing probe (supplementary to the cablese experiment).

Purpose: measure the BILLED completion tokens of the two write conditions
(R-PLAIN, R-CABLESE) with thinking disabled, against the same 50-passage
bank and the same prompts as ab/cablese.py. The cbl2-era headline run went
out with GLM default thinking ON, so its usage_completion_tokens include
invisible reasoning; this probe supplies the thinking-off billing basis
the frozen ledgers cannot.

Reuses the pipeline's own call path (prompts, client, call_model,
_approx_tokens, load_tasks). Writes a ledger.jsonl with per-call usage
including reasoning_tokens, plus summary.json with storage-basis and
billing-basis savings.

Refuses to run unless TCB_REASONING_OFF=1 — a thinking-on billing probe
would just reproduce the cbl2 divergence.

CLI:
  python3 -m ab.billing_probe [--results-dir D] [--passages N] [--model M]
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

from ab.cablese import (
    RECORD_PLAIN_SYSTEM,
    RECORD_PLAIN_USER,
    RECORD_CABLESE_SYSTEM,
    RECORD_CABLESE_USER,
    _approx_tokens,
)
from ab.expAB import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from ab.instrument import call_model, default_client
from ab.run_v21 import TASKS2_DIR, load_tasks

CONDITIONS = (
    ("R-PLAIN", RECORD_PLAIN_SYSTEM, RECORD_PLAIN_USER),
    ("R-CABLESE", RECORD_CABLESE_SYSTEM, RECORD_CABLESE_USER),
)


def median_or_none(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def run(results_dir, passages, model):
    if os.environ.get("TCB_REASONING_OFF") != "1":
        sys.exit(
            "error: billing_probe measures the thinking-off billing basis; "
            "export TCB_REASONING_OFF=1 before running."
        )
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ledger_path = os.path.join(results_dir, "ledger.jsonl")
    summary_path = os.path.join(results_dir, "summary.json")

    tasks = load_tasks(TASKS2_DIR)
    if passages is not None:
        tasks = tasks[:passages]

    client = default_client
    per_cond = {cond: [] for cond, _, _ in CONDITIONS}
    with open(ledger_path, "w", encoding="utf-8") as fh:
        meta = {
            "type": "run_meta",
            "run_id": f"billing_probe_{stamp}",
            "model": model,
            "reasoning_off": True,
            "tasks": len(tasks),
        }
        fh.write(json.dumps(meta) + "\n")

        for task in tasks:
            for cond, system, user_tmpl in CONDITIONS:
                started = time.monotonic()
                result = call_model(
                    client, model, system,
                    user_tmpl.format(passage=task["passage"]),
                    DEFAULT_MAX_TOKENS,
                )
                rec = {
                    "type": "billing_record",
                    "task_id": task["id"],
                    "condition": cond,
                    "approx_tokens": _approx_tokens(result["raw"]),
                    "usage": result["usage"],  # includes reasoning_tokens
                    "latency_ms": round(
                        (time.monotonic() - started) * 1000.0, 3),
                    "text_chars": len(result["raw"]),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                per_cond[cond].append(rec)
                print(f"{task['id']} {cond}: approx="
                      f"{rec['approx_tokens']} billed="
                      f"{result['usage']['completion_tokens']} "
                      f"reasoning="
                      f"{result['usage'].get('reasoning_tokens')}",
                      flush=True)

    def stats(cond):
        recs = per_cond[cond]
        return {
            "n": len(recs),
            "approx_tokens_median": median_or_none(
                [r["approx_tokens"] for r in recs]),
            "billed_completion_median": median_or_none(
                [r["usage"]["completion_tokens"] for r in recs]),
            "reasoning_tokens_median": median_or_none(
                [r["usage"].get("reasoning_tokens") for r in recs]),
            "latency_ms_median": median_or_none(
                [r["latency_ms"] for r in recs]),
        }

    plain, cablese = stats("R-PLAIN"), stats("R-CABLESE")

    def savings(a, b, key):
        if a.get(key) and b.get(key):
            return round(100.0 * (1.0 - b[key] / a[key]), 1)
        return None

    summary = {
        "run_id": meta["run_id"],
        "model": model,
        "reasoning_off": True,
        "R-PLAIN": plain,
        "R-CABLESE": cablese,
        "savings_pct_storage_basis": savings(
            plain, cablese, "approx_tokens_median"),
        "savings_pct_billing_basis": savings(
            plain, cablese, "billed_completion_median"),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--passages", type=int, default=None,
                   help="cap the task list (default: all)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)
    run(args.results_dir, args.passages, args.model)


if __name__ == "__main__":
    main()
