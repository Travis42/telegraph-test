#!/usr/bin/env python3
"""Lowercase-cablese readability verification (paired with frozen cbl2).

Question: does the recommended lowercase rendering read at parity, like
the ALL-CAPS rendering cbl2 verified (recovery 1.09)?

Design (one variable changed vs cbl2):
- quiz items: REUSED verbatim from the frozen cbl2 ledger (same
  questions, same expected answers, same anchoring)
- writer/reader model + protocol: glm-5.3-flash via default_client,
  default thinking ON (matches cbl2; only the record casing differs)
- records: freshly written with the cablese prompts + lowercase suffix
- grading: grade_v2, the same pure grader behind the published v2
  accuracies

Outputs a ledger (writes + answers + grades) and summary.json with
A-FROM-CABLESE-LOWERCASE accuracy and ratios against the frozen cbl2
A-FROM-PLAIN (0.758) and A-FROM-CABLESE (0.825).

CLI: python3 -m ab.lc_readability --results-dir D [--limit N]
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

from ab.cablese import (
    RECORD_CABLESE_SYSTEM,
    RECORD_CABLESE_USER,
    ANSWER_FROM_RECORD_USER,
)
from ab.grade import grade_v2
from ab.instrument import call_model, default_client
from ab.run_v21 import DEFAULT_MODEL, QA_SYSTEM, TASKS2_DIR, load_tasks

CBL2_LEDGER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "runs", "cablese_cbl2", "ledger.jsonl")

LOWERCASE_SUFFIX = "Write the record in lowercase letters, not all caps."

# Frozen cbl2 v2 headline accuracies (regrade_v2.json, verified 24/24)
FROZEN_A_FROM_PLAIN = 0.758
FROZEN_A_FROM_CABLESE = 0.825


def load_cbl2_quiz_items(ledger_path=CBL2_LEDGER):
    """task_id -> list of {q_index, q, a} from the frozen run."""
    by_task = {}
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "quiz_item":
                continue
            by_task.setdefault(rec["task_id"], []).append({
                "q_index": rec.get("q_index"),
                "q": rec["question"],
                "a": rec["answer"],
            })
    for items in by_task.values():
        items.sort(key=lambda it: it["q_index"] if it["q_index"] is not None else 0)
    return by_task


def run(results_dir, limit=None, model=DEFAULT_MODEL):
    os.makedirs(results_dir, exist_ok=True)
    ledger_path = os.path.join(results_dir, "ledger.jsonl")
    summary_path = os.path.join(results_dir, "summary.json")

    tasks = load_tasks(TASKS2_DIR)
    if limit is not None:
        tasks = tasks[:limit]
    quiz = load_cbl2_quiz_items()
    task_ids = [t["id"] for t in tasks]
    missing = [tid for tid in task_ids if tid not in quiz]
    if missing:
        sys.exit(f"error: cbl2 ledger has no quiz items for: {missing[:5]}")

    client = default_client
    n_correct = n_total = n_anchored = n_correct_anchored = 0
    per_task = {}
    with open(ledger_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "type": "run_meta",
            "run_id": f"lc_readability_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model": model,
            "thinking": "default-on (cbl2 protocol)",
            "variant": "lowercase cablese",
            "tasks": len(tasks),
        }) + "\n")

        for ti, task in enumerate(tasks):
            tid = task["id"]
            # 1) write the lowercase cablese record (default thinking)
            system = RECORD_CABLESE_SYSTEM + " " + LOWERCASE_SUFFIX
            user = (RECORD_CABLESE_USER + " " + LOWERCASE_SUFFIX).format(
                passage=task["passage"])
            wres = call_model(client, model, system, user, 4000)
            record = wres["raw"]
            fh.write(json.dumps({
                "type": "record", "condition": "R-CABLESE-LOWERCASE",
                "task_id": tid, "text": record,
                "usage": wres["usage"], "latency_ms": wres["latency_ms"],
            }) + "\n")
            fh.flush()

            # 2) answer every frozen quiz item from the record
            t_correct = t_total = 0
            for item in quiz[tid]:
                ares = call_model(
                    client, model, QA_SYSTEM,
                    ANSWER_FROM_RECORD_USER.format(
                        record=record, question=item["q"]),
                    4000)
                g = grade_v2(task["passage"], item["q"], item["a"],
                             ares["raw"])
                n_total += 1
                t_total += 1
                if g["anchored"]:
                    n_anchored += 1
                if g["correct"]:
                    n_correct += 1
                    n_correct_anchored += 1
                    t_correct += 1
                fh.write(json.dumps({
                    "type": "answer", "condition": "A-FROM-CABLESE-LOWERCASE",
                    "task_id": tid, "q_index": item["q_index"],
                    "question": item["q"], "expected": item["a"],
                    "raw": ares["raw"], "grade": g,
                    "usage": ares["usage"],
                }) + "\n")
            fh.flush()
            per_task[tid] = {"correct": t_correct, "total": t_total}
            print(f"[{ti+1}/{len(tasks)}] {tid}: {t_correct}/{t_total}",
                  flush=True)

    acc = n_correct / n_total if n_total else None
    summary = {
        "condition": "A-FROM-CABLESE-LOWERCASE",
        "model": model,
        "n_items": n_total,
        "n_anchored": n_anchored,
        "accuracy_v2": round(acc, 4) if acc is not None else None,
        "frozen_A_FROM_PLAIN": FROZEN_A_FROM_PLAIN,
        "frozen_A_FROM_CABLESE": FROZEN_A_FROM_CABLESE,
        "recovery_vs_plain": round(acc / FROZEN_A_FROM_PLAIN, 4)
            if acc is not None else None,
        "recovery_vs_cablese_caps": round(acc / FROZEN_A_FROM_CABLESE, 4)
            if acc is not None else None,
        "per_task": per_task,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"},
                     indent=2), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="cap the task list (default: all 50)")
    args = p.parse_args(argv)
    run(args.results_dir, args.limit)


if __name__ == "__main__":
    main()
