#!/usr/bin/env python3
"""SPEC-VALIDATION §3: retro-regrade an existing run with grader v2.

Standalone, offline, NO new API calls: reads the verbatim answers
(raw_answer), expected answers, and passages referenced by an existing
run's ledger (ledger.jsonl, or slices/slice_*.jsonl) and re-grades
every quiz answer with the passage-anchored deterministic grader
(ab/grade.py). The LLM judge is quarantined under diagnostics/ and the
45 judge-overturned misses are adjudicated by code (anchored match vs
passage), never by the judge verdict. Writes <run-dir>/regrade_v2/
(new summary + diagnostics) and NEVER touches the original
summary.json.

Passage resolution: a run-dir passages.json ({task_id: passage}) wins;
otherwise task ids are resolved against the passage bank (--tasks-dir,
default ab/tasks2). Unresolvable task ids are a hard error (a silent
wrong-passage regrade would be worse than no regrade).

CLI:
  python3 -m ab.regrade --run-dir results/baseline_or-c3-1255 \
      [--tasks-dir ab/tasks2]
"""

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ab.baseline import summarize_baseline, wilson_ci  # noqa: E402
from ab.cablese import (  # noqa: E402
    ANSWER_CONDITIONS,
    MCNEMAR_PAIRS,
    mcnemar_exact,
)
from ab.grade import grade_v2  # noqa: E402
from ab.run_v21 import TASKS2_DIR, load_tasks  # noqa: E402


class RegradeError(Exception):
    """Explicit regrade failure (missing run material)."""


def load_ledger_records(run_dir):
    """Collect every record from ledger.jsonl and/or slices/*.jsonl
    (deduped by content line — a dir may hold both)."""
    records, seen = [], set()
    paths = []
    ledger = os.path.join(run_dir, "ledger.jsonl")
    if os.path.isfile(ledger):
        paths.append(ledger)
    slices_dir = os.path.join(run_dir, "slices")
    if os.path.isdir(slices_dir):
        for name in sorted(os.listdir(slices_dir)):
            if name.startswith("slice_") and name.endswith(".jsonl"):
                paths.append(os.path.join(slices_dir, name))
    if not paths:
        raise RegradeError(f"no ledger.jsonl or slices/slice_*.jsonl "
                           f"under {run_dir}")
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                key = hashlib.sha256(
                    line.strip().encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                records.append(json.loads(line))
    return records


def resolve_passages(run_dir, tasks_dir):
    passages = {}
    local = os.path.join(run_dir, "passages.json")
    if os.path.isfile(local):
        with open(local, encoding="utf-8") as fh:
            passages = json.load(fh)
    else:
        for task in load_tasks(tasks_dir):
            passages[task["id"]] = task["passage"]
    return passages


def _sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def regrade_cablese(run_dir, tasks_dir=None):
    """Re-grade a cablese run's cond_answer records under grader v2,
    anchored to each record's SOURCE PASSAGE (same bank the cablese
    run used). Zero API calls — pure recompute from the ledger plus
    the passage bank. Writes <run-dir>/regrade_v2.json (report) and
    <run-dir>/regrade_v2/regraded_cond_answers.jsonl (per-record
    detail); NEVER touches the original summary.json (its sha256 is
    recorded in the report for tamper-evidence). Returns the report."""
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise RegradeError(f"run dir not found: {run_dir}")
    records = load_ledger_records(run_dir)
    cond_answers = [r for r in records
                    if r.get("type") == "cond_answer"]
    if not cond_answers:
        raise RegradeError(f"no cond_answer records under {run_dir}")
    rec_recs = [r for r in records if r.get("type") == "record"]

    passages = resolve_passages(run_dir, tasks_dir or TASKS2_DIR)
    missing = sorted({r["task_id"] for r in cond_answers
                      if r["task_id"] not in passages})
    if missing:
        raise RegradeError(
            f"cannot resolve passages for {len(missing)} task id(s): "
            f"{missing[:5]}... (provide run-dir passages.json or "
            f"--tasks-dir)")

    regraded, anchoring_rejections = [], 0
    for rec in cond_answers:
        g = grade_v2(passages[rec["task_id"]], rec.get("question", ""),
                     rec["expected"], rec["raw_answer"])
        out = dict(rec)
        out["anchored"] = g["anchored"]
        out["grade_v2"] = g["diagnostics"]
        out["correct"] = g["correct"]  # v2 verdict REPLACES the stored one
        regraded.append(out)
        if not g["anchored"]:
            anchoring_rejections += 1

    conditions = {}
    for cond in ANSWER_CONDITIONS:
        answers = [r for r in regraded if r["condition"] == cond]
        k = sum(1 for r in answers if r["correct"])
        n = len(answers)
        conditions[cond] = {
            "n": n,
            "correct": k,
            "v2_accuracy": round(k / n, 6) if n else None,
            "wilson_ci": wilson_ci(k, n),
        }

    def _acc(cond):
        return conditions[cond]["v2_accuracy"]

    def _ratio(num, den):
        if num is None or den in (None, 0):
            return None
        return round(num / den, 6)

    def _hits(cond):
        return {(r["task_id"], r["q_index"]): r["correct"]
                for r in regraded if r["condition"] == cond}

    mcnemar = {f"{x}_vs_{y}": mcnemar_exact(_hits(x), _hits(y))
               for x, y in MCNEMAR_PAIRS}

    # record-axis token accounting (independent confirmation of the
    # arithmetic savings number, from the record ledger entries)
    token_means = {}
    for cond in ("R-PLAIN", "R-CABLESE"):
        toks = [r["approx_tokens"] for r in rec_recs
                if r.get("condition") == cond
                and r.get("approx_tokens") is not None]
        token_means[cond] = {
            "n_records": len(toks),
            "mean_approx_tokens": (round(sum(toks) / len(toks), 6)
                                   if toks else None),
        }
    mean_plain = token_means["R-PLAIN"]["mean_approx_tokens"]
    mean_cablese = token_means["R-CABLESE"]["mean_approx_tokens"]
    savings_pct_v2 = (round(100.0 * (1.0 - mean_cablese / mean_plain), 6)
                      if mean_plain and mean_cablese is not None else None)

    summary_path = os.path.join(run_dir, "summary.json")
    report = {
        "run_id": os.path.basename(run_dir) + "_regrade_v2",
        "source_run_dir": run_dir,
        "conditions": conditions,
        "anchoring_rejections": anchoring_rejections,
        "recovery_ratio_v2": _ratio(_acc("A-FROM-CABLESE"),
                                    _acc("A-FROM-PLAIN")),
        "decode_ratio_v2": _ratio(_acc("DECODE"),
                                  _acc("A-PLAIN-DIRECT")),
        "mcnemar_v2": mcnemar,
        "record_tokens_v2": {
            "R-PLAIN": token_means["R-PLAIN"],
            "R-CABLESE": token_means["R-CABLESE"],
            "savings_pct_v2": savings_pct_v2,
            "note": ("mean approx_tokens per record entry, recomputed "
                     "from the ledger; no new API calls"),
        },
        "original_summary_sha256": (
            _sha256_file(summary_path)
            if os.path.isfile(summary_path) else None),
        "answers_regraded": len(regraded),
        "note": ("grader v2 verdicts only (passage-anchored, "
                 "deterministic); no LLM judges in the truth path; "
                 "no new API calls were made"),
    }

    out_dir = os.path.join(run_dir, "regrade_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "regraded_cond_answers.jsonl"), "w",
              encoding="utf-8") as fh:
        for rec in regraded:
            fh.write(json.dumps(rec, sort_keys=True,
                                ensure_ascii=False) + "\n")
    report_path = os.path.join(run_dir, "regrade_v2.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, indent=2,
                  ensure_ascii=False)
        fh.write("\n")
    print(f"[regrade] {len(regraded)} cond_answers re-graded with "
          f"grader v2 -> {report_path}")
    return report


def regrade_run(run_dir, tasks_dir=None):
    """Re-grade an existing run dir with grader v2. Auto-detects the
    record mix: quiz_answer records take the baseline path (writes
    <run-dir>/regrade_v2/), cond_answer records take the cablese path
    (writes <run-dir>/regrade_v2.json); a mixed run does both and
    returns {"quiz_answer": summary, "cond_answer": report}. Single-type
    runs return that path's result directly. Never touches the original
    summary.json."""
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise RegradeError(f"run dir not found: {run_dir}")
    records = load_ledger_records(run_dir)
    has_quiz = any(r.get("type") == "quiz_answer" for r in records)
    has_cond = any(r.get("type") == "cond_answer" for r in records)
    if not (has_quiz or has_cond):
        raise RegradeError(
            f"no quiz_answer or cond_answer records under {run_dir}")

    cond_report = regrade_cablese(run_dir, tasks_dir=tasks_dir) \
        if has_cond else None
    quiz_summary = _regrade_quiz(run_dir, tasks_dir=tasks_dir) \
        if has_quiz else None
    if has_quiz and has_cond:
        return {"quiz_answer": quiz_summary, "cond_answer": cond_report}
    return quiz_summary if has_quiz else cond_report


def _regrade_quiz(run_dir, tasks_dir=None):
    """Baseline quiz_answer path (original SPEC-VALIDATION §3 tool)."""
    records = load_ledger_records(run_dir)
    passages = resolve_passages(run_dir, tasks_dir or TASKS2_DIR)
    answers = [r for r in records if r.get("type") == "quiz_answer"]
    judges = [r for r in records if r.get("type") == "judge"]
    drops = [r for r in records if r.get("type") == "dedupe_drop"]
    if not answers:
        raise RegradeError(f"no quiz_answer records under {run_dir}")

    missing = sorted({r["task_id"] for r in answers
                      if r["task_id"] not in passages})
    if missing:
        raise RegradeError(
            f"cannot resolve passages for {len(missing)} task id(s): "
            f"{missing[:5]}... (provide run-dir passages.json or "
            f"--tasks-dir)")

    regraded = []
    for rec in answers:
        g = grade_v2(passages[rec["task_id"]], rec.get("question", ""),
                     rec["expected"], rec["raw_answer"])
        out = dict(rec)
        out["anchored"] = g["anchored"]
        out["grade_v2"] = g["diagnostics"]
        out["correct"] = g["correct"]  # grader v2 verdict REPLACES raw
        regraded.append(out)

    # Code adjudication of judge-overturned misses (SPEC-VALIDATION
    # §3): each judged grader_wrong miss is re-decided by the anchored
    # match alone — the judge verdict is recorded, never applied.
    verdicts = {(j["task_id"], j["q_index"]): j for j in judges}
    overturns = []
    for rec in regraded:
        j = verdicts.get((rec["task_id"], rec["q_index"]))
        if j is not None and j.get("verdict") == "grader_wrong":
            overturns.append({
                "task_id": rec["task_id"], "q_index": rec["q_index"],
                "judge_verdict": "grader_wrong",
                "grader_v2_correct": rec["correct"],
                "grader_v2": rec["grade_v2"],
            })

    out_dir = os.path.join(run_dir, "regrade_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "regraded_answers.jsonl"), "w",
              encoding="utf-8") as fh:
        for rec in regraded:
            fh.write(json.dumps(rec, sort_keys=True,
                                ensure_ascii=False) + "\n")

    summary = summarize_baseline(regraded, judges, drops,
                                 run_id=os.path.basename(run_dir)
                                 + "_regrade_v2")
    summary["regrade"] = {
        "source_run_dir": run_dir,
        "answers_regraded": len(regraded),
        "judge_overturns_adjudicated_by_code": len(overturns),
        "judge_overturns_upheld_by_grader_v2": sum(
            1 for o in overturns if o["grader_v2_correct"]),
        "note": ("grader v2 verdicts only; judge data quarantined "
                 "under diagnostics; no new API calls were made"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2,
                  ensure_ascii=False)
        fh.write("\n")
    with open(os.path.join(out_dir, "diagnostics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "judge_records": judges,
            "judge_overturn_adjudications": overturns,
        }, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"[regrade] {len(regraded)} answers re-graded with grader v2 "
          f"-> {out_dir}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ab.regrade", description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tasks-dir", default=None,
                        help="passage bank used to resolve task ids "
                             "(default ab/tasks2)")
    args = parser.parse_args(argv)
    try:
        regrade_run(args.run_dir, tasks_dir=args.tasks_dir)
    except RegradeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
