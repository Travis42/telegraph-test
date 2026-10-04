#!/usr/bin/env python3
"""Record-decode experiment (SPEC-RECORD-DECODE.md) — fill cell (b).

The cbl2 answer axis measured decoding transcribed ANSWERS (decode_ratio
0.862). This cell measures the complementary path: expand the whole
GLM-written R-CABLESE record back to plain English, then answer the
anchored question set given ONLY the expanded record:

  R-DECODED        one fresh GLM call expands the cbl2 R-CABLESE record
                   to plain English (record-sized mirror of the frozen
                   DECODE prompts, ONLY-record semantics)
  A-FROM-DECODED   fresh GLM call answers each anchored question given
                   ONLY R-DECODED (ANSWER_FROM_RECORD_USER reused from
                   ab.cablese, record=R-DECODED)

Reuses cbl2 artifacts read-only via crossmodel's guarded loader
(--source-run, default results/cablese_cbl2): the per-passage anchored
pool (SPEC-VALIDATION §2 load-time gate — unanchored reused items are
rejected + logged, quota drawn from the anchored pool only) and the
v2-regraded A-FROM-PLAIN / A-FROM-CABLESE verdicts for the ratios and
the paired McNemar. Grading is grader v2 anchored to the SOURCE passage
(grade_v2 only, zero LLM judges — SPEC-VALIDATION knockout principle).

Headline: decoded_recovery_ratio = acc(A-FROM-DECODED)/acc(A-FROM-PLAIN);
record comparison decode_ratio_rec = acc(A-FROM-DECODED)/acc(A-FROM-
CABLESE); paired McNemar A-FROM-DECODED vs A-FROM-PLAIN; token
accounting mean R-DECODED vs R-CABLESE vs R-PLAIN (expansion overhead).

CLI:
  python3 -m ab.record_decode record-decode [--source-run D]
      [--results-dir D] [--stub] [--max-tokens M] [--task-offset O]
      [--task-limit L] [--slice-id S] [--passages N]
  python3 -m ab.record_decode record-decode-merge --results-dir D

Slicing mirrors the cablese contract (slice math over the unsliced
sorted source-passage list; parallel slices tile the serial run
exactly). The CLI dispatch passes --task-offset/--task-limit/--slice-id
through to run_record_decode EXPLICITLY — the expB 14475 incident
(dispatch dropping slice args) is regression-tested at the CLI level
(see tests/test_record_decode.py).
"""

import argparse
import datetime
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ab.baseline import (  # noqa: E402
    summarize_baseline,
)
from ab.cablese import (  # noqa: E402
    ANSWER_FROM_RECORD_USER,
    _ratio,
    mcnemar_exact,
)
from ab.crossmodel import (  # noqa: E402
    CrossModelError,
    SOURCE_QUESTIONS_PER_PASSAGE,
    _slice_tasks,
    load_source_run,
)
from ab.expAB import (  # noqa: E402
    DEFAULT_MAX_SECONDS,
    DEFAULT_MODEL,
    DeadlineExceeded,
    ExpABError,
    Ledger,
    QA_SYSTEM,
    check_deadline,
    git_sha,
)
from ab.grade import grade_v2  # noqa: E402
from ab.instrument import (  # noqa: E402
    InstrumentError,
    call_model,
    default_client,
)
from ab.run_v21 import TASKS2_DIR, load_tasks  # noqa: E402
from tcb.tokencount import approx_count  # noqa: E402


class RecordDecodeError(Exception):
    """Explicit record-decode failure (missing source material, bad
    slice)."""


DEFAULT_SOURCE_RUN = os.path.join(_ROOT, "results", "cablese_cbl2")

DECODED_CONDITION = "A-FROM-DECODED"
DECODED_RECORD_CONDITION = "R-DECODED"

# Frozen prompts (SPEC-RECORD-DECODE design §1): record-sized mirror of
# cablese's DECODE_SYSTEM/DECODE_USER semantics — the expansion sees
# ONLY the cablese record, never the source passage.

RECORD_DECODE_SYSTEM = (
    "You expand cablese (telegraphese) text back into plain English, "
    "keeping every fact, number and proper noun exactly as given."
)

RECORD_DECODE_USER = (
    "Expand the following cablese record back into plain English, given "
    "ONLY this record. Keep every fact, number and proper noun exactly; "
    "do not add information.\n\nCablese record: {record}\nPlain English:"
)

SOURCE_PLAIN_CONDS = ("A-FROM-PLAIN", "A-FROM-CABLESE")


# --- source v2 verdicts (A-FROM-PLAIN / A-FROM-CABLESE) -----------------------


def load_source_v2_hits(source_run, tasks_by_id=None):
    """Per-question grader-v2 verdicts for the source run's A-FROM-PLAIN
    and A-FROM-CABLESE conditions, for the ratios and the paired
    McNemar. Prefers the regrade artifact
    (<source_run>/regrade_v2/regraded_cond_answers.jsonl — the v2 truth
    of record for cbl2); if absent, recomputes the identical verdicts
    offline from the source ledger with grade_v2 anchored to the source
    passage (no new API calls). Returns {cond: {(task_id, q_index):
    bool}}."""
    hits = {c: {} for c in SOURCE_PLAIN_CONDS}
    regrade_path = os.path.join(source_run, "regrade_v2",
                                "regraded_cond_answers.jsonl")
    if os.path.isfile(regrade_path):
        with open(regrade_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("type") == "cond_answer" \
                        and rec.get("condition") in hits:
                    hits[rec["condition"]][
                        (rec["task_id"], rec["q_index"])] = \
                        bool(rec["correct"])
        return hits
    # offline recompute (regrade-equivalent by construction)
    ledger_path = os.path.join(source_run, "ledger.jsonl")
    if not os.path.isfile(ledger_path):
        raise RecordDecodeError(f"source ledger missing: {ledger_path}")
    if tasks_by_id is None:
        tasks_by_id = {t["id"]: t for t in load_tasks(TASKS2_DIR)}
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") != "cond_answer" \
                    or rec.get("condition") not in hits:
                continue
            task = tasks_by_id.get(rec["task_id"])
            if task is None:
                raise RecordDecodeError(
                    f"source task {rec['task_id']!r} absent from the "
                    f"tasks bank — cannot anchor the recompute")
            g = grade_v2(task["passage"], rec.get("question", ""),
                         rec["expected"], rec["raw_answer"])
            hits[rec["condition"]][(rec["task_id"], rec["q_index"])] = \
                g["correct"]
    return hits


def load_source_record_tokens(source_run):
    """{(task_id, condition): approx_tokens} for the source run's
    R-PLAIN / R-CABLESE record entries (token-accounting inputs; the
    ledger's stored approx_tokens, not a recompute)."""
    ledger_path = os.path.join(source_run, "ledger.jsonl")
    if not os.path.isfile(ledger_path):
        raise RecordDecodeError(f"source ledger missing: {ledger_path}")
    tokens = {}
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") == "record" and rec.get("condition") in (
                    "R-PLAIN", "R-CABLESE") \
                    and rec.get("approx_tokens") is not None:
                tokens[(rec["task_id"], rec["condition"])] = \
                    rec["approx_tokens"]
    return tokens


def _acc_from_hits(hits):
    if not hits:
        return None
    return round(sum(hits.values()) / len(hits), 6)


# --- summary ------------------------------------------------------------------


def summarize_record_decode(answers, decoded_records, source_hits,
                            source_tokens, task_ids, run_id="record_decode",
                            source_run=None, anchoring_gate=None):
    """Compute summary.json content purely from record lists (used by
    both the serial/slice runs and record-decode-merge, so merge is
    serial-equivalent by construction; source-side v2 verdicts and
    record tokens are recomputed from the same read-only artifacts).
    overall_raw (grader v2) is the single headline; no judge anywhere."""
    conditions = {
        DECODED_CONDITION: summarize_baseline(answers, [], []),
    }

    plain_hits = source_hits.get("A-FROM-PLAIN", {})
    cablese_hits = source_hits.get("A-FROM-CABLESE", {})
    decoded_hits = {(r["task_id"], r["q_index"]): r["correct"]
                    for r in answers}

    def mean_tokens(cond):
        vals = [v for (tid, c), v in source_tokens.items()
                if c == cond and tid in task_ids] \
            if cond != DECODED_RECORD_CONDITION else \
            [r["approx_tokens"] for r in decoded_records
             if r.get("approx_tokens") is not None]
        return {"n_records": len(vals),
                "mean_approx_tokens": (round(sum(vals) / len(vals), 6)
                                       if vals else None)}

    token_block = {cond: mean_tokens(cond) for cond in
                   (DECODED_RECORD_CONDITION, "R-CABLESE", "R-PLAIN")}
    m_dec = token_block[DECODED_RECORD_CONDITION]["mean_approx_tokens"]
    m_cab = token_block["R-CABLESE"]["mean_approx_tokens"]
    m_pln = token_block["R-PLAIN"]["mean_approx_tokens"]

    acc_dec = conditions[DECODED_CONDITION]["overall_raw"]["accuracy"]
    acc_pln = _acc_from_hits(plain_hits)
    acc_cab = _acc_from_hits(cablese_hits)

    summary = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "source_run": source_run,
        "conditions": conditions,
        "decoded_recovery_ratio": _ratio(acc_dec, acc_pln),
        "decode_ratio_rec": _ratio(acc_dec, acc_cab),
        "source_accuracies_v2": {
            "A-FROM-PLAIN": acc_pln,
            "A-FROM-CABLESE": acc_cab,
            "n_plain": len(plain_hits),
            "n_cablese": len(cablese_hits),
        },
        "mcnemar": {
            "A-FROM-DECODED_vs_A-FROM-PLAIN": mcnemar_exact(
                decoded_hits, plain_hits),
        },
        "record_tokens": {
            **token_block,
            "decoded_vs_plain_ratio": _ratio(m_dec, m_pln),
            "decoded_vs_cablese_ratio": _ratio(m_dec, m_cab),
            "note": "mean approx_count per record (tcb.tokencount); "
                    "expansion overhead = does decoding give the "
                    "tokens back? expected ~1x R-PLAIN",
        },
        "anchoring_gate": anchoring_gate or {
            "rejected_unanchored": 0, "passages": {}},
    }
    return summary


# --- experiment ----------------------------------------------------------------


def run_record_decode(client, model, max_tokens, results_dir,
                      source_run, tasks_dir=None, passages=None,
                      task_offset=0, task_limit=None, slice_id=None,
                      quota_per_passage=SOURCE_QUESTIONS_PER_PASSAGE,
                      max_seconds=DEFAULT_MAX_SECONDS):
    """Record-decode experiment over tasks[offset:offset+limit] of the
    FULL (optionally passages-capped) sorted source-passage list —
    slice math over the unsliced list, so parallel slices tile the
    serial run exactly (cablese contract). slice_id redirects output to
    slices/slice_<id>.* with identical record shape. Reused cbl2 quiz
    items pass the SPEC-VALIDATION anchoring gate (crossmodel's guarded
    loader); grading is grader v2 anchored to the source passage; zero
    judge calls."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir
    if slice_id:
        slice_dir = os.path.join(out_dir, "slices")
        os.makedirs(slice_dir, exist_ok=True)
        ledger_path = os.path.join(slice_dir, f"slice_{slice_id}.jsonl")
        summary_path = os.path.join(slice_dir,
                                    f"slice_{slice_id}.summary.json")
    else:
        ledger_path = os.path.join(out_dir, "ledger.jsonl")
        summary_path = os.path.join(out_dir, "summary.json")
    run_meta = {"run_id": f"record_decode_{stamp}", "git_sha": git_sha(),
                "model": model, "source_run": os.path.abspath(source_run)}
    if slice_id:
        run_meta["slice_id"] = slice_id
    ledger = Ledger(ledger_path, run_meta)
    deadline = time.monotonic() + max_seconds

    tasks_by_id = {t["id"]: t for t in load_tasks(tasks_dir or TASKS2_DIR)}
    try:
        source_passages = load_source_run(
            source_run, tasks_by_id=tasks_by_id,
            quota_per_passage=quota_per_passage)
        ids = _slice_tasks(source_passages, tasks_by_id, passages,
                           task_offset, task_limit)
    except CrossModelError as exc:
        raise RecordDecodeError(str(exc)) from exc

    # SPEC-VALIDATION §2 gate bookkeeping (crossmodel conventions):
    # rejected items are logged, never silent; reduced n recorded.
    gate_rejects, gate_passages = [], {}
    for tid in ids:
        slot = source_passages[tid]
        for rej in slot["unanchored"]:
            ledger.append({
                "type": "unanchored_reject",
                "task_id": rej["task_id"], "q_index": rej["q_index"],
                "q_type": rej["q_type"], "question": rej["question"],
                "answer": rej["answer"],
                "anchor_score": rej["anchor_score"],
            })
            gate_rejects.append(rej)
        pn = {"quota": quota_per_passage,
              "anchored_pool": slot["anchored_pool"],
              "n": len(slot["questions"])}
        ledger.append({"type": "passage_n", "task_id": tid, **pn})
        gate_passages[tid] = pn
    if gate_rejects:
        out_base = os.path.dirname(os.path.abspath(ledger_path))
        with open(os.path.join(out_base,
                               "unanchored_source_items.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(gate_rejects, fh, sort_keys=True, indent=2,
                      ensure_ascii=False)
            fh.write("\n")
    anchoring_gate = {"rejected_unanchored": len(gate_rejects),
                      "passages": gate_passages}

    source_hits = load_source_v2_hits(source_run, tasks_by_id=tasks_by_id)
    source_tokens = load_source_record_tokens(source_run)

    answers = []
    decoded_records = []
    for tid in ids:
        task = tasks_by_id[tid]
        src = source_passages[tid]

        # 1) R-DECODED: one fresh call expands the cablese record,
        #    given ONLY the record.
        check_deadline(deadline, "record_decode")
        result = call_model(
            client, model, RECORD_DECODE_SYSTEM,
            RECORD_DECODE_USER.format(record=src["R-CABLESE"]),
            max_tokens)
        decoded = result["raw"]
        rec = {
            "type": "record", "condition": DECODED_RECORD_CONDITION,
            "task_id": tid, "register": task["register"],
            "len_class": task["len_class"],
            "approx_tokens": approx_count(decoded),
            "usage_completion_tokens": result["usage"]["completion_tokens"],
            "text": decoded,
        }
        ledger.append({
            "type": "model_call", "call": "record_decode",
            "model": model, "task_id": tid, "raw": decoded,
            "usage": result["usage"],
            "latency_ms": result["latency_ms"],
        })
        ledger.append(rec)
        decoded_records.append(rec)

        # 2) A-FROM-DECODED: answer each anchored question given ONLY
        #    the expanded record; grade v2 anchored to the SOURCE passage.
        for item in src["questions"]:
            check_deadline(deadline, "answer")
            system, user = QA_SYSTEM, ANSWER_FROM_RECORD_USER.format(
                record=decoded, question=item["q"])
            result = call_model(client, model, system, user, max_tokens)
            g = grade_v2(task["passage"], item["q"], item["a"],
                         result["raw"])
            arec = {
                "type": "cond_answer", "condition": DECODED_CONDITION,
                "task_id": tid, "register": task["register"],
                "len_class": task["len_class"],
                "q_index": item["q_index"], "q_type": item["q_type"],
                "question": item["q"], "expected": item["a"],
                "raw_answer": result["raw"], "anchored": g["anchored"],
                "grade_v2": g["diagnostics"], "correct": g["correct"],
            }
            ledger.append({
                "type": "model_call", "call": "cond_qa",
                "condition": DECODED_CONDITION, "model": model,
                "task_id": tid, "q_index": item["q_index"],
                "raw": result["raw"], "usage": result["usage"],
                "latency_ms": result["latency_ms"],
                "correct": g["correct"], "expected": item["a"],
            })
            ledger.append(arec)
            answers.append(arec)

    # NO judge pass: the truth path ends at grader v2 (SPEC-VALIDATION
    # Amendment 1; knockout principle).

    summary = summarize_record_decode(
        answers, decoded_records, source_hits, source_tokens, set(ids),
        run_id=run_meta["run_id"], source_run=os.path.abspath(source_run),
        anchoring_gate=anchoring_gate)
    summary["passages"] = len(ids)
    if slice_id:
        summary["slice_id"] = slice_id
        summary["task_offset"] = task_offset
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2,
                  ensure_ascii=False)
        fh.write("\n")
    print(f"[record_decode] passages={len(ids)} "
          f"decoded_recovery_ratio={summary['decoded_recovery_ratio']} "
          f"decode_ratio_rec={summary['decode_ratio_rec']} "
          f"-> {summary_path}")
    return summary


def merge_record_decode_slices(results_dir, tasks_dir=None):
    """Concatenate slices/slice_*.jsonl, recompute the summary from ALL
    records (deterministic, sorted) and write the combined
    ledger.jsonl + summary.json. Source-side stats are reloaded from
    the read-only source run recorded in the slice run_meta."""
    slices_dir = os.path.join(results_dir, "slices")
    if not os.path.isdir(slices_dir):
        raise RecordDecodeError(f"no slices directory at {slices_dir}")
    names = sorted(n for n in os.listdir(slices_dir)
                   if re.fullmatch(r"slice_.+\.jsonl", n))
    if not names:
        raise RecordDecodeError(f"no slice_*.jsonl files in {slices_dir}")
    records_out, answers, decoded_records = [], [], []
    gate_rejects, gate_passages = [], {}
    source_run = None
    for name in names:
        with open(os.path.join(slices_dir, name), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records_out.append(rec)
                rtype = rec.get("type")
                if rtype == "cond_answer" \
                        and rec.get("condition") == DECODED_CONDITION:
                    answers.append(rec)
                elif rtype == "record" and rec.get("condition") \
                        == DECODED_RECORD_CONDITION:
                    decoded_records.append(rec)
                elif rtype == "unanchored_reject":
                    gate_rejects.append(rec)
                elif rtype == "passage_n":
                    gate_passages[rec["task_id"]] = {
                        k: rec[k] for k in ("quota", "anchored_pool", "n")}
                if source_run is None and rec.get("source_run"):
                    source_run = rec["source_run"]
    if source_run is None:
        raise RecordDecodeError(
            f"no source_run recorded in any slice under {slices_dir}")
    records_out.sort(key=lambda r: json.dumps(r, sort_keys=True,
                                              ensure_ascii=False))
    ledger_path = os.path.join(results_dir, "ledger.jsonl")
    with open(ledger_path, "w", encoding="utf-8") as fh:
        for rec in records_out:
            fh.write(json.dumps(rec, sort_keys=True,
                                ensure_ascii=False) + "\n")
    task_ids = {r["task_id"] for r in answers} | {
        r["task_id"] for r in decoded_records}
    summary = summarize_record_decode(
        answers, decoded_records,
        load_source_v2_hits(
            source_run,
            tasks_by_id={t["id"]: t
                         for t in load_tasks(tasks_dir or TASKS2_DIR)}),
        load_source_record_tokens(source_run), task_ids,
        run_id="record_decode_merged", source_run=source_run,
        anchoring_gate={"rejected_unanchored": len(gate_rejects),
                        "passages": gate_passages})
    summary["passages"] = len(task_ids)
    summary["slices_merged"] = names
    with open(os.path.join(results_dir, "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2,
                  ensure_ascii=False)
        fh.write("\n")
    print(f"[record_decode] merged {len(names)} slices, "
          f"{len(answers)} answers "
          f"-> {os.path.join(results_dir, 'summary.json')}")
    return summary


# --- stub client (declared smoke mode; never a silent fallback) ---------------


def _stub_response(answer, user):
    pt = approx_count(user) + 8
    return {
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": answer},
                      "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt,
                  "completion_tokens": max(4, approx_count(answer)),
                  "total_tokens": pt + 4},
    }


def make_stub_client():
    """Deterministic stub for smoke mode and tests: routes on the frozen
    prompt strings — the record-decode expansion comes out LONGER than
    the cablese record (so expansion overhead is visible in the token
    accounting), everything else delegates to the baseline stub (fixed
    non-matching answers). A declared stub, never a fallback."""
    from ab.baseline import make_stub_client as _baseline_stub_client
    inner = _baseline_stub_client()

    def client(payload):
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        if "expand" in system.casefold() and "cablese" in system.casefold():
            return _stub_response(
                "The ship Aurora sailed from Bristol in March 1848 with "
                "a cargo of iron, a crew of nineteen, and Captain Elias "
                "Hoy as master, putting into Falmouth after a storm and "
                "then continuing to Lisbon and Porto, expanded back into "
                "complete plain English prose for the whole record.", user)
        return inner(payload)

    return client


# --- CLI -----------------------------------------------------------------------


DEFAULT_RECORD_DECODE_MAX_TOKENS = 4000


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ab.record_decode",
        description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("record-decode", "record-decode-merge"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--max-tokens", type=int,
                       default=DEFAULT_RECORD_DECODE_MAX_TOKENS)
        p.add_argument("--max-seconds", type=float,
                       default=DEFAULT_MAX_SECONDS,
                       help="wall-clock guard on all-loops stages "
                            "(default %(default)s)")
        p.add_argument("--results-dir",
                       default=os.path.join(_ROOT, "results"))
        if name == "record-decode-merge":
            p.add_argument("--tasks-dir", default=None,
                           help="passage bank used to anchor the source "
                                "verdict recompute (default ab/tasks2)")
        else:
            p.add_argument("--source-run", default=DEFAULT_SOURCE_RUN,
                           help="completed cbl2 run dir (read-only input; "
                                "default %(default)s)")
            p.add_argument("--tasks-dir", default=None)
            p.add_argument("--passages", type=int, default=None,
                           help="cap the source passage bank; slice math "
                                "is over the capped list")
            p.add_argument("--task-offset", type=int, default=0,
                           help="run tasks[offset:offset+limit] of the "
                                "FULL sorted passage list (parallel "
                                "slicing)")
            p.add_argument("--task-limit", type=int, default=None)
            p.add_argument("--slice-id", default=None,
                           help="write slices/slice_<id>.* instead of "
                                "the single ledger/summary")
            p.add_argument("--stub", action="store_true",
                           help="declared deterministic stub client "
                                "(smoke)")
    args = parser.parse_args(argv)

    if args.cmd == "record-decode-merge":
        try:
            merge_record_decode_slices(args.results_dir,
                                       tasks_dir=args.tasks_dir)
        except RecordDecodeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    stub = args.stub or os.environ.get("TCB_RECORD_DECODE_STUB") == "1"
    client = make_stub_client() if stub else default_client

    try:
        # NOTE (expB 14475 regression class): every slice arg is
        # passed through to run_record_decode EXPLICITLY; the
        # CLI-level regression test
        # test_cli_dispatch_passes_offset_and_slice_args_14475_
        # record_decode asserts this end of the contract.
        run_record_decode(
            client, args.model, args.max_tokens, args.results_dir,
            source_run=args.source_run,
            tasks_dir=args.tasks_dir,
            passages=args.passages,
            task_offset=args.task_offset,
            task_limit=args.task_limit,
            slice_id=args.slice_id,
            max_seconds=args.max_seconds,
        )
    except DeadlineExceeded as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (RecordDecodeError, ExpABError, InstrumentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
