#!/usr/bin/env python3
"""Cablese output-recovery experiment (SPEC-CABLESE.md).

Two axes over the tasks2 bank (50 passages), one fresh 24-question
stratified set per passage generated ONCE (ab/baseline.py's generator,
reused verbatim) and answered under every condition:

Record axis ("the model leaves a note in cablese; later someone reads it"):
  R-PLAIN           model writes a complete plain-English record
  R-CABLESE         model writes a complete cablese (telegraphese) record
  A-FROM-PLAIN      fresh-context model answers the 24 Qs given R-PLAIN
  A-FROM-CABLESE    fresh-context model answers the 24 Qs given R-CABLESE

Answer axis ("the model replies in cablese; the reply is decoded"):
  A-PLAIN-DIRECT    model answers each Q from the passage, plain English
  A-CABLESE-DIRECT  model answers each Q from the passage, in cablese
  DECODE            a separate fresh call expands each cablese answer
                    back to plain English given ONLY that answer

Every miss in every condition goes through the baseline judge taxonomy
(grader_wrong/partial/wrong/ambiguous) plus the sampled-hit
false-positive estimate; Wilson CIs + bootstrap-over-passages come from
ab/baseline.py unchanged. Headline fields: cablese_savings_pct
(records), recovery_ratio (A-FROM-CABLESE / A-FROM-PLAIN), decode_ratio
(DECODE / A-PLAIN-DIRECT); paired exact McNemar on both axes.

CLI:
  python3 -m ab.cablese cablese [--passages N] [--results-dir D]
      [--stub] [--max-tokens M] [--task-offset O] [--task-limit L]
      [--slice-id S]
  python3 -m ab.cablese cablese-merge --results-dir D

Slicing mirrors the baseline contract (slice math over the unsliced
capped list; parallel slices tile the serial run exactly). The CLI
dispatch passes --task-offset/--task-limit/--slice-id through to
run_cablese explicitly — the expB 14475 incident (dispatch dropping
slice args) is regression-tested at the CLI level (see
tests/test_cablese.py).
"""

import argparse
import datetime
import json
import math
import os
import random
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ab.baseline import (  # noqa: E402
    JUDGE_SAMPLE_HITS,
    JUDGE_SEED,
    JUDGE_SYSTEM,
    JUDGE_USER,
    generate_stratified_quiz,
    make_stub_client as _baseline_stub_client,
    parse_verdict,
    summarize_baseline,
    wilson_ci,
)
from ab.grade import grade_v2  # noqa: E402
from ab.expAB import (  # noqa: E402
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DeadlineExceeded,
    ExpABError,
    Ledger,
    QA_SYSTEM,
    check_deadline,
    git_sha,
)
from ab.instrument import call_model, default_client  # noqa: E402
from ab.run_ab import QA_PROMPT  # noqa: E402
from ab.run_v21 import TASKS2_DIR, load_tasks  # noqa: E402
from tcb.tokencount import approx_count  # noqa: E402


class CableseError(Exception):
    """Explicit cablese-run failure (missing material, bad slice)."""


# --- frozen prompts ----------------------------------------------------------

CABLESE_RULES = (
    "Write in cablese (telegraphese): drop articles and filler words; "
    "abbreviate common words; telegram style, short dense phrases "
    "rather than full sentences; ALL facts retained, nothing omitted; "
    "numbers and proper nouns verbatim."
)

RECORD_PLAIN_SYSTEM = (
    "You write a complete plain-English record of a text. Capture ALL "
    "facts the text states, in normal prose; include the whole content "
    "of the text, with nothing de-selected."
)

RECORD_PLAIN_USER = (
    "Write a complete record of the following text in plain English. "
    "Capture all facts, normal prose, whole content, nothing "
    "de-selected.\n\nText:\n{passage}"
)

RECORD_CABLESE_SYSTEM = (
    "You write a complete record of a text in cablese (telegraphese). "
    + CABLESE_RULES
    + " The record must still carry the whole content of the text."
)

RECORD_CABLESE_USER = (
    "Write a complete record of the following text in cablese "
    "(telegraphese). Rules: drop articles and fillers; abbreviate; "
    "telegram style; ALL facts retained; numbers and proper nouns "
    "verbatim. Whole content — nothing de-selected.\n\nText:\n{passage}"
)

ANSWER_FROM_RECORD_USER = (
    "Answer the question using only the record. Answer briefly in "
    "normal English.\n\nRecord: {record}\n\nQuestion: {question}\nAnswer:"
)

A_CABLESE_DIRECT_SYSTEM = (
    "You answer reading-comprehension tasks faithfully. " + CABLESE_RULES
    + " Keep the answer answer-sized: no longer than the facts asked for."
)

A_CABLESE_DIRECT_USER = (
    "Answer the question using only the passage, writing your answer in "
    "cablese (telegraphese): drop articles and fillers; abbreviate; "
    "telegram style; keep ALL facts the answer needs; numbers and "
    "proper nouns verbatim.\n\nPassage: {passage}\n\nQuestion: "
    "{question}\nAnswer:"
)

DECODE_SYSTEM = (
    "You expand cablese (telegraphese) text back into plain English, "
    "keeping every fact, number and proper noun exactly as given."
)

DECODE_USER = (
    "Expand the following cablese answer back into plain English, given "
    "ONLY this answer. Keep every fact, number and proper noun exactly; "
    "do not add information.\n\nCablese answer: {answer}\nPlain English:"
)

RECORD_CONDITIONS = ("R-PLAIN", "R-CABLESE")
ANSWER_CONDITIONS = (
    "A-FROM-PLAIN",
    "A-FROM-CABLESE",
    "A-PLAIN-DIRECT",
    "A-CABLESE-DIRECT",
    "DECODE",
)
ALL_CONDITIONS = RECORD_CONDITIONS + ANSWER_CONDITIONS

MCNEMAR_PAIRS = (
    ("A-FROM-CABLESE", "A-FROM-PLAIN"),
    ("DECODE", "A-PLAIN-DIRECT"),
)


# --- paired statistics -------------------------------------------------------


def mcnemar_exact(hits_a, hits_b):
    """Exact two-sided McNemar over paired booleans.

    hits_a/hits_b map the same pair keys -> bool (correct / not).
    Returns {"n", "a", "b", "c", "d", "p_value"} where b / c are the
    discordant cells; p is the exact binomial two-sided test on the
    discordant pairs (n = b + c). With no discordance p is 1.0; with no
    pairs at all everything is None."""
    keys = sorted(set(hits_a) & set(hits_b))
    if not keys:
        return {"n": 0, "a": None, "b": None, "c": None, "d": None,
                "p_value": None}
    a = sum(1 for k in keys if hits_a[k] and hits_b[k])
    b = sum(1 for k in keys if hits_a[k] and not hits_b[k])
    c = sum(1 for k in keys if not hits_a[k] and hits_b[k])
    d = sum(1 for k in keys if not hits_a[k] and not hits_b[k])
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        m = min(b, c)
        tail = sum(math.comb(n_disc, i) for i in range(m + 1)) / (
            2 ** n_disc)
        p = min(1.0, 2.0 * tail)
    return {"n": len(keys), "a": a, "b": b, "c": c, "d": d,
            "p_value": round(p, 6)}


def _lexical_tokens(text):
    s = text.casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return frozenset(s.split())


def lexical_similarity(text_a, text_b):
    """Jaccard overlap of normalized token sets (decode fidelity stat)."""
    a, b = _lexical_tokens(text_a), _lexical_tokens(text_b)
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 6)


# --- summary -----------------------------------------------------------------


def _pair_hits(answers):
    return {(r["task_id"], r["q_index"]): r["correct"]
            for r in answers}


def _headline_acc(block):
    """Headline accuracy of a condition block — grader v2 raw (the
    judge is diagnostic only; SPEC-VALIDATION knockout principle)."""
    return None if block is None else block["overall_raw"]["accuracy"]


def _ratio(num, den):
    if den is None or num is None or den == 0:
        return None
    return round(num / den, 6)


def summarize_cablese(answers_by_cond, judges_by_cond, drops, records,
                      run_id="cablese"):
    """Compute summary.json content purely from record lists (used by
    both the serial/slice runs and cablese-merge, so merge is
    serial-equivalent by construction). Reuses ab/baseline.py's
    summarize_baseline for every per-condition block (grader v2 raw
    headline, Wilson CIs, bootstrap-over-passages, taxonomy under
    diagnostics, sampled-hit FP estimate)."""
    conditions = {}
    for cond in ALL_CONDITIONS:
        answers = answers_by_cond.get(cond, [])
        judges = judges_by_cond.get(cond, [])
        conditions[cond] = summarize_baseline(
            answers, judges, drops if drops else [])

    rec_plain = {r["task_id"]: r for r in records
                 if r.get("condition") == "R-PLAIN"}
    rec_cablese = {r["task_id"]: r for r in records
                   if r.get("condition") == "R-CABLESE"}
    passage_tokens = {}
    for tid in sorted(set(rec_plain) & set(rec_cablese)):
        passage_tokens[tid] = {
            "plain_approx_tokens": rec_plain[tid]["approx_tokens"],
            "cablese_approx_tokens": rec_cablese[tid]["approx_tokens"],
        }

    plain_total = sum(v["plain_approx_tokens"]
                      for v in passage_tokens.values())
    cablese_total = sum(v["cablese_approx_tokens"]
                        for v in passage_tokens.values())
    savings = (round(100.0 * (1.0 - cablese_total / plain_total), 6)
               if plain_total else None)

    mcnemar = {}
    for x, y in MCNEMAR_PAIRS:
        mcnemar[f"{x}_vs_{y}"] = mcnemar_exact(
            _pair_hits(answers_by_cond.get(x, [])),
            _pair_hits(answers_by_cond.get(y, [])))

    sims = [r["lexical_sim_vs_plain_direct"]
            for r in answers_by_cond.get("DECODE", [])
            if r.get("lexical_sim_vs_plain_direct") is not None]

    summary = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "conditions": conditions,
        "mcnemar": mcnemar,
        "record_tokens": {
            "per_passage": [
                {"task_id": tid, **vals}
                for tid, vals in sorted(passage_tokens.items())
            ],
            "plain_approx_tokens_total": plain_total,
            "cablese_approx_tokens_total": cablese_total,
            "note": "approx_count heuristic (tcb.tokencount); API usage "
                    "completion tokens are on the record ledger entries",
        },
        "decode_fidelity": {
            "n": len(sims),
            "mean_lexical_similarity": (round(sum(sims) / len(sims), 6)
                                        if sims else None),
        },
        "cablese_savings_pct": savings,
        "recovery_ratio": _ratio(
            _headline_acc(conditions.get("A-FROM-CABLESE")),
            _headline_acc(conditions.get("A-FROM-PLAIN"))),
        "decode_ratio": _ratio(
            _headline_acc(conditions.get("DECODE")),
            _headline_acc(conditions.get("A-PLAIN-DIRECT"))),
    }
    return summary


# --- experiment --------------------------------------------------------------


def _answer_condition_prompts(cond, task, record, item):
    """(system, user) for one answering call in condition cond."""
    if cond == "A-FROM-PLAIN" or cond == "A-FROM-CABLESE":
        return QA_SYSTEM, ANSWER_FROM_RECORD_USER.format(
            record=record, question=item["q"])
    if cond == "A-PLAIN-DIRECT":
        return QA_SYSTEM, QA_PROMPT.format(context=task["passage"],
                                           question=item["q"])
    if cond == "A-CABLESE-DIRECT":
        return A_CABLESE_DIRECT_SYSTEM, A_CABLESE_DIRECT_USER.format(
            passage=task["passage"], question=item["q"])
    raise CableseError(f"unknown answering condition {cond!r}")


def _approx_tokens(text):
    return approx_count(text)


def run_cablese(client, model, max_tokens, results_dir, tasks_dir=None,
                passages=None, task_offset=0, task_limit=None,
                slice_id=None, judge_sample_hits=JUDGE_SAMPLE_HITS,
                max_seconds=DEFAULT_MAX_SECONDS):
    """Cablese output-recovery experiment over tasks[offset:offset+limit]
    of the FULL (optionally passages-capped) list — slice math is over
    the unsliced list, so parallel slices tile the serial run exactly
    (baseline contract). slice_id redirects output to
    slices/slice_<id>.* with identical record shape."""
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
    run_meta = {"run_id": f"cablese_{stamp}", "git_sha": git_sha(),
                "model": model}
    if slice_id:
        run_meta["slice_id"] = slice_id
    ledger = Ledger(ledger_path, run_meta)
    deadline = time.monotonic() + max_seconds

    all_tasks = load_tasks(tasks_dir or TASKS2_DIR)
    if passages is not None:
        if passages < 0:
            raise CableseError(f"--passages {passages} must be >= 0")
        all_tasks = all_tasks[:passages]
    if task_offset < 0 or task_offset > len(all_tasks):
        raise CableseError(
            f"--task-offset {task_offset} out of range "
            f"(0..{len(all_tasks)} passages)")
    end = None if task_limit is None else task_offset + task_limit
    tasks = all_tasks[task_offset:end]
    if not tasks:
        raise CableseError(f"no passages selected (offset={task_offset}, "
                           f"limit={task_limit})")
    task_by_id = {t["id"]: t for t in tasks}

    answers_by_cond = {c: [] for c in ANSWER_CONDITIONS}
    judges_by_cond = {c: [] for c in ANSWER_CONDITIONS}
    records, drops = [], []
    plain_direct_raw = {}

    for task in tasks:
        # 1) one fresh stratified quiz per passage, reused by ALL
        #    conditions (clean within-experiment pairing).
        check_deadline(deadline, "generate")
        kept, dropped, unanchored = generate_stratified_quiz(
            client, model, task, max_tokens, ledger)
        for it in unanchored:
            ledger.append({
                "type": "anchor_reject", "task_id": task["id"],
                "q_type": it["q_type"], "question": it["q"],
                "answer": it["a"],
            })
        for d in dropped:
            ledger.append({
                "type": "dedupe_drop", "task_id": task["id"],
                "q_type": d["item"]["q_type"],
                "question": d["item"]["q"], "answer": d["item"]["a"],
                "duplicate_of_kept_index": d["kept_index"],
                "overlap": d["overlap"],
            })
            drops.append({"task_id": task["id"]})
        for qi, item in enumerate(kept):
            ledger.append({
                "type": "quiz_item", "task_id": task["id"],
                "q_index": qi, "q_type": item["q_type"],
                "question": item["q"], "answer": item["a"],
            })

        # 2) record axis: R-PLAIN and R-CABLESE.
        rec_texts = {}
        for cond, system, user_tmpl in (
            ("R-PLAIN", RECORD_PLAIN_SYSTEM, RECORD_PLAIN_USER),
            ("R-CABLESE", RECORD_CABLESE_SYSTEM, RECORD_CABLESE_USER),
        ):
            check_deadline(deadline, "record")
            result = call_model(client, model, system,
                                user_tmpl.format(passage=task["passage"]),
                                max_tokens)
            rec_texts[cond] = result["raw"]
            rec = {
                "type": "record", "condition": cond,
                "task_id": task["id"], "register": task["register"],
                "len_class": task["len_class"],
                "approx_tokens": _approx_tokens(result["raw"]),
                "usage_completion_tokens":
                    result["usage"]["completion_tokens"],
                "text": result["raw"],
            }
            ledger.append({
                "type": "model_call", "call": f"record_{cond}",
                "task_id": task["id"], "raw": result["raw"],
                "usage": result["usage"],
                "latency_ms": result["latency_ms"],
            })
            ledger.append(rec)
            records.append(rec)

        # 3) answer axis: four direct/from-record conditions.
        for cond in ("A-FROM-PLAIN", "A-FROM-CABLESE",
                     "A-PLAIN-DIRECT", "A-CABLESE-DIRECT"):
            record_text = rec_texts[
                "R-PLAIN" if cond == "A-FROM-PLAIN" else "R-CABLESE"]
            for qi, item in enumerate(kept):
                check_deadline(deadline, "quiz")
                system, user = _answer_condition_prompts(
                    cond, task, record_text, item)
                result = call_model(client, model, system, user,
                                    max_tokens)
                g = grade_v2(task["passage"], item["q"], item["a"],
                             result["raw"])
                rec = {
                    "type": "cond_answer", "condition": cond,
                    "task_id": task["id"], "register": task["register"],
                    "len_class": task["len_class"],
                    "q_index": qi, "q_type": item["q_type"],
                    "question": item["q"], "expected": item["a"],
                    "raw_answer": result["raw"], "anchored": g["anchored"],
                    "grade_v2": g["diagnostics"],
                    "correct": g["correct"],
                }
                ledger.append({
                    "type": "model_call", "call": "cond_qa",
                    "condition": cond, "task_id": task["id"],
                    "q_index": qi, "raw": result["raw"],
                    "usage": result["usage"],
                    "latency_ms": result["latency_ms"],
                    "correct": g["correct"], "expected": item["a"],
                })
                ledger.append(rec)
                answers_by_cond[cond].append(rec)
                if cond == "A-PLAIN-DIRECT":
                    plain_direct_raw[(task["id"], qi)] = result["raw"]

        # 4) DECODE: expand each cablese answer given ONLY that answer.
        for rec in list(answers_by_cond["A-CABLESE-DIRECT"]):
            if rec["task_id"] != task["id"]:
                continue
            check_deadline(deadline, "decode")
            result = call_model(
                client, model, DECODE_SYSTEM,
                DECODE_USER.format(answer=rec["raw_answer"]), max_tokens)
            g = grade_v2(task["passage"], rec["question"], rec["expected"],
                         result["raw"])
            drec = {
                "type": "cond_answer", "condition": "DECODE",
                "task_id": task["id"], "register": task["register"],
                "len_class": task["len_class"],
                "q_index": rec["q_index"], "q_type": rec["q_type"],
                "question": rec["question"], "expected": rec["expected"],
                "raw_answer": result["raw"], "anchored": g["anchored"],
                "grade_v2": g["diagnostics"],
                "correct": g["correct"],
                "cablese_answer": rec["raw_answer"],
                "lexical_sim_vs_plain_direct": lexical_similarity(
                    result["raw"],
                    plain_direct_raw.get((task["id"], rec["q_index"]),
                                         "")),
            }
            ledger.append({
                "type": "model_call", "call": "decode",
                "task_id": task["id"], "q_index": rec["q_index"],
                "raw": result["raw"], "usage": result["usage"],
                "latency_ms": result["latency_ms"],
                "correct": g["correct"], "expected": rec["expected"],
            })
            ledger.append(drec)
            answers_by_cond["DECODE"].append(drec)

    # 5) judge pass: EVERY miss in EVERY condition + the seeded random
    #    hit sample per condition (baseline conventions, sampled hits
    #    feed the false-positive estimate only).
    for cond in ANSWER_CONDITIONS:
        answers = answers_by_cond[cond]
        misses = [r for r in answers if not r["correct"]]
        hits = sorted((r for r in answers if r["correct"]),
                      key=lambda r: (r["task_id"], r["q_index"]))
        rng = random.Random(JUDGE_SEED)
        n_sample = min(judge_sample_hits, len(hits))
        sampled = {(r["task_id"], r["q_index"])
                   for r in rng.sample(hits, n_sample)} if n_sample else set()
        to_judge = [(r, False) for r in misses]
        to_judge += [(r, True) for r in hits
                     if (r["task_id"], r["q_index"]) in sampled]
        for rec, is_hit in to_judge:
            check_deadline(deadline, "judge")
            passage = task_by_id[rec["task_id"]]["passage"]
            result = call_model(
                client, model, JUDGE_SYSTEM,
                JUDGE_USER.format(passage=passage,
                                  question=rec["question"],
                                  expected=rec["expected"],
                                  given=rec["raw_answer"]),
                max_tokens,
            )
            verdict = parse_verdict(result["raw"])
            judge_rec = {
                "type": "judge", "condition": cond,
                "task_id": rec["task_id"], "q_index": rec["q_index"],
                "sampled_hit": is_hit, "verdict": verdict,
                "raw": result["raw"],
            }
            ledger.append({
                "type": "model_call", "call": "judge",
                "condition": cond, "task_id": rec["task_id"],
                "q_index": rec["q_index"], "raw": result["raw"],
                "usage": result["usage"],
                "latency_ms": result["latency_ms"], "verdict": verdict,
            })
            ledger.append(judge_rec)
            judges_by_cond[cond].append(judge_rec)

    summary = summarize_cablese(answers_by_cond, judges_by_cond, drops,
                                records, run_id=run_meta["run_id"])
    summary["passages"] = len(tasks)
    if slice_id:
        summary["slice_id"] = slice_id
        summary["task_offset"] = task_offset
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"[cablese] passages={len(tasks)} "
          f"recovery_ratio={summary['recovery_ratio']} "
          f"decode_ratio={summary['decode_ratio']} "
          f"savings_pct={summary['cablese_savings_pct']} "
          f"-> {summary_path}")
    return summary


def merge_cablese_slices(results_dir):
    """Concatenate slices/slice_*.jsonl, recompute the summary from ALL
    records (deterministic, sorted) and write the combined
    ledger.jsonl + summary.json."""
    slices_dir = os.path.join(results_dir, "slices")
    if not os.path.isdir(slices_dir):
        raise CableseError(f"no slices directory at {slices_dir}")
    names = sorted(n for n in os.listdir(slices_dir)
                   if re.fullmatch(r"slice_.+\.jsonl", n))
    if not names:
        raise CableseError(f"no slice_*.jsonl files in {slices_dir}")
    records_out, records, answers_by_cond = [], [], {
        c: [] for c in ANSWER_CONDITIONS}
    judges_by_cond = {c: [] for c in ANSWER_CONDITIONS}
    drops, rec_recs = [], []
    for name in names:
        with open(os.path.join(slices_dir, name), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records_out.append(rec)
                rtype = rec.get("type")
                if rtype == "cond_answer":
                    answers_by_cond[rec["condition"]].append(rec)
                elif rtype == "judge":
                    judges_by_cond[rec["condition"]].append(rec)
                elif rtype == "dedupe_drop":
                    drops.append(rec)
                elif rtype == "record":
                    rec_recs.append(rec)
    records_out.sort(key=lambda r: json.dumps(r, sort_keys=True,
                                              ensure_ascii=False))
    ledger_path = os.path.join(results_dir, "ledger.jsonl")
    with open(ledger_path, "w", encoding="utf-8") as fh:
        for rec in records_out:
            fh.write(json.dumps(rec, sort_keys=True,
                                ensure_ascii=False) + "\n")
    summary = summarize_cablese(answers_by_cond, judges_by_cond, drops,
                                rec_recs, run_id="cablese_merged")
    summary["passages"] = len({r["task_id"]
                               for rs in answers_by_cond.values()
                               for r in rs})
    summary["slices_merged"] = names
    with open(os.path.join(results_dir, "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"[cablese] merged {len(names)} slices, "
          f"{sum(len(v) for v in answers_by_cond.values())} answers "
          f"-> {os.path.join(results_dir, 'summary.json')}")
    return summary


# --- stub client (declared smoke mode; never a silent fallback) --------------


def _stub_response(answer, user):
    pt = approx_count(user) + 8
    return {
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": answer},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens":
                  max(4, approx_count(answer)),
                  "total_tokens": pt + 4},
    }


def make_stub_client():
    """Deterministic stub for smoke mode and tests: baseline stub
    routing for quiz generation / judging / plain QA, plus fixed record
    and decode replies (the cablese record is shorter than the plain
    one, so record-token accounting has a positive savings). A declared
    stub, never a fallback."""
    inner = _baseline_stub_client()

    def client(payload):
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        if "plain-English record" in system:
            return _stub_response(
                "The ship Aurora sailed from Bristol in March 1848 with "
                "a cargo of iron, a crew of nineteen, and Captain Elias "
                "Hoy as master, putting into Falmouth after a storm and "
                "then continuing to Lisbon and Porto in plain English "
                "prose for the whole record.", user)
        if "telegraphese" in system and "record" in system:
            return _stub_response(
                "rcrd: aurora brstl 1848 iron crw 19 hoy mstr", user)
        if "Expand" in system and "cablese" in system:
            return _stub_response("stub decoded answer", user)
        return inner(payload)

    return client


# --- CLI ---------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ab.cablese", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("cablese", "cablese-merge"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
        p.add_argument("--max-seconds", type=float,
                       default=DEFAULT_MAX_SECONDS,
                       help="wall-clock guard on all-loops stages "
                            "(default %(default)s)")
        p.add_argument("--results-dir", default=os.path.join(_ROOT, "results"))
        p.add_argument("--stub", action="store_true",
                       help="declared deterministic stub client (smoke)")
        if name == "cablese":
            p.add_argument("--tasks-dir", default=None)
            p.add_argument("--passages", type=int, default=None,
                           help="cap the passage bank; slice math is over "
                                "the capped list")
            p.add_argument("--task-offset", type=int, default=0,
                           help="run tasks[offset:offset+limit] of the "
                                "FULL passage list (parallel slicing)")
            p.add_argument("--task-limit", type=int, default=None)
            p.add_argument("--slice-id", default=None,
                           help="write slices/slice_<id>.* instead of the "
                                "single ledger/summary")
    args = parser.parse_args(argv)

    if args.cmd == "cablese-merge":
        try:
            merge_cablese_slices(args.results_dir)
        except CableseError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    stub = args.stub or os.environ.get("TCB_CABLESE_STUB") == "1"
    client = make_stub_client() if stub else default_client

    try:
        # NOTE (expB 14475 regression class): every slice arg is passed
        # through to run_cablese EXPLICITLY; the CLI-level regression
        # test test_cli_dispatch_passes_offset_and_slice_args_14475
        # asserts this end of the contract.
        run_cablese(
            client, args.model, args.max_tokens, args.results_dir,
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
    except (CableseError, ExpABError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
