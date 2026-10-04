#!/usr/bin/env python3
"""Baseline deep measurement (SPEC-BASELINE.md Phase 1).

24 stratified questions per passage (12 fact-recall + 4 numeric +
4 ordering/spatial + 4 inference), generated in 2 calls of 12; a
deterministic lexical dedupe (normalized token overlap >= 0.6 drops the
LATER duplicate; drops are recorded, never silently lost); quiz
  answering through the existing instrument conventions (temperature 0,
  answer-from-passage QA prompt, grader v2 passage-anchored
  deterministic grading per SPEC-VALIDATION); a judge pass over
  every miss plus a random sample of hits classifying
  grader_wrong/partial/wrong/ambiguous (frozen prompt, raw persisted)
  is DIAGNOSTIC ONLY — no LLM in the truth path; Wilson +
  bootstrap-over-passages CIs; raw (grader v2) accuracy is the single
  headline in summary.json.

CLI:
  python3 -m ab.baseline baseline [--passages N] [--results-dir D]
      [--stub] [--task-offset O] [--task-limit L] [--slice-id S]
  python3 -m ab.baseline baseline-merge --results-dir D

Slicing mirrors the expB contract: slice math is over the unsliced
(capped) passage list, so parallel slices tile the serial run exactly.
The CLI dispatch passes --task-offset/--task-limit/--slice-id through
to run_baseline explicitly — the expB 14475 incident (dispatch dropping
slice args) is regression-tested at the CLI level from day one.
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
from ab.grade import anchored, anchor_score, grade_v2  # noqa: E402
from ab.instrument import call_model, default_client, parse_json_block  # noqa: E402
from ab.run_ab import QA_PROMPT  # noqa: E402
from ab.run_v21 import TASKS2_DIR, load_tasks  # noqa: E402
from tcb.tokencount import approx_count  # noqa: E402

# --- frozen instrument constants (SPEC-BASELINE Phase 1) --------------------

QUESTIONS_PER_PASSAGE = 24
HALF1_PLAN = [("fact", 12)]
HALF2_PLAN = [("numeric", 4), ("ordering", 4), ("inference", 4)]
ALL_TYPES = ("fact", "numeric", "ordering", "inference")
HALF2_ORDER = [t for t, n in HALF2_PLAN for _ in range(n)]

DEDUPE_THRESHOLD = 0.6
QUOTA = dict(HALF1_PLAN + HALF2_PLAN)  # per-type items required per passage
GEN_RETRY_ROUNDS = 1  # extra generation round before a quota hard error
JUDGE_SAMPLE_HITS = 100
JUDGE_SEED = 20261003
BOOT_SEED = 20261003
BOOT_RESAMPLES = 2000
WILSON_Z = 1.959964  # two-sided 95%

VERDICTS = ("grader_wrong", "partial", "wrong", "ambiguous")

GEN_SYSTEM = (
    "You write reading-comprehension quiz questions about a text. Reply "
    "ONLY with a JSON array of objects {\"q\": question, \"a\": answer, "
    "\"type\": tag}, where each answer is a short span of the text and "
    "tag is the requested question type. No other output."
)

GEN_HALF1_USER = (
    "Write 12 fact-recall questions answerable ONLY from this text. Each "
    "question asks about one concrete fact stated in the text (a name, "
    "place, event, role, or attribute) and each tag is \"fact\".\n\n"
    "Text:\n{passage}"
)

GEN_HALF2_USER = (
    "Write 12 questions answerable ONLY from this text: 4 numeric "
    "questions about numbers, quantities, dates or measurements "
    "(tag \"numeric\"), 4 ordering or spatial questions about sequence "
    "or relative position (tag \"ordering\"), and 4 inference questions "
    "requiring a conclusion drawn from combining statements in the text "
    "(tag \"inference\"). Order them: numeric first, then ordering, then "
    "inference.\n\nText:\n{passage}"
)

JUDGE_SYSTEM = (
    "You judge reading-comprehension quiz answers. You are given the "
    "passage, the question, the expected answer, and the answer given "
    "by a test subject. Reply with EXACTLY ONE WORD and nothing else:\n"
    "grader_wrong - the given answer is actually correct; a lexical "
    "grader comparing it to the expected answer would wrongly mark it "
    "wrong\n"
    "partial - the given answer is only partly correct\n"
    "wrong - the given answer is incorrect\n"
    "ambiguous - it cannot be determined from the passage"
)

JUDGE_USER = (
    "Passage:\n{passage}\n\nQuestion: {question}\n\nExpected answer: "
    "{expected}\n\nGiven answer: {given}\n\nReply with the single word "
    "grader_wrong, partial, wrong, or ambiguous."
)


class BaselineError(Exception):
    """Explicit baseline-run failure (missing material, bad slice)."""


# --- lexical dedupe ----------------------------------------------------------


def _q_tokens(question):
    s = question.casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return frozenset(s.split())


def token_overlap(question_a, question_b):
    """Deterministic normalized-token containment: |A∩B| / min(|A|,|B|)."""
    a, b = _q_tokens(question_a), _q_tokens(question_b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def dedupe_questions(items, threshold=DEDUPE_THRESHOLD):
    """Drop the LATER of any pair with token overlap >= threshold.

    Returns (kept, dropped) where each dropped entry records which kept
    index it duplicated and the overlap — drops are data, not losses.
    """
    kept, dropped = [], []
    for item in items:
        dup_index, dup_overlap = None, None
        for j, kept_item in enumerate(kept):
            overlap = token_overlap(item["q"], kept_item["q"])
            if overlap >= threshold:
                dup_index, dup_overlap = j, overlap
                break
        if dup_index is None:
            kept.append(item)
        else:
            dropped.append({
                "item": item,
                "kept_index": dup_index,
                "overlap": round(dup_overlap, 4),
            })
    return kept, dropped


# --- verdict parsing ---------------------------------------------------------


def parse_verdict(raw):
    """Parse a judge reply into one of VERDICTS, or 'unparsed'.

    Deterministic: a leading verdict word wins (grader_wrong checked
    before wrong); otherwise the first verdict word found anywhere in
    the reply; otherwise 'unparsed' (recorded, never defaulted).
    """
    if not isinstance(raw, str):
        return "unparsed"
    text = raw.strip().casefold()
    m = re.match(r"^[\s>*-]*(grader_wrong|partial|wrong|ambiguous)\b", text)
    if m:
        return m.group(1)
    for verdict in VERDICTS:
        if verdict in text:
            return verdict
    return "unparsed"


# --- statistics --------------------------------------------------------------


def wilson_ci(k, n, z=WILSON_Z):
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (round(max(0.0, center - half), 6),
            round(min(1.0, center + half), 6))


def bootstrap_overall_ci(answers, n_boot=BOOT_RESAMPLES,
                          seed=BOOT_SEED):
    """Bootstrap 95% CI for overall accuracy, resampling PASSAGES with
    replacement (answers within a passage correlate). Deterministic:
    passages processed in sorted task_id order, seeded RNG."""
    by_task = {}
    for rec in answers:
        k, n = by_task.get(rec["task_id"], (0, 0))
        ok = rec["correct"]
        by_task[rec["task_id"]] = (k + (1 if ok else 0), n + 1)
    pairs = [by_task[tid] for tid in sorted(by_task)]
    if not pairs:
        return None
    total_n = sum(n for _, n in pairs)
    if total_n == 0:
        return None
    rng = random.Random(seed)
    m = len(pairs)
    accs = []
    for _ in range(n_boot):
        k_sum, n_sum = 0, 0
        for _ in range(m):
            k_i, n_i = pairs[rng.randrange(m)]
            k_sum += k_i
            n_sum += n_i
        accs.append(k_sum / n_sum if n_sum else 0.0)
    accs.sort()
    lo = accs[int(0.025 * (len(accs) - 1))]
    hi = accs[int(0.975 * (len(accs) - 1))]
    return (round(lo, 6), round(hi, 6))


def _accuracy_block(answers):
    n = len(answers)
    if n == 0:
        return {"n": 0, "correct": 0, "accuracy": None,
                "wilson_95": None}
    k = sum(1 for rec in answers if rec["correct"])
    return {
        "n": n,
        "correct": k,
        "accuracy": round(k / n, 6),
        "wilson_95": wilson_ci(k, n),
    }


# --- summary -----------------------------------------------------------------


def summarize_baseline(answers, judges, drops, run_id="baseline"):
    """Compute summary.json content purely from record lists (used by
    both the serial/slice runs and baseline-merge, so merge is
    serial-equivalent by construction). Grader v2 raw is the SINGLE
    headline; judge output is quarantined under diagnostics/ and feeds
    no correctness value (SPEC-VALIDATION §1)."""
    verdicts = {(j["task_id"], j["q_index"]): j["verdict"] for j in judges}

    def group(key_fn):
        out = {}
        for rec in answers:
            out.setdefault(key_fn(rec), []).append(rec)
        return out

    per_type = {t: _accuracy_block(v) for t, v in sorted(group(
        lambda r: r["q_type"]).items())}
    per_register = {t: _accuracy_block(v) for t, v in sorted(group(
        lambda r: r["register"]).items())}

    misses = [r for r in answers if not r["correct"]]
    judged_miss_keys = [(r["task_id"], r["q_index"]) for r in misses
                        if (r["task_id"], r["q_index"]) in verdicts]
    taxonomy = {v: 0 for v in VERDICTS}
    taxonomy["unparsed"] = 0
    for key in judged_miss_keys:
        v = verdicts[key]
        taxonomy[v if v in taxonomy else "unparsed"] += 1
    taxonomy["unjudged"] = len(misses) - len(judged_miss_keys)

    hit_judges = [j for j in judges if j.get("sampled_hit")]
    total_judged = len(judges)
    unparsed_judged = sum(1 for j in judges if j["verdict"] == "unparsed")
    hit_wrong = sum(1 for j in hit_judges if j["verdict"] == "wrong")
    grader_wrong_misses = taxonomy["grader_wrong"]

    per_passage = []
    for tid, recs in sorted(group(lambda r: r["task_id"]).items()):
        block = _accuracy_block(recs)
        per_passage.append({
            "task_id": tid,
            "register": recs[0]["register"],
            "n": block["n"],
            "raw_correct": block["correct"],
            "raw_accuracy": block["accuracy"],
        })

    summary = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "questions_per_passage_plan": QUESTIONS_PER_PASSAGE,
        "overall_raw": _accuracy_block(answers),
        "bootstrap_raw_95": bootstrap_overall_ci(answers),
        "per_type_raw": per_type,
        "per_register_raw": {t: _accuracy_block(v) for t, v in sorted(
            group(lambda r: r["register"]).items())},
        "diagnostics": {
            "judge": {
                "misses_judged": len(judged_miss_keys),
                "hits_sampled": len(hit_judges),
                "verdicts_all": {v: sum(1 for j in judges
                                         if j["verdict"] == v)
                                 for v in VERDICTS + ("unparsed",)},
                "parse_rate": (round(
                    (total_judged - unparsed_judged) / total_judged, 6)
                    if total_judged else None),
                "grader_false_negative_rate": (
                    round(grader_wrong_misses / len(judged_miss_keys), 6)
                    if judged_miss_keys else None),
                "grader_false_positive_rate_estimate": (
                    round(hit_wrong / len(hit_judges), 6)
                    if hit_judges else None),
            },
            "miss_taxonomy": taxonomy,
            "judge_note": (
                "judge output is diagnostic only (SPEC-VALIDATION): no "
                "verdict feeds any correctness value; the single "
                "headline is grader v2 raw (overall_raw)."
            ),
        },
        "dedupe": {
            "kept": len(answers),
            "dropped": len(drops),
            "threshold": DEDUPE_THRESHOLD,
            "drop_rate": (round(len(drops) / (len(drops) + len(answers)), 6)
                          if (drops or answers) else None),
        },
        "per_passage": per_passage,
    }
    return summary


# --- generation / answering / judging ----------------------------------------


def _valid_items(parsed, allowed_types, fallback_types, task_id, ledger):
    """Validate parsed generation items; repair a missing/unknown type by
    position (recorded via a type_repair ledger record, never silent)."""
    items = []
    if not isinstance(parsed, list):
        return items
    for i, entry in enumerate(parsed):
        if not (isinstance(entry, dict) and isinstance(entry.get("q"), str)
                and isinstance(entry.get("a"), str) and entry["q"] and entry["a"]):
            continue
        q_type = entry.get("type")
        if q_type not in allowed_types:
            fallback = (fallback_types[i] if i < len(fallback_types)
                        else allowed_types[0])
            ledger.append({
                "type": "type_repair", "task_id": task_id,
                "declared_type": q_type, "repaired_type": fallback,
                "question": entry["q"],
            })
            q_type = fallback
        items.append({"q": entry["q"], "a": entry["a"], "q_type": q_type})
    return items


def generate_stratified_quiz(client, model, task, max_tokens, ledger):
    """Two generation calls of 12 (half1: 12 fact; half2: 4+4+4), then
    the deterministic cross-half dedupe. SPEC-VALIDATION §2: every
    generated item is anchor-validated against the passage BEFORE it
    can be asked — unanchored items are rejected and returned for
    logging; the per-type quota must be fillable from the anchored
    pool (one retry round first) or the run fails hard. Returns
    (kept, dropped, unanchored). Generation itself stays LLM
    authoring; acceptance is code-only."""
    halves = [
        ("gen_half1", GEN_HALF1_USER.format(passage=task["passage"]),
         HALF1_PLAN),
        ("gen_half2", GEN_HALF2_USER.format(passage=task["passage"]),
         HALF2_PLAN),
    ]
    items, kept, dropped = [], [], []
    for _round in range(1 + GEN_RETRY_ROUNDS):
        for call_name, user, plan in halves:
            allowed = tuple(t for t, _ in plan)
            fallback = [t for t, n in plan for _ in range(n)]
            result = call_model(client, model, GEN_SYSTEM, user, max_tokens,
                                parse=parse_json_block)
            ledger.append({
                "type": "model_call", "call": call_name,
                "task_id": task["id"], "raw": result["raw"],
                "parse_error": result["parse_error"],
                "usage": result["usage"], "latency_ms": result["latency_ms"],
            })
            if result["parse_error"] is not None:
                ledger.append({
                    "type": "parse_error", "call": call_name,
                    "task_id": task["id"],
                    "parse_error": result["parse_error"],
                    "raw": result["raw"],
                })
                continue
            items.extend(_valid_items(result["parsed"], allowed, fallback,
                                      task["id"], ledger))
        anchored_items = [it for it in items
                          if anchored(task["passage"], it["a"])]
        kept, dropped = dedupe_questions(anchored_items)
        counts = {}
        for it in kept:
            counts[it["q_type"]] = counts.get(it["q_type"], 0) + 1
        if all(counts.get(t, 0) >= n for t, n in QUOTA.items()):
            break
    counts = {}
    for it in kept:
        counts[it["q_type"]] = counts.get(it["q_type"], 0) + 1
    deficient = {t: counts.get(t, 0) for t, n in QUOTA.items()
                 if counts.get(t, 0) < n}
    if deficient:
        raise BaselineError(
            f"quota impossible from anchored pool for task {task['id']!r} "
            f"after {1 + GEN_RETRY_ROUNDS} generation rounds "
            f"(have {counts}, need {QUOTA})")
    unanchored = [it for it in items
                  if not anchored(task["passage"], it["a"])]
    return kept, dropped, unanchored


def run_baseline(client, model, max_tokens, results_dir, tasks_dir=None,
                 passages=None, task_offset=0, task_limit=None,
                 slice_id=None, judge_sample_hits=JUDGE_SAMPLE_HITS,
                 max_seconds=DEFAULT_MAX_SECONDS):
    """Baseline deep measurement over tasks[offset:offset+limit] of the
    FULL (optionally passages-capped) list — slice math is over the
    unsliced list, so parallel slices tile the serial run exactly.
    slice_id redirects output to slices/slice_<id>.* with identical
    record shape."""
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
    run_meta = {"run_id": f"baseline_{stamp}", "git_sha": git_sha(),
                "model": model}
    if slice_id:
        run_meta["slice_id"] = slice_id
    ledger = Ledger(ledger_path, run_meta)
    deadline = time.monotonic() + max_seconds

    all_tasks = load_tasks(tasks_dir or TASKS2_DIR)
    if passages is not None:
        if passages < 0:
            raise BaselineError(f"--passages {passages} must be >= 0")
        all_tasks = all_tasks[:passages]
    if task_offset < 0 or task_offset > len(all_tasks):
        raise BaselineError(
            f"--task-offset {task_offset} out of range "
            f"(0..{len(all_tasks)} passages)")
    end = None if task_limit is None else task_offset + task_limit
    tasks = all_tasks[task_offset:end]
    if not tasks:
        raise BaselineError(f"no passages selected (offset={task_offset}, "
                            f"limit={task_limit})")
    task_by_id = {t["id"]: t for t in tasks}

    # SPEC-VALIDATION §2 bank gate: only passage-anchored bank items
    # may enter the run; exclusions are logged, never silent.
    bank_rejects = []
    anchored_tasks = []
    for t in tasks:
        if anchored(t["passage"], t["expected_answer"]):
            anchored_tasks.append(t)
        else:
            bank_rejects.append({
                "task_id": t["id"],
                "expected_answer": t["expected_answer"],
                "anchor_score": round(anchor_score(
                    t["passage"], t["expected_answer"]), 6),
            })
    n_selected = len(tasks)
    if bank_rejects:
        out_base = os.path.dirname(os.path.abspath(ledger_path))
        with open(os.path.join(out_base, "unanchored_bank_items.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(bank_rejects, fh, sort_keys=True, indent=2,
                      ensure_ascii=False)
            fh.write("\n")
        tasks = anchored_tasks
    if not tasks:
        raise BaselineError(
            f"no anchored passages remain ({len(bank_rejects)} excluded)")

    answers, drops, gen_rejects = [], [], []
    for task in tasks:
        check_deadline(deadline, "generate")
        kept, dropped, unanchored = generate_stratified_quiz(
            client, model, task, max_tokens, ledger)
        for it in unanchored:
            ledger.append({
                "type": "anchor_reject", "task_id": task["id"],
                "q_type": it["q_type"], "question": it["q"],
                "answer": it["a"],
            })
            gen_rejects.append({
                "task_id": task["id"], "q_type": it["q_type"],
                "question": it["q"], "answer": it["a"],
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
            check_deadline(deadline, "quiz")
            result = call_model(
                client, model, QA_SYSTEM,
                QA_PROMPT.format(context=task["passage"],
                                 question=item["q"]),
                max_tokens,
            )
            g = grade_v2(task["passage"], item["q"], item["a"],
                         result["raw"])
            rec = {
                "type": "quiz_answer", "task_id": task["id"],
                "register": task["register"],
                "len_class": task["len_class"],
                "q_index": qi, "q_type": item["q_type"],
                "question": item["q"], "expected": item["a"],
                "raw_answer": result["raw"], "anchored": g["anchored"],
                "grade_v2": g["diagnostics"],
                "correct": g["correct"],
            }
            ledger.append({
                "type": "model_call", "call": "quiz_qa",
                "task_id": task["id"], "q_index": qi,
                "raw": result["raw"], "usage": result["usage"],
                "latency_ms": result["latency_ms"],
                "correct": g["correct"], "expected": item["a"],
            })
            ledger.append(rec)
            answers.append(rec)

    # Judge pass: EVERY miss + a seeded random sample of hits.
    misses = [r for r in answers if not r["correct"]]
    hits = sorted((r for r in answers if r["correct"]),
                  key=lambda r: (r["task_id"], r["q_index"]))
    rng = random.Random(JUDGE_SEED)
    n_sample = min(judge_sample_hits, len(hits))
    sampled = {(r["task_id"], r["q_index"])
               for r in rng.sample(hits, n_sample)} if n_sample else set()
    judges = []
    to_judge = [(r, False) for r in misses]
    to_judge += [(r, True) for r in hits
                 if (r["task_id"], r["q_index"]) in sampled]
    for rec, is_hit in to_judge:
        check_deadline(deadline, "judge")
        passage = task_by_id[rec["task_id"]]["passage"]
        result = call_model(
            client, model, JUDGE_SYSTEM,
            JUDGE_USER.format(passage=passage, question=rec["question"],
                              expected=rec["expected"],
                              given=rec["raw_answer"]),
            max_tokens,
        )
        verdict = parse_verdict(result["raw"])
        judge_rec = {
            "type": "judge", "task_id": rec["task_id"],
            "q_index": rec["q_index"], "sampled_hit": is_hit,
            "verdict": verdict, "raw": result["raw"],
        }
        ledger.append({
            "type": "model_call", "call": "judge",
            "task_id": rec["task_id"], "q_index": rec["q_index"],
            "raw": result["raw"], "usage": result["usage"],
            "latency_ms": result["latency_ms"], "verdict": verdict,
        })
        ledger.append(judge_rec)
        judges.append(judge_rec)

    if gen_rejects:
        out_base = os.path.dirname(os.path.abspath(ledger_path))
        with open(os.path.join(out_base, "unanchored_generated_items.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(gen_rejects, fh, sort_keys=True, indent=2,
                      ensure_ascii=False)
            fh.write("\n")

    summary = summarize_baseline(answers, judges, drops,
                                 run_id=run_meta["run_id"])
    summary["passages"] = len(tasks)
    summary["bank_validation"] = {
        "selected": n_selected,
        "anchored": len(tasks),
        "excluded_unanchored": len(bank_rejects),
    }
    summary["generated_validation"] = {
        "rejected_unanchored": len(gen_rejects),
    }
    if slice_id:
        summary["slice_id"] = slice_id
        summary["task_offset"] = task_offset
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"[baseline] passages={len(tasks)} "
          f"raw={summary['overall_raw']['accuracy']} "
          f"-> {summary_path}")
    return summary


def merge_baseline_slices(results_dir):
    """Concatenate slices/slice_*.jsonl, recompute the summary from ALL
    records (deterministic, sorted) and write the combined
    ledger.jsonl + summary.json."""
    slices_dir = os.path.join(results_dir, "slices")
    if not os.path.isdir(slices_dir):
        raise BaselineError(f"no slices directory at {slices_dir}")
    names = sorted(n for n in os.listdir(slices_dir)
                   if re.fullmatch(r"slice_.+\.jsonl", n))
    if not names:
        raise BaselineError(f"no slice_*.jsonl files in {slices_dir}")
    records, answers, judges, drops = [], [], [], []
    for name in names:
        with open(os.path.join(slices_dir, name), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append(rec)
                if rec.get("type") == "quiz_answer":
                    answers.append(rec)
                elif rec.get("type") == "judge":
                    judges.append(rec)
                elif rec.get("type") == "dedupe_drop":
                    drops.append(rec)
    records.sort(key=lambda r: json.dumps(r, sort_keys=True,
                                          ensure_ascii=False))
    ledger_path = os.path.join(results_dir, "ledger.jsonl")
    with open(ledger_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True,
                                ensure_ascii=False) + "\n")
    summary = summarize_baseline(answers, judges, drops,
                                 run_id="baseline_merged")
    summary["passages"] = len({r["task_id"] for r in answers})
    summary["slices_merged"] = names
    with open(os.path.join(results_dir, "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"[baseline] merged {len(names)} slices, {len(answers)} answers "
          f"-> {os.path.join(results_dir, 'summary.json')}")
    return summary


# --- stub client (declared smoke mode; never a silent fallback) --------------


def make_stub_client():
    """Deterministic stub for smoke mode and tests: generation returns
    the requested number of tagged questions, judge replies 'wrong',
    quiz answers never match. Ledger records client use via run config;
    a declared stub, never a fallback."""

    def client(payload):
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        if "quiz questions" in system:
            n = 12
            m = re.search(r"Write (\d+)", user)
            if m:
                n = int(m.group(1))
            text = user.rsplit("Text:\n", 1)[1]
            anchor = " ".join(text.split()[:3])
            half2 = "inference" in user
            prefix = "h2" if half2 else "h1"
            cycle = ["numeric", "ordering", "inference"]
            items = []
            for i in range(n):
                q_type = ("fact" if not half2
                          else cycle[i % len(cycle)])
                tag = f"{q_type}" if half2 else "fact"
                items.append({
                    "q": f"{prefix}{i}a {prefix}{i}b {prefix}{i}c "
                         f"{prefix}{i}d {q_type}?",
                    "a": anchor, "type": tag,
                })
            answer = json.dumps(items)
        elif "judge" in system.casefold():
            answer = "wrong"
        elif "Answer:" in user:
            answer = "stub-answer"
        else:
            answer = "stub"
        pt = approx_count(user) + 8
        return {
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": 4,
                      "total_tokens": pt + 4},
        }

    return client


# --- CLI ---------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ab.baseline", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("baseline", "baseline-merge"):
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
        if name == "baseline":
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

    if args.cmd == "baseline-merge":
        try:
            merge_baseline_slices(args.results_dir)
        except BaselineError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    stub = args.stub or os.environ.get("TCB_BASELINE_STUB") == "1"
    client = make_stub_client() if stub else default_client

    try:
        # NOTE (expB 14475 regression class): every slice arg is passed
        # through to run_baseline EXPLICITLY; the CLI-level regression
        # test test_cli_dispatch_passes_offset_and_slice_args_14475
        # asserts this end of the contract.
        run_baseline(
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
    except (BaselineError, ExpABError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
