#!/usr/bin/env python3
"""Cross-model cablese legibility matrix (SPEC-CROSSMODEL.md).

Kill-test for the router/SaaS thesis: cbl2 showed GLM-5.3-Flash reads
its own cablese at 99.3% recovery. A router needs the same to hold
ACROSS model families: model A writes, model B reads. This experiment
measures the cross-model legibility matrix in both directions using
the EXISTING cbl2 artifacts (results/cablese_cbl2/) as read-only
inputs:

Direction A — "GLM writes, X reads": each OpenRouter reader panel
  answers the cbl2 24-Q sets given GLM's R-PLAIN (control) and
  R-CABLESE records. Headline per reader: recovery_ratio =
  raw_acc_v2(cablese) / raw_acc_v2(plain).

Direction B — "X writes, GLM reads": each OpenRouter writer panel
  writes plain and cablese records with the SAME frozen CABLESE_RULES
  prompt as cbl2; GLM answers the 24 Qs from each. Headline per
  writer: recovery_ratio + savings_pct (cablese vs plain record
  tokens).

SPEC-VALIDATION compliance (Amendment 1, Theory directive 2026-10-03
14:19): NO LLM judge anywhere in the truth path. Grading is grader v2
(passage-anchored, ab/grade.py) and overall_raw (v2) is the SINGLE
headline everywhere (recovery ratios and McNemar use raw pairs). The
reused cbl2 quiz items pass a load-time anchoring gate against their
source passage: unanchored items are rejected + logged; per-passage
answering quotas are drawn from the anchored pool only (a pool
smaller than quota records a reduced n, never a run failure).

Call routes: OpenRouter (env OR_API_KEY, OR_API_URL default
https://openrouter.ai/api/v1/chat/completions, model string per
payload; frozen panels with the 404-drop-with-note fallback rule — no
substitution) and GLM-5.3-Flash via the existing relay client
(ZAI_API_KEY/TCB_ZAI_API_URL/TCB_MODEL, same as cablese runs). The
per-call model id lands in every ledger model_call record.

CLI:
  python3 -m ab.crossmodel crossmodel --source-run <dir>
      [--results-dir D] [--stub] [--max-tokens M] [--model GLM]
      [--task-offset O] [--task-limit L] [--slice-id S] [--passages N]
  python3 -m ab.crossmodel crossmodel-merge --results-dir D

Slicing mirrors the cablese contract (slice math over the unsliced
sorted-passage list; parallel slices tile the serial run exactly). The
CLI dispatch passes --task-offset/--task-limit/--slice-id through to
run_crossmodel EXPLICITLY — the expB 14475 incident (dispatch dropping
slice args) is regression-tested at the CLI level (see
tests/test_crossmodel.py).
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ab.baseline import (  # noqa: E402
    summarize_baseline,
)
from ab.cablese import (  # noqa: E402
    ANSWER_FROM_RECORD_USER,
    RECORD_CABLESE_SYSTEM,
    RECORD_CABLESE_USER,
    RECORD_PLAIN_SYSTEM,
    RECORD_PLAIN_USER,
    _ratio,
    mcnemar_exact,
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
from ab.grade import anchored, anchor_score, grade_v2  # noqa: E402
from ab.instrument import (  # noqa: E402
    InstrumentError,
    TransientAPIError,
    call_model,
    default_client,
)
from ab.run_v21 import TASKS2_DIR, load_tasks  # noqa: E402
from tcb.tokencount import approx_count  # noqa: E402


class CrossModelError(Exception):
    """Explicit crossmodel-run failure (missing source records, bad
    slice, missing env)."""


# --- frozen panels (SPEC-CROSSMODEL; 404 -> drop with note, NO substitute) ----

READER_PANEL = (
    {"id": "R1", "model": "google/gemma-4-31b-it"},
    {"id": "R2", "model": "qwen/qwen3.8-27b"},
    {"id": "R3", "model": "nvidia/nemotron-3-super-120b-a12b"},
    {"id": "R4", "model": "google/gemma-4-26b-a4b-it"},
)

WRITER_PANEL = (
    {"id": "W1", "model": "google/gemma-4-31b-it"},
    {"id": "W2", "model": "qwen/qwen3.8-27b"},
    {"id": "W3", "model": "openai/gpt-5-mini"},
)

REFERENCE_GLM_RECOVERY_RATIO = 0.997  # cbl2 headline (FINDINGS 2026-10-03)

DIR_A_CONDS = ("A-FROM-PLAIN", "A-FROM-CABLESE")
DIR_B_RECORD_CONDS = ("R-PLAIN", "R-CABLESE")
DIR_B_ANSWER_CONDS = ("GLM-FROM-PLAIN", "GLM-FROM-CABLESE")

# cbl2 quiz sets are 24 Qs per passage; the crossmodel answering quota
# is drawn from the anchored pool only (SPEC-VALIDATION Amendment 1).
SOURCE_QUESTIONS_PER_PASSAGE = 24


def all_conditions():
    conds = []
    for panel in READER_PANEL:
        for c in DIR_A_CONDS:
            conds.append(f"{c}/{panel['id']}")
    for panel in WRITER_PANEL:
        for c in DIR_B_ANSWER_CONDS:
            conds.append(f"{c}/{panel['id']}")
    return tuple(conds)


# --- OpenRouter client (env-driven; 404 -> ModelUnavailableError) --------------


OR_API_URL_DEFAULT = "https://openrouter.ai/api/v1/chat/completions"


class ModelUnavailableError(InstrumentError):
    """The provider returned HTTP 404 for the requested model id at
    submit — panel must be dropped with a note, never substituted."""


class ModelUnusableError(ModelUnavailableError):
    """HTTP 200 but the model yields no usable content — the
    reasoning-model shape (canary 14659: choices[0].message.content
    None/empty even after one immediate retry of the same payload).
    Subclasses ModelUnavailableError so the frozen-panel drop machinery
    catches it: drop with a note carrying the reason, never
    substitution."""

    def __init__(self, message, model=None):
        super().__init__(message)
        self.model = model


def _drop_status(exc):
    if isinstance(exc, ModelUnusableError):
        return (f"empty content (reasoning-model shape): "
                f"model {exc.model!r} empty after one retry")
    return "HTTP 404 at submit"


# --- canary 14669: adaptive reasoning-disable param -----------------------------
#
# At least one panel endpoint REQUIRES reasoning and hard-rejects the
# unified reasoning-disable param with HTTP 400 ("Reasoning is
# mandatory for this endpoint and cannot be disabled."). Adaptation:
# keep sending "reasoning": {"enabled": false} by default; on a 400
# whose body matches (case-insensitive) "reasoning" AND ("mandatory"
# OR "cannot be disabled"), retry the SAME payload once WITHOUT the
# reasoning key and record the model id here so every subsequent
# payload for that model omits the param from the start. Per-process
# memory is fine: the 24 array slots share nothing and each slot
# re-learns in one call.

_REASONING_MANDATORY = set()


class ORBadRequestError(InstrumentError):
    """HTTP 400 from OpenRouter; carries the response detail so the
    adaptive reasoning-disable machinery can pattern-match it. Any 400
    that is NOT the reasoning-mandatory shape propagates unchanged (no
    special-casing)."""

    def __init__(self, message):
        super().__init__(message)
        self.detail = message


def _reasoning_mandatory_detail(detail):
    lower = detail.lower()
    return ("reasoning" in lower
            and ("mandatory" in lower or "cannot be disabled" in lower))


def _with_reasoning_mode(payload, mode):
    """Apply an explicit reasoning mode ('effort_low' etc.) for models
    where the unified disable is rejected (gpt-5-mini: 'Reasoning is
    mandatory for this endpoint and cannot be disabled', 400)."""
    if mode == "effort_low":
        return {**payload,
                "reasoning": {"effort": "low", "exclude": True}}
    raise ValueError(f"unknown reasoning_mode {mode!r}")


def _with_reasoning_disable(payload):
    """Pin the unified OpenRouter reasoning-disable param unless the
    model is known reasoning-mandatory (then omit it entirely)."""
    if payload.get("model") in _REASONING_MANDATORY:
        return {k: v for k, v in payload.items() if k != "reasoning"}
    return {**payload, "reasoning": {"enabled": False}}


def make_openrouter_client(reasoning_mode=None):
    """Minimal stdlib HTTP client for the OpenRouter endpoint. The
    endpoint URL comes from OR_API_URL (default
    https://openrouter.ai/api/v1/chat/completions); the key must be in
    OR_API_KEY. 429/5xx (and opted-in 401) raise TransientAPIError for
    the standard backoff; 404 raises ModelUnavailableError (frozen
    panel fallback rule); 400 raises ORBadRequestError (the adaptive
    reasoning-disable machinery pattern-matches its detail). Every
    OR-bound payload carries the unified OpenRouter reasoning-disable
    param (reasoning: {"enabled": false}) — these calls are
    single-question QA, no chain-of-thought wanted; providers that
    ignore the param are unaffected, and models known to hard-require
    reasoning (canary 14669) omit the param instead."""

    def client(payload):
        key = os.environ.get("OR_API_KEY")
        if not key:
            sys.exit(
                "error: OR_API_KEY is not set in the environment; "
                "export OR_API_KEY=<your key> and re-run."
            )
        payload = _with_reasoning_disable(payload) if reasoning_mode is None \
            else _with_reasoning_mode(payload, reasoning_mode)
        url = os.environ.get("OR_API_URL", OR_API_URL_DEFAULT)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code == 404:
                raise ModelUnavailableError(
                    f"OpenRouter returned HTTP 404 for model "
                    f"{payload.get('model')!r}: {detail}")
            if exc.code == 400:
                raise ORBadRequestError(
                    f"OpenRouter returned HTTP 400 for model "
                    f"{payload.get('model')!r}: {detail}")
            transient_401 = (
                exc.code == 401
                and os.environ.get("TCB_TREAT_401_TRANSIENT") == "1"
            )
            if exc.code == 429 or transient_401 or 500 <= exc.code < 600:
                raise TransientAPIError(
                    f"OpenRouter returned HTTP {exc.code}: {detail}",
                    retry_after=(exc.headers.get("Retry-After")
                                 if exc.headers else None),
                )
            sys.exit(f"error: OpenRouter returned HTTP {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            sys.exit(f"error: OpenRouter unreachable: {exc}")

    return client


# --- canary 14659: reasoning-model content=None guard ---------------------------


def _wrap_or_client(client, ledger):
    """Wrap an OR client with the empty-content guard (canary 14659:
    a reasoning-family panel — qwen3.8 / nemotron-reasoning class —
    returned choices[0].message.content = None with HTTP 200 and the
    instrument died with "model content is not a string: None").

    After HTTP 200, content = message.get("content"). If content is
    None/empty: (a) a truthy message["refusal"] raises RuntimeError
    with the refusal text (hard, no drop — content-policy issues must
    surface loudly); (b) otherwise retry the SAME payload once
    immediately (some providers transiently return empty), recording a
    type="empty_content_retry" ledger event either way; if the retry
    is also empty -> ModelUnusableError, which the existing
    frozen-panel drop machinery catches (drop with note, never
    substitution). Also pins the reasoning-disable param on every
    OR-bound payload (single-question QA, no chain-of-thought) —
    adaptively: a 400 whose body matches the reasoning-mandatory
    shape (canary 14669) retries the SAME payload once WITHOUT the
    reasoning key, records the model in _REASONING_MANDATORY (so all
    later payloads for that model omit the param from the start) and
    appends a type="reasoning_param_dropped" ledger event. Any other
    400 propagates unchanged."""

    def _message(response):
        try:
            msg = response["choices"][0]["message"]
            return msg if isinstance(msg, dict) else {}
        except (KeyError, IndexError, TypeError):
            return {}

    def _refusal_check(message, payload):
        refusal = message.get("refusal")
        if refusal:
            raise RuntimeError(
                f"OpenRouter refusal (model "
                f"{payload.get('model')!r}): {refusal}")

    def wrapped(payload):
        payload = _with_reasoning_disable(payload)
        try:
            response = client(payload)
        except ORBadRequestError as exc:
            if ("reasoning" not in payload
                    or not _reasoning_mandatory_detail(exc.detail)):
                raise
            model = payload.get("model")
            _REASONING_MANDATORY.add(model)
            ledger.append({"type": "reasoning_param_dropped",
                           "model": model})
            bare = {k: v for k, v in payload.items()
                    if k != "reasoning"}
            payload = bare
            response = client(bare)  # one retry, param dropped
        message = _message(response)
        if message.get("content"):
            return response
        _refusal_check(message, payload)
        response = client(payload)  # one immediate retry, same payload
        message = _message(response)
        if message.get("content"):
            ledger.append({"type": "empty_content_retry",
                           "model": payload.get("model"),
                           "recovered": True})
            return response
        _refusal_check(message, payload)
        ledger.append({"type": "empty_content_retry",
                       "model": payload.get("model"),
                       "recovered": False})
        raise ModelUnusableError("empty content (reasoning-model shape)",
                                 model=payload.get("model"))

    return wrapped


# --- source-run input contract (read-only, guarded) ----------------------------


def load_source_run(source_dir, tasks_by_id=None,
                    quota_per_passage=None):
    """Load the cbl2 artifacts: quiz_item + R-PLAIN + R-CABLESE records
    from <source_dir>/ledger.jsonl. HARD-REFUSES (CrossModelError) if
    the directory/ledger is missing or any passage lacks its required
    records. Returns {task_id: {"questions": [...sorted by q_index],
    "R-PLAIN": text, "R-CABLESE": text, "register", "len_class"}}.

    SPEC-VALIDATION §2 load-time gate (Amendment 1): when tasks_by_id
    is given, every REUSED quiz item is validated with grade_v2's
    anchor machinery (anchored(passage, answer)) against its source
    passage; unanchored items are rejected (slot["unanchored"], never
    asked) and the per-passage quota is drawn from the anchored pool
    only (first quota_per_passage items by q_index; a pool smaller
    than quota yields a reduced n, never a failure)."""
    if not os.path.isdir(source_dir):
        raise CrossModelError(f"source run not found: {source_dir}")
    ledger_path = os.path.join(source_dir, "ledger.jsonl")
    if not os.path.isfile(ledger_path):
        raise CrossModelError(f"source ledger missing: {ledger_path}")
    passages = {}
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            tid = rec.get("task_id")
            if rec.get("type") == "quiz_item":
                slot = passages.setdefault(tid, {})
                slot.setdefault("questions", []).append({
                    "q_index": rec["q_index"], "q_type": rec["q_type"],
                    "q": rec["question"], "a": rec["answer"],
                })
            elif (rec.get("type") == "record"
                  and rec.get("condition") in DIR_B_RECORD_CONDS):
                slot = passages.setdefault(tid, {})
                slot[rec["condition"]] = rec["text"]
                slot["register"] = rec.get("register")
                slot["len_class"] = rec.get("len_class")
    if not passages:
        raise CrossModelError(
            f"no quiz_item/record entries in {ledger_path}")
    missing = []
    for tid, slot in sorted(passages.items()):
        if not slot.get("questions"):
            missing.append(f"{tid}: no quiz items")
        for cond in DIR_B_RECORD_CONDS:
            if not slot.get(cond):
                missing.append(f"{tid}: no {cond} record")
    if missing:
        raise CrossModelError(
            f"source run {source_dir} incomplete ({len(missing)} gap(s), "
            f"e.g. {missing[:3]}) — refusing to start")
    for tid, slot in passages.items():
        slot["questions"].sort(key=lambda it: it["q_index"])
        # anchoring gate (skipped for passages absent from the tasks
        # bank — the slice guard raises on those instead)
        task = tasks_by_id.get(tid) if tasks_by_id else None
        if task is None:
            slot["unanchored"] = []
            slot["anchored_pool"] = len(slot["questions"])
            continue
        kept, rejects = [], []
        for it in slot["questions"]:
            if anchored(task["passage"], it["a"]):
                kept.append(it)
            else:
                rejects.append({
                    "task_id": tid, "q_index": it["q_index"],
                    "q_type": it["q_type"], "question": it["q"],
                    "answer": it["a"],
                    "anchor_score": round(anchor_score(
                        task["passage"], it["a"]), 6),
                })
        slot["questions"] = (kept[:quota_per_passage]
                             if quota_per_passage is not None else kept)
        slot["unanchored"] = rejects
        slot["anchored_pool"] = len(kept)
    return passages


# --- v2-only headline (SPEC-VALIDATION Amendment 1: no LLM judge in
#     the truth path; overall_raw under grader v2 is the single
#     headline — computed on top of the reused baseline block; judge
#     verdicts, if any exist in the record stream, stay quarantined
#     under diagnostics/ by summarize_baseline and never touch a
#     headline field) ---------------------------------------------------------


def _condition_block(answers, judges):
    """summarize_baseline block: overall_raw (v2) headline machinery,
    CIs, taxonomy; judge data quarantined under diagnostics/."""
    return summarize_baseline(answers, judges, [])


def _pair_hits(answers):
    return {(r["task_id"], r["q_index"]): r["correct"] for r in answers}


def _panel_note(drop):
    return (f"panel {drop['panel']} model {drop['model']!r} unavailable "
            f"at submit ({drop['status']}); dropped with this note, "
            f"NOT substituted (frozen-panel rule)")


# --- summary ------------------------------------------------------------------


def summarize_crossmodel(answers_by_cond, judges_by_cond, writer_records,
                         panel_drops, source_run=None, run_id="crossmodel",
                         anchoring_gate=None):
    """Compute summary.json content purely from record lists (used by
    both the serial/slice runs and crossmodel-merge, so merge is
    serial-equivalent by construction). overall_raw (grader v2) is the
    single headline; judge records (diagnostics only) never touch it."""
    conditions = {}
    for cond in all_conditions():
        conditions[cond] = _condition_block(
            answers_by_cond.get(cond, []),
            judges_by_cond.get(cond, []))

    dropped = {d["panel"] for d in panel_drops}

    def acc(cond):
        block = conditions.get(cond)
        if block is None:
            return None
        return block["overall_raw"]["accuracy"]

    direction_a = {}
    mcnemar = {}
    for panel in READER_PANEL:
        pid = panel["id"]
        plain_a = acc(f"A-FROM-PLAIN/{pid}")
        cablese_a = acc(f"A-FROM-CABLESE/{pid}")
        entry = {
            "model": panel["model"],
            "plain_accuracy": plain_a,
            "cablese_accuracy": cablese_a,
            "recovery_ratio": _ratio(cablese_a, plain_a),
            "reference_glm_recovery_ratio": REFERENCE_GLM_RECOVERY_RATIO,
        }
        if pid in dropped:
            entry["dropped"] = True
            entry["drop_note"] = _panel_note(
                next(d for d in panel_drops if d["panel"] == pid))
        direction_a[pid] = entry
        mcnemar[f"A:{pid}"] = mcnemar_exact(
            _pair_hits(answers_by_cond.get(f"A-FROM-CABLESE/{pid}", [])),
            _pair_hits(answers_by_cond.get(f"A-FROM-PLAIN/{pid}", [])))

    direction_b = {}
    for panel in WRITER_PANEL:
        pid = panel["id"]
        plain_a = acc(f"GLM-FROM-PLAIN/{pid}")
        cablese_a = acc(f"GLM-FROM-CABLESE/{pid}")
        rec_plain = {r["task_id"]: r for r in writer_records
                     if r.get("condition") == f"R-PLAIN/{pid}"}
        rec_cablese = {r["task_id"]: r for r in writer_records
                       if r.get("condition") == f"R-CABLESE/{pid}"}
        tokens = {}
        for tid in sorted(set(rec_plain) & set(rec_cablese)):
            tokens[tid] = {
                "plain_approx_tokens": rec_plain[tid]["approx_tokens"],
                "cablese_approx_tokens":
                    rec_cablese[tid]["approx_tokens"],
            }
        plain_total = sum(v["plain_approx_tokens"]
                          for v in tokens.values())
        cablese_total = sum(v["cablese_approx_tokens"]
                            for v in tokens.values())
        entry = {
            "model": panel["model"],
            "plain_accuracy": plain_a,
            "cablese_accuracy": cablese_a,
            "recovery_ratio": _ratio(cablese_a, plain_a),
            "writer_record_tokens": {
                "per_passage": [{"task_id": tid, **vals}
                                for tid, vals in sorted(tokens.items())],
                "plain_approx_tokens_total": plain_total,
                "cablese_approx_tokens_total": cablese_total,
            },
            "savings_pct": (round(100.0 * (1.0 - cablese_total
                                           / plain_total), 6)
                            if plain_total else None),
        }
        if pid in dropped:
            entry["dropped"] = True
            entry["drop_note"] = _panel_note(
                next(d for d in panel_drops if d["panel"] == pid))
        direction_b[pid] = entry
        mcnemar[f"B:{pid}"] = mcnemar_exact(
            _pair_hits(answers_by_cond.get(f"GLM-FROM-CABLESE/{pid}", [])),
            _pair_hits(answers_by_cond.get(f"GLM-FROM-PLAIN/{pid}", [])))

    summary = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "source_run": source_run,
        "panels": {"readers": [dict(p) for p in READER_PANEL],
                   "writers": [dict(p) for p in WRITER_PANEL]},
        "panel_drops": sorted(panel_drops,
                              key=lambda d: d["panel"]),
        "conditions": conditions,
        "direction_a": direction_a,
        "direction_b": direction_b,
        "mcnemar": mcnemar,
        "anchoring_gate": anchoring_gate or {
            "rejected_unanchored": 0, "passages": {}},
    }
    return summary


# --- experiment ----------------------------------------------------------------


def _slice_tasks(source_passages, tasks_by_id, passages_cap, task_offset,
                 task_limit):
    all_ids = sorted(source_passages)
    if passages_cap is not None:
        if passages_cap < 0:
            raise CrossModelError(f"--passages {passages_cap} must be >= 0")
        all_ids = all_ids[:passages_cap]
    if task_offset < 0 or task_offset > len(all_ids):
        raise CrossModelError(
            f"--task-offset {task_offset} out of range "
            f"(0..{len(all_ids)} passages)")
    end = None if task_limit is None else task_offset + task_limit
    ids = all_ids[task_offset:end]
    if not ids:
        raise CrossModelError(f"no passages selected (offset={task_offset}, "
                              f"limit={task_limit})")
    missing = [tid for tid in ids if tid not in tasks_by_id]
    if missing:
        raise CrossModelError(
            f"source passages absent from the tasks bank: {missing[:3]}")
    return ids


def run_crossmodel(or_client, glm_client, glm_model, max_tokens,
                   results_dir, source_run, tasks_dir=None,
                   passages=None, task_offset=0, task_limit=None,
                   slice_id=None,
                   quota_per_passage=SOURCE_QUESTIONS_PER_PASSAGE,
                   max_seconds=DEFAULT_MAX_SECONDS):
    """Cross-model legibility matrix over tasks[offset:offset+limit] of
    the FULL (optionally passages-capped) sorted source-passage list —
    slice math over the unsliced list, so parallel slices tile the
    serial run exactly (cablese contract). slice_id redirects output
    to slices/slice_<id>.* with identical record shape. The reused
    cbl2 quiz items pass the SPEC-VALIDATION anchoring gate
    (quota_per_passage drawn from the anchored pool only; reduced n
    recorded, never a run failure)."""
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
    run_meta = {"run_id": f"crossmodel_{stamp}", "git_sha": git_sha(),
                "glm_model": glm_model, "source_run":
                    os.path.abspath(source_run)}
    if slice_id:
        run_meta["slice_id"] = slice_id
    ledger = Ledger(ledger_path, run_meta)
    or_client = _wrap_or_client(or_client, ledger)
    deadline = time.monotonic() + max_seconds

    tasks_by_id = {t["id"]: t for t in load_tasks(tasks_dir or TASKS2_DIR)}
    source_passages = load_source_run(
        source_run, tasks_by_id=tasks_by_id,
        quota_per_passage=quota_per_passage)
    ids = _slice_tasks(source_passages, tasks_by_id, passages,
                       task_offset, task_limit)

    # SPEC-VALIDATION §2 gate bookkeeping: rejected items are logged,
    # never silent (ledger record + unanchored_source_items.json next
    # to the ledger, baseline conventions); per-passage reduced n is
    # recorded in the summary, not failed on.
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

    conds = all_conditions()
    answers_by_cond = {c: [] for c in conds}
    judges_by_cond = {c: [] for c in conds}
    writer_records = []
    panel_drops = []
    dropped = set()

    def _note_drop(panel, status="HTTP 404 at submit"):
        drop = {"type": "panel_drop", "panel": panel["id"],
                "model": panel["model"], "status": status}
        ledger.append(drop)
        panel_drops.append(drop)
        dropped.add(panel["id"])

    def call_or(panel, system, user):
        """OR call with the frozen-panel 404 fallback rule: a 404 at
        submit drops the panel with a note (never substitutes)."""
        if panel["id"] in dropped:
            raise ModelUnavailableError("panel already dropped")
        try:
            return call_model(or_client, panel["model"], system, user,
                              max_tokens)
        except ModelUnavailableError as exc:
            if panel["id"] not in dropped:
                _note_drop(panel, _drop_status(exc))
            raise

    def answer_call(client, model, cond, panel_id, panel_model, task,
                    item, record_text):
        system, user = QA_SYSTEM, ANSWER_FROM_RECORD_USER.format(
            record=record_text, question=item["q"])
        try:
            result = call_model(client, model, system, user, max_tokens)
        except ModelUnavailableError as exc:
            panel = {"id": panel_id, "model": panel_model}
            if panel_id not in dropped:
                _note_drop(panel, _drop_status(exc))
            raise
        g = grade_v2(task["passage"], item["q"], item["a"], result["raw"])
        rec = {
            "type": "cond_answer", "condition": cond,
            "direction": "A" if cond.startswith("A-FROM") else "B",
            "panel": panel_id, "panel_model": panel_model,
            "task_id": task["id"], "register": task["register"],
            "len_class": task["len_class"],
            "q_index": item["q_index"], "q_type": item["q_type"],
            "question": item["q"], "expected": item["a"],
            "raw_answer": result["raw"], "anchored": g["anchored"],
            "grade_v2": g["diagnostics"], "correct": g["correct"],
        }
        ledger.append({
            "type": "model_call", "call": "cond_qa", "condition": cond,
            "panel": panel_id, "model": model,
            "task_id": task["id"], "q_index": item["q_index"],
            "raw": result["raw"], "usage": result["usage"],
            "latency_ms": result["latency_ms"],
            "correct": g["correct"], "expected": item["a"],
        })
        ledger.append(rec)
        answers_by_cond[cond].append(rec)

    for tid in ids:
        task = tasks_by_id[tid]
        src = source_passages[tid]
        items = src["questions"]

        # Direction A: each reader answers from GLM's cbl2 records.
        for panel in READER_PANEL:
            if panel["id"] in dropped:
                continue
            for cond_key, rec_key in (
                ("A-FROM-PLAIN", "R-PLAIN"),
                ("A-FROM-CABLESE", "R-CABLESE"),
            ):
                for item in items:
                    check_deadline(deadline, "direction_a")
                    try:
                        answer_call(or_client, panel["model"],
                                    f"{cond_key}/{panel['id']}",
                                    panel["id"], panel["model"], task,
                                    item, src[rec_key])
                    except ModelUnavailableError:
                        break

        # Direction B: each writer records, GLM answers.
        for panel in WRITER_PANEL:
            if panel["id"] in dropped:
                continue
            rec_texts = {}
            for cond_key, system, user_tmpl in (
                ("R-PLAIN", RECORD_PLAIN_SYSTEM, RECORD_PLAIN_USER),
                ("R-CABLESE", RECORD_CABLESE_SYSTEM,
                 RECORD_CABLESE_USER),
            ):
                check_deadline(deadline, "direction_b_record")
                try:
                    result = call_or(panel, system, user_tmpl.format(
                        passage=task["passage"]))
                except ModelUnavailableError:
                    break
                rec_texts[cond_key] = result["raw"]
                rec = {
                    "type": "record",
                    "condition": f"{cond_key}/{panel['id']}",
                    "panel": panel["id"], "model": panel["model"],
                    "task_id": tid, "register": task["register"],
                    "len_class": task["len_class"],
                    "approx_tokens": approx_count(result["raw"]),
                    "usage_completion_tokens":
                        result["usage"]["completion_tokens"],
                    "text": result["raw"],
                }
                ledger.append({
                    "type": "model_call",
                    "call": f"record_{cond_key}",
                    "panel": panel["id"], "model": panel["model"],
                    "task_id": tid, "raw": result["raw"],
                    "usage": result["usage"],
                    "latency_ms": result["latency_ms"],
                })
                ledger.append(rec)
                writer_records.append(rec)
            for cond_key, rec_key in (
                ("GLM-FROM-PLAIN", "R-PLAIN"),
                ("GLM-FROM-CABLESE", "R-CABLESE"),
            ):
                if rec_key not in rec_texts:
                    continue  # writer panel dropped mid-record
                for item in items:
                    check_deadline(deadline, "direction_b_answer")
                    answer_call(glm_client, glm_model,
                                f"{cond_key}/{panel['id']}",
                                panel["id"], panel["model"], task,
                                item, rec_texts[rec_key])

    # SPEC-VALIDATION Amendment 1: NO judge pass. The truth path ends
    # at grader v2; any judge records found in a slice stream during
    # merge stay quarantined under diagnostics/ by summarize_baseline.

    summary = summarize_crossmodel(
        answers_by_cond, judges_by_cond, writer_records, panel_drops,
        source_run=os.path.abspath(source_run), run_id=run_meta["run_id"],
        anchoring_gate=anchoring_gate)
    summary["passages"] = len(ids)
    if slice_id:
        summary["slice_id"] = slice_id
        summary["task_offset"] = task_offset
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2,
                  ensure_ascii=False)
        fh.write("\n")
    ratios = {pid: v["recovery_ratio"]
              for pid, v in summary["direction_a"].items()}
    print(f"[crossmodel] passages={len(ids)} "
          f"recovery_ratios={json.dumps(ratios, sort_keys=True)} "
          f"panel_drops={len(panel_drops)} -> {summary_path}")
    return summary


def merge_crossmodel_slices(results_dir):
    """Concatenate slices/slice_*.jsonl, recompute the summary from ALL
    records (deterministic, sorted) and write the combined
    ledger.jsonl + summary.json."""
    slices_dir = os.path.join(results_dir, "slices")
    if not os.path.isdir(slices_dir):
        raise CrossModelError(f"no slices directory at {slices_dir}")
    names = sorted(n for n in os.listdir(slices_dir)
                   if re.fullmatch(r"slice_.+\.jsonl", n))
    if not names:
        raise CrossModelError(f"no slice_*.jsonl files in {slices_dir}")
    records_out, answers_by_cond = [], {c: [] for c in all_conditions()}
    judges_by_cond = {c: [] for c in all_conditions()}
    writer_records, panel_drops = [], []
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
                if rtype == "cond_answer":
                    answers_by_cond[rec["condition"]].append(rec)
                elif rtype == "judge":
                    judges_by_cond[rec["condition"]].append(rec)
                elif rtype == "record":
                    writer_records.append(rec)
                elif rtype == "panel_drop":
                    drop = {k: v for k, v in rec.items()
                            if k in ("panel", "model", "status")}
                    if drop not in panel_drops:
                        panel_drops.append(drop)
                elif rtype == "unanchored_reject":
                    gate_rejects.append(rec)
                elif rtype == "passage_n":
                    gate_passages[rec["task_id"]] = {
                        k: rec[k] for k in ("quota", "anchored_pool", "n")}
                if source_run is None and rec.get("source_run"):
                    source_run = rec["source_run"]
    records_out.sort(key=lambda r: json.dumps(r, sort_keys=True,
                                              ensure_ascii=False))
    ledger_path = os.path.join(results_dir, "ledger.jsonl")
    with open(ledger_path, "w", encoding="utf-8") as fh:
        for rec in records_out:
            fh.write(json.dumps(rec, sort_keys=True,
                                ensure_ascii=False) + "\n")
    summary = summarize_crossmodel(
        answers_by_cond, judges_by_cond, writer_records, panel_drops,
        source_run=source_run, run_id="crossmodel_merged",
        anchoring_gate={"rejected_unanchored": len(gate_rejects),
                        "passages": gate_passages})
    summary["passages"] = len({r["task_id"]
                               for rs in answers_by_cond.values()
                               for r in rs})
    summary["slices_merged"] = names
    with open(os.path.join(results_dir, "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2,
                  ensure_ascii=False)
        fh.write("\n")
    print(f"[crossmodel] merged {len(names)} slices, "
          f"{sum(len(v) for v in answers_by_cond.values())} answers "
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
    """Deterministic stub for smoke mode and tests: routes on the
    frozen prompt strings — plain records come out long, cablese
    records short (so writer savings are positive), everything else
    replies a fixed non-matching answer. No judge branch: the judge
    pass is gone (SPEC-VALIDATION Amendment 1). A declared stub,
    never a fallback."""

    def client(payload):
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        if "plain-English record" in system:
            return _stub_response(
                "The ship Aurora sailed from Bristol in March 1848 with "
                "a cargo of iron, a crew of nineteen, and Captain Elias "
                "Hoy as master, putting into Falmouth after a storm and "
                "then continuing to Lisbon and Porto, all recorded here "
                "in complete plain English prose.", user)
        if "telegraphese" in system and "record" in system:
            return _stub_response(
                "rcrd: aurora brstl 1848 iron crw 19 hoy mstr", user)
        return _stub_response("stub crossmodel answer", user)

    return client


# --- CLI -----------------------------------------------------------------------

DEFAULT_CROSSMODEL_MAX_TOKENS = 4000


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ab.crossmodel",
        description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("crossmodel", "crossmodel-merge"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL,
                       help="GLM reader model via the relay client")
        p.add_argument("--max-tokens", type=int,
                       default=DEFAULT_CROSSMODEL_MAX_TOKENS)
        p.add_argument("--max-seconds", type=float,
                       default=DEFAULT_MAX_SECONDS,
                       help="wall-clock guard on all-loops stages "
                            "(default %(default)s)")
        p.add_argument("--results-dir",
                       default=os.path.join(_ROOT, "results"))
        if name == "crossmodel":
            p.add_argument("--source-run", required=True,
                           help="completed cbl2 run dir (read-only input)")
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

    if args.cmd == "crossmodel-merge":
        try:
            merge_crossmodel_slices(args.results_dir)
        except CrossModelError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    stub = args.stub or os.environ.get("TCB_CROSSMODEL_STUB") == "1"
    or_client = make_stub_client() if stub else make_openrouter_client()
    glm_client = make_stub_client() if stub else default_client

    try:
        # NOTE (expB 14475 regression class): every slice arg is
        # passed through to run_crossmodel EXPLICITLY; the CLI-level
        # regression test test_cli_dispatch_passes_offset_and_slice_
        # args_14475_crossmodel asserts this end of the contract.
        run_crossmodel(
            or_client, glm_client, args.model, args.max_tokens,
            args.results_dir,
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
    except (CrossModelError, ExpABError, InstrumentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
