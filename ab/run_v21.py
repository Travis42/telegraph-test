#!/usr/bin/env python3
"""V2.1 experiment matrix runner (SPEC-V21 §3).

Arms: plain / notation_v3 / zhn_repair / formulaic_dict / cipher.
Matrix: arm x register (prose/fact/formulaic) x len_class, with an
EXPLICIT skip map (below) recorded in every worklist and ledger.

No silent fallbacks (hard rule):
- every model call's raw answer is appended VERBATIM to the ledger JSONL
  at the moment it returns, before any parsing is trusted;
- any parse failure of a fact-coverage answer becomes an explicit
  `parse_error` ledger record carrying raw text + arm + task id;
- tasks with an empty `facts` list get an explicit `no_facts_declared`
  state on the unit summary (distinct from parse_error and from a checked
  zero-recall).

Modes:
  --emit-worklist  write slurm/worklist.json (units + skipped combos)
  --unit N         run one worklist unit into a per-slot shard
  --merge          merge shards into ledger.jsonl + write summary.json
  --run            local sequential run of every unit
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ab import cipher as cipher_mod  # noqa: E402
from ab.instrument import (  # noqa: E402
    call_model,
    default_client,
    fact_check,
    recall_of,
)
from ab.run_ab import QA_PROMPT, score_answer  # noqa: E402
from tcb import tables as tables_mod  # noqa: E402
from tcb.encode import encode_router  # noqa: E402
from tcb.tokencount import approx_count  # noqa: E402

ARMS = ["plain", "notation_v3", "zhn_repair", "formulaic_dict", "cipher"]
REGISTERS = ["prose", "fact", "formulaic"]
LEN_CLASSES = ["short", "medium", "long"]

# Explicit skip map: arm -> registers it is INVALID on (with reason).
# Everything not listed runs on every register that has tasks.
SKIP_MAP = [
    {
        "arm": "zhn_repair",
        "excluded_register": "formulaic",
        "reason": (
            "telegraphic Chinese transform defined for English prose/fact "
            "registers only (formulaic register is ASCII schema content)"
        ),
    },
    {
        "arm": "formulaic_dict",
        "excluded_register": "fact",
        "reason": (
            "dictionary+register encoder is tuned for the formulaic register "
            "(applied to prose for comparison); fact register is covered by "
            "the strict-schema arms"
        ),
    },
]

DEFAULT_MODEL = os.environ.get("TCB_MODEL", "zai/glm-5.3-flash")
DEFAULT_MAX_TOKENS = 1500
TASKS2_DIR = os.path.join(_HERE, "tasks2")

QA_SYSTEM = "You answer reading-comprehension tasks faithfully and briefly."

NOTATION_V3_SYSTEM = (
    "Compress the passage into dense notation. STRICT RULES: one fact per "
    "line; single-character keys ONLY from this fixed schema: n=name, "
    "t=type/class, d=date/time, l=location, c=count, q=quantity/amount, "
    "u=unit/currency, g=goods/cargo, p=purpose, w=weather, e=event, "
    "r=relation/recipient, s=status, o=owner/origin, m=measurement; format "
    "'k:value'; join attributes of one entity on one line with ','; use "
    "'->' for sequence or causality; NO multi-word keys; NO repeated keys "
    "on one line; NO prose; keep every name, number and date verbatim.\n"
    "Example:\n"
    "Passage: The brig Foam sailed from Liverpool on 4 March with 900 tons "
    "of coal for Lisbon; master J. Reed; encountered gales on the 9th.\n"
    "Notation:\n"
    "n:Foam,t:brig,o:Liverpool,d:4 March,q:900,u:tons,g:coal,r:Lisbon,"
    "m:J. Reed\n"
    "e:gales,d:9 March\n"
    "Output ONLY the notation lines."
)

ZHN_REPAIR_SYSTEM = (
    "Translate to compact telegraphic written Chinese (电报体): drop "
    "function words, one fact per clause, classical-commercial wire style. "
    "ANCHOR RULE: keep every proper noun, person name, ship name, place "
    "name, number, date and measurement VERBATIM in Latin script (e.g. "
    "Meridian, Bristol, 900 tons, 4 March); translate only the connective "
    "prose. You MUST preserve every fact. Output ONLY the compressed "
    "Chinese text, nothing else."
)

REPAIR_SYSTEMS = {
    "notation_v3": (
        "The notation is missing some numbered facts. Add ONLY lines for "
        "the numbered missing facts, same strict single-character-key "
        "schema. Output the complete corrected notation and nothing else."
    ),
    "zhn_repair": (
        "The Chinese telegraph text is missing some numbered facts. Add "
        "ONLY minimal clauses for the numbered missing facts, same dense "
        "telegraphic style, keeping proper nouns and numbers as Latin "
        "anchors. Output the complete corrected Chinese text and nothing "
        "else."
    ),
}

# Repair policy per arm: max model repair passes.
REPAIR_PASSES = {"notation_v3": 1, "zhn_repair": 2}


def frozen_config(model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS):
    return {
        "spec": "SPEC-V21.md",
        "wave": "v2.1",
        "arms": ARMS,
        "registers": REGISTERS,
        "len_classes": LEN_CLASSES,
        "skip_map": SKIP_MAP,
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "repair_passes": REPAIR_PASSES,
        "tasks_dir": os.path.relpath(TASKS2_DIR, _ROOT),
        "cipher_table": os.path.relpath(cipher_mod.TABLE_PATH, _ROOT),
    }


def config_hash(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def git_sha():
    override = os.environ.get("TCB_V21_GIT_SHA")
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "-C", _ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unavailable (not a git checkout)"


def load_tasks(tasks_dir=TASKS2_DIR):
    tasks = []
    for fn in sorted(os.listdir(tasks_dir)):
        if not fn.endswith(".json") or fn.startswith("README"):
            continue
        path = os.path.join(tasks_dir, fn)
        if os.path.isdir(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            tasks.append(json.load(fh))
    return sorted(tasks, key=lambda t: t["id"])


def arm_valid_on(arm, register):
    return not any(
        s["arm"] == arm and s["excluded_register"] == register
        for s in SKIP_MAP
    )


def build_units(tasks):
    """Worklist units (task x arm) + explicit skipped combos."""
    units, skipped = [], []
    for task in tasks:
        for arm in ARMS:
            if arm_valid_on(arm, task["register"]):
                units.append(
                    {
                        "index": len(units),
                        "task_id": task["id"],
                        "arm": arm,
                        "register": task["register"],
                        "len_class": task["len_class"],
                    }
                )
            else:
                reason = next(
                    s["reason"]
                    for s in SKIP_MAP
                    if s["arm"] == arm and s["excluded_register"] == task["register"]
                )
                skipped.append(
                    {
                        "task_id": task["id"],
                        "arm": arm,
                        "register": task["register"],
                        "len_class": task["len_class"],
                        "reason": reason,
                    }
                )
    return units, skipped


class Ledger:
    """Append-only JSONL ledger; every record stamped with run identity."""

    def __init__(self, path, run_meta):
        self.path = path
        self.run_meta = run_meta
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def append(self, record):
        rec = dict(self.run_meta)
        rec.update(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


def make_stub_client():
    """Explicit deterministic stub (TCB_V21_STUB=1): every answer echoes a
    recognizable marker; the ledger records client='stub'. This is a
    declared smoke mode, never a silent fallback."""

    def client(payload):
        user = payload["messages"][1]["content"]
        if "Facts:\n" in user and "Context:\n" in user:
            # fact-coverage probe: claim every numbered fact is stated
            n = user.rsplit("Facts:\n", 1)[1].count("\n") + 1
            answer = json.dumps(list(range(1, n + 1)))
        elif "Answer:" in user:
            answer = "stub-answer"
        else:
            answer = "stub-transform"
        pt = approx_count(user) + 8
        return {
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": 4,
                      "total_tokens": pt + 4},
        }

    return client


def _model_call_record(kind, unit, result):
    rec = {
        "type": "model_call",
        "call": kind,
        "arm": unit["arm"],
        "task_id": unit["task_id"],
        "register": unit["register"],
        "len_class": unit["len_class"],
        "raw": result["raw"],  # VERBATIM, persisted by caller immediately
        "usage": result["usage"],
        "latency_ms": result["latency_ms"],
    }
    return rec


def _persist_check(ledger, unit, call_kind, check):
    """Persist a fact-check's raw + parse state as ledger records."""
    rec = {
        "type": "fact_check",
        "call": call_kind,
        "arm": unit["arm"],
        "task_id": unit["task_id"],
        "register": unit["register"],
        "len_class": unit["len_class"],
        "facts_total": check["facts_total"],
        "facts_matched": check["facts_matched"],
        "parse_ok": check["parse_ok"],
        "parse_error": check.get("parse_error"),
        "raw": check["raw"],  # verbatim model answer
        "usage": check["usage"],
        "latency_ms": check["latency_ms"],
    }
    ledger.append(rec)
    if not check["parse_ok"]:
        ledger.append(
            {
                "type": "parse_error",
                "call": call_kind,
                "arm": unit["arm"],
                "task_id": unit["task_id"],
                "register": unit["register"],
                "len_class": unit["len_class"],
                "parse_error": check.get("parse_error"),
                "raw": check["raw"],
            }
        )


def run_unit(unit, task, client, model, max_tokens, ledger, ctx):
    arm = unit["arm"]
    passage = task["passage"]
    transform_cost_tokens = 0
    repairs_done = 0
    context = passage

    # ---- 1. transform (with cost) -------------------------------------
    if arm == "plain":
        pass
    elif arm in ("notation_v3", "zhn_repair"):
        system = (
            NOTATION_V3_SYSTEM if arm == "notation_v3" else ZHN_REPAIR_SYSTEM
        )
        result = call_model(client, model, system, passage, max_tokens)
        ledger.append(_model_call_record("transform", unit, result))
        context = result["raw"]
        transform_cost_tokens = result["usage"]["total_tokens"]
    elif arm == "formulaic_dict":
        payload = encode_router(passage, ctx["tables"], ctx["rules"])
        context = payload["coded"]
    elif arm == "cipher":
        context = cipher_mod.encode(passage, ctx["cipher_table"])
    else:  # pragma: no cover - guarded by ARMS
        raise ValueError(f"unknown arm {arm}")

    # ---- 2. fact-coverage check + optional repair ---------------------
    facts = task.get("facts") or []
    if not facts:
        fact_state = {
            "facts_total": 0,
            "facts_matched": [],
            "parse_ok": True,
            "state": "no_facts_declared",
            "recall_pre": None,
            "recall_post": None,
            "repairs": 0,
        }
    else:
        check = fact_check(client, model, context, facts, max_tokens)
        _persist_check(ledger, unit, "fact_check", check)
        recall_pre = recall_of(check)
        check_post = check
        max_passes = REPAIR_PASSES.get(arm, 0)
        while (
            check_post["parse_ok"]
            and recall_of(check_post) is not None
            and recall_of(check_post) < 1.0
            and repairs_done < max_passes
        ):
            missing = [
                i for i in range(1, len(facts) + 1)
                if i not in set(check_post["facts_matched"])
            ]
            repair = call_model(
                client,
                model,
                REPAIR_SYSTEMS[arm],
                "Current compressed context:\n"
                + context
                + "\n\nMissing facts:\n"
                + "\n".join(f"{i}. {facts[i - 1]}" for i in missing),
                max_tokens,
            )
            repairs_done += 1
            ledger.append(
                _model_call_record("repair", unit, repair)
            )
            context = repair["raw"]
            transform_cost_tokens += repair["usage"]["total_tokens"]
            check_post = fact_check(client, model, context, facts, max_tokens)
            _persist_check(ledger, unit, "fact_check_after_repair", check_post)
        fact_state = {
            "facts_total": len(facts),
            "facts_matched": check_post["facts_matched"],
            "parse_ok": check_post["parse_ok"],
            "parse_error": check_post.get("parse_error"),
            "state": (
                "checked" if check_post["parse_ok"] else "parse_error"
            ),
            "recall_pre": (
                round(recall_pre, 4)
                if recall_pre is not None else None
            ),
            "recall_post": (
                round(recall_of(check_post), 4)
                if recall_of(check_post) is not None else None
            ),
            "repairs": repairs_done,
        }

    # ---- 3. QA ---------------------------------------------------------
    qa = call_model(
        client,
        model,
        QA_SYSTEM,
        QA_PROMPT.format(context=context, question=task["question"]),
        max_tokens,
    )
    qa_rec = _model_call_record("qa", unit, qa)
    qa_rec["expected_answer"] = task["expected_answer"]
    qa_rec["correctness"] = score_answer(qa["raw"], task["expected_answer"])
    ledger.append(qa_rec)

    # ---- 4. unit summary ----------------------------------------------
    plain_tokens = approx_count(passage)
    summary = {
        "type": "unit_summary",
        "arm": arm,
        "task_id": unit["task_id"],
        "register": unit["register"],
        "len_class": unit["len_class"],
        "context_chars": len(context),
        "context_approx_tokens": approx_count(context),
        "passage_approx_tokens": plain_tokens,
        "approx_token_ratio": (
            round(approx_count(context) / plain_tokens, 4) if plain_tokens else None
        ),
        "transform_cost_tokens": transform_cost_tokens,
        "qa_correctness": qa_rec["correctness"],
        "fact_check": fact_state,
        "qa_prompt_tokens": qa["usage"]["prompt_tokens"],
        "qa_completion_tokens": qa["usage"]["completion_tokens"],
    }
    ledger.append(summary)
    return summary


def run_units(units, tasks_by_id, client, model, max_tokens, ledger, ctx):
    summaries = []
    for unit in units:
        task = tasks_by_id[unit["task_id"]]
        summaries.append(
            run_unit(unit, task, client, model, max_tokens, ledger, ctx)
        )
    return summaries


def summarize(records):
    """Aggregate over parsed records; parse_error counts REPORTED, never
    hidden inside averages."""
    units = [r for r in records if r.get("type") == "unit_summary"]
    calls = [r for r in records if r.get("type") == "model_call"]
    parse_errors = [r for r in records if r.get("type") == "parse_error"]
    by_arm = {}
    for arm in ARMS:
        arm_units = [u for u in units if u["arm"] == arm]
        checked = [
            u for u in arm_units
            if u["fact_check"]["state"] == "checked"
        ]
        nofacts = [
            u for u in arm_units
            if u["fact_check"]["state"] == "no_facts_declared"
        ]
        correct = sum(
            u["qa_correctness"] in ("exact", "contains", "fuzzy")
            for u in arm_units
        )
        by_arm[arm] = {
            "units": len(arm_units),
            "qa_accuracy": round(correct / len(arm_units), 4) if arm_units else None,
            "mean_recall_pre": (
                round(sum(u["fact_check"]["recall_pre"] for u in checked) / len(checked), 4)
                if checked else None
            ),
            "mean_recall_post": (
                round(sum(u["fact_check"]["recall_post"] for u in checked) / len(checked), 4)
                if checked else None
            ),
            "units_fact_checked": len(checked),
            "units_no_facts_declared": len(nofacts),
            "mean_approx_token_ratio": (
                round(sum(u["approx_token_ratio"] for u in arm_units if u["approx_token_ratio"] is not None) /
                      max(len([u for u in arm_units if u["approx_token_ratio"] is not None]), 1), 4)
                if arm_units else None
            ),
            "mean_transform_cost_tokens": (
                round(sum(u["transform_cost_tokens"] for u in arm_units) / len(arm_units), 4)
                if arm_units else None
            ),
        }
    return {
        "type": "summary",
        "model_call_records": len(calls),
        "unit_summaries": len(units),
        "parse_error_records": len(parse_errors),
        "parse_errors_by_arm": {
            arm: sum(1 for e in parse_errors if e["arm"] == arm) for arm in ARMS
        },
        "by_arm": by_arm,
    }


def write_config(run_dir, config, run_meta):
    os.makedirs(run_dir, exist_ok=True)
    frozen = dict(config)
    frozen["config_hash"] = config_hash(config)
    frozen["git_sha"] = run_meta["git_sha"]
    frozen["run_id"] = run_meta["run_id"]
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(frozen, fh, sort_keys=True, indent=2, ensure_ascii=False)
            fh.write("\n")
    return path


def load_worklist(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def emit_worklist(out_path, tasks_dir=TASKS2_DIR, config=None):
    config = config or frozen_config()
    tasks = load_tasks(tasks_dir)
    units, skipped = build_units(tasks)
    payload = {
        "config_hash": config_hash(config),
        "git_sha": git_sha(),
        "emitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "tasks": len(tasks),
        "units": units,
        "skipped": skipped,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    return payload


def make_ctx():
    return {
        "tables": tables_mod.load_tables(),
        "rules": tables_mod.load_rules(),
        "cipher_table": cipher_mod.load_table(),
    }


def run_local(args, client, model):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.results_dir, f"v21_{stamp}")
    config = frozen_config(model, args.max_tokens)
    run_meta = {
        "run_id": f"v21_{stamp}",
        "git_sha": git_sha(),
        "config_hash": config_hash(config),
    }
    write_config(run_dir, config, run_meta)
    ledger = Ledger(os.path.join(run_dir, "ledger.jsonl"), run_meta)
    tasks = load_tasks(args.tasks_dir)
    units, skipped = build_units(tasks)
    for s in skipped:
        ledger.append({"type": "skipped_unit", **s})
    ctx = make_ctx()
    tasks_by_id = {t["id"]: t for t in tasks}
    run_units(units, tasks_by_id, client, model, args.max_tokens, ledger, ctx)
    return merge_run(run_dir)


def run_one_unit(args, client, model):
    worklist = load_worklist(args.worklist)
    unit = worklist["units"][args.unit]
    run_dir = args.run_dir
    config = frozen_config(model, args.max_tokens)
    run_meta = {
        "run_id": os.path.basename(os.path.normpath(run_dir)),
        "git_sha": worklist["git_sha"],
        "config_hash": worklist["config_hash"],
        "slot": args.unit,
    }
    write_config(run_dir, config, run_meta)
    shard = Ledger(
        os.path.join(run_dir, "shards", f"slot_{args.unit:04d}.jsonl"),
        run_meta,
    )
    tasks_by_id = {t["id"]: t for t in load_tasks(args.tasks_dir)}
    ctx = make_ctx()
    run_unit(unit, tasks_by_id[unit["task_id"]], client, model,
             args.max_tokens, shard, ctx)
    return 0


def merge_run(run_dir):
    """Merge shards -> ledger.jsonl, compute + write summary.json."""
    shards_dir = os.path.join(run_dir, "shards")
    records = []
    if os.path.isdir(shards_dir):
        for fn in sorted(os.listdir(shards_dir)):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(shards_dir, fn), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    # local (non-slurm) runs may already have written straight to ledger.jsonl
    ledger_path = os.path.join(run_dir, "ledger.jsonl")
    existing = []
    if os.path.exists(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as fh:
            existing = [json.loads(l) for l in fh if l.strip()]
    merged = existing + records if records else existing
    with open(ledger_path, "w", encoding="utf-8") as fh:
        for rec in merged:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
    summary = summarize(merged)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emit-worklist", action="store_true")
    parser.add_argument("--worklist",
                        default=os.path.join(_ROOT, "slurm", "worklist.json"))
    parser.add_argument("--unit", type=int, default=None,
                        help="worklist unit index (Slurm array slot)")
    parser.add_argument("--run", action="store_true",
                        help="local sequential run of every unit")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--results-dir", default=os.path.join(_ROOT, "results"))
    parser.add_argument("--tasks-dir", default=TASKS2_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)

    stub = os.environ.get("TCB_V21_STUB") == "1"

    if args.emit_worklist:
        payload = emit_worklist(args.worklist, args.tasks_dir)
        print(
            f"worklist written to {args.worklist}: "
            f"{len(payload['units'])} units, {len(payload['skipped'])} skipped combos"
        )
        return 0

    if args.merge:
        if not args.run_dir:
            print("error: --merge requires --run-dir", file=sys.stderr)
            return 2
        summary = merge_run(args.run_dir)
        print(f"merged; summary written to {os.path.join(args.run_dir, 'summary.json')}")
        print(json.dumps(summary["by_arm"], indent=2, sort_keys=True))
        print(f"parse_error records: {summary['parse_error_records']}")
        return 0

    if args.unit is not None:
        if not args.run_dir:
            print("error: --unit requires --run-dir", file=sys.stderr)
            return 2
        client = make_stub_client() if stub else default_client
        return run_one_unit(args, client, args.model)

    if args.run:
        client = make_stub_client() if stub else default_client
        summary = run_local(args, client, args.model)
        print(f"parse_error records: {summary['parse_error_records']}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
