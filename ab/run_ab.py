#!/usr/bin/env python3
"""A/B runner: each task run in PLAIN vs CODED context, same model,
temperature 0, fixed max_tokens. Two prompt kinds per condition (QA +
reconstruction). Token accounting is usage-field ground truth from the API
response; results are persisted as JSONL with a printed summary.

Requires ZAI_API_KEY in the environment; aborts cleanly if missing.
Unit tests stub the API boundary by injecting a fake client callable.
"""

import argparse
import datetime
import difflib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tcb import tables as tables_mod  # noqa: E402
from tcb.encode import encode_router  # noqa: E402

API_URL = os.environ.get(
    "TCB_ZAI_API_URL", "https://api.z.ai/api/paas/v4/chat/completions"
)
DEFAULT_MODEL = os.environ.get("TCB_MODEL", "zai/glm-5.3-flash")
DEFAULT_MAX_TOKENS = 256
TEMPERATURE = 0

QA_PROMPT = (
    "Answer the question using only the passage. Answer briefly.\n\n"
    "Passage: {context}\n\nQuestion: {question}\nAnswer:"
)
RECONSTRUCTION_PROMPT = (
    "Reconstruct the original passage in full English.\n\n{context}"
)


def default_client(payload):
    """Minimal stdlib HTTP client for the ZAI (OpenAI-compatible) endpoint."""
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        sys.exit(
            "error: ZAI_API_KEY is not set in the environment; "
            "export ZAI_API_KEY=<your key> and re-run."
        )
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        sys.exit(f"error: API returned HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}")
    except urllib.error.URLError as exc:
        sys.exit(f"error: API unreachable: {exc}")


def call_model(client, model, system, user, max_tokens):
    payload = {
        # accept "provider/model" aliases; the endpoint wants the bare model id
        "model": model.split("/", 1)[-1],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    started = time.monotonic()
    response = client(payload)
    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
    usage = response["usage"]  # faithful usage-field shape
    answer = response["choices"][0]["message"]["content"]
    return {
        "answer": answer,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_ms": latency_ms,
    }


def load_tasks(tasks_dir):
    tasks = []
    for fn in sorted(os.listdir(tasks_dir)):
        if fn.startswith("task_") and fn.endswith(".json"):
            with open(os.path.join(tasks_dir, fn), "r", encoding="utf-8") as fh:
                tasks.append(json.load(fh))
    return tasks


def coded_context(passage, tables=None, rules=None):
    return encode_router(passage, tables, rules)["coded"]


def _normalize(text):
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score_answer(answer, expected):
    a, e = _normalize(answer), _normalize(expected)
    if a == e:
        return "exact"
    if e in a:
        return "contains"
    if difflib.SequenceMatcher(None, a, e).ratio() >= 0.8:
        return "fuzzy"
    return "wrong"


CORRECT_KINDS = {"exact", "contains", "fuzzy"}


def run(tasks, client, model, max_tokens, out_path):
    tables = tables_mod.load_tables()
    rules = tables_mod.load_rules()
    records = []
    for task in tasks:
        contexts = {
            "plain": task["passage"],
            "coded": coded_context(task["passage"], tables, rules),
        }
        for condition, context in sorted(contexts.items()):
            prompts = {
                "qa": QA_PROMPT.format(context=context, question=task["question"]),
                "reconstruction": RECONSTRUCTION_PROMPT.format(context=context),
            }
            for kind, prompt in sorted(prompts.items()):
                result = call_model(
                    client,
                    model,
                    "You answer reading-comprehension tasks faithfully and briefly.",
                    prompt,
                    max_tokens,
                )
                rec = {
                    "type": "run",
                    "task_id": task["id"],
                    "domain": task["domain"],
                    "condition": condition,
                    "prompt_kind": kind,
                    "model": model,
                    "temperature": TEMPERATURE,
                }
                rec.update(result)
                if kind == "qa":
                    rec["expected_answer"] = task["expected_answer"]
                    rec["correctness"] = score_answer(
                        result["answer"], task["expected_answer"]
                    )
                records.append(rec)
                with open(out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return records


def summarize(records):
    def med(values):
        return statistics.median(values) if values else None

    def by(kind, condition):
        return [r for r in records if r["prompt_kind"] == kind and r["condition"] == condition]

    pairs_in = [
        (p["prompt_tokens"], c["prompt_tokens"])
        for p, c in zip(by("qa", "plain"), by("qa", "coded"))
        if p["task_id"] == c["task_id"]
    ]
    pairs_out = [
        (p["completion_tokens"], c["completion_tokens"])
        for p, c in zip(by("qa", "plain"), by("qa", "coded"))
        if p["task_id"] == c["task_id"]
    ]
    qa_plain_correct = sum(r["correctness"] in CORRECT_KINDS for r in by("qa", "plain"))
    qa_coded_correct = sum(r["correctness"] in CORRECT_KINDS for r in by("qa", "coded"))
    n_qa = len(by("qa", "plain"))
    wins = sum(1 for p, c in pairs_in if c < p)
    losses = sum(1 for p, c in pairs_in if c > p)
    summary = {
        "type": "summary",
        "runs": len(records),
        "qa_tasks": n_qa,
        "median_input_token_delta": med([c - p for p, c in pairs_in]) if pairs_in else None,
        "median_output_token_delta": med([c - p for p, c in pairs_out]) if pairs_out else None,
        "plain_qa_correct": qa_plain_correct,
        "coded_qa_correct": qa_coded_correct,
        "correctness_delta": qa_coded_correct - qa_plain_correct if n_qa else None,
        "input_token_wins": wins,
        "input_token_losses": losses,
    }
    return summary


def print_summary(summary):
    print("=== A/B summary ===")
    print(f"runs: {summary['runs']}  (qa tasks per condition: {summary['qa_tasks']})")
    print(f"median input-token delta (coded - plain):  {summary['median_input_token_delta']}")
    print(f"median output-token delta (coded - plain): {summary['median_output_token_delta']}")
    print(f"QA correct  plain: {summary['plain_qa_correct']}  coded: {summary['coded_qa_correct']}"
          f"  delta: {summary['correctness_delta']}")
    print(f"input-token wins/losses: {summary['input_token_wins']}/{summary['input_token_losses']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="A/B runner (plain vs coded context)")
    parser.add_argument("--tasks", default=os.path.join(_HERE, "tasks"))
    parser.add_argument("--results-dir", default=os.path.join(_HERE, "results"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)

    if not os.environ.get("ZAI_API_KEY"):
        print(
            "error: ZAI_API_KEY is not set in the environment; "
            "export ZAI_API_KEY=<your key> and re-run.",
            file=sys.stderr,
        )
        return 2
    tasks = load_tasks(args.tasks)
    if not tasks:
        print("error: no tasks found", file=sys.stderr)
        return 2
    os.makedirs(args.results_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"ab_{stamp}.jsonl")
    records = run(tasks, default_client, args.model, args.max_tokens, out_path)
    summary = summarize(records)
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, sort_keys=True) + "\n")
    print(f"results written to {out_path}")
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
