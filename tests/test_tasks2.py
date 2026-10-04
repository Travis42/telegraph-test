"""Task bank v2: schema validation of every shipped task + the
generation-verification gate with a stubbed model."""

import json
import os

import pytest

from ab.tasks2 import generate

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS2 = os.path.join(os.path.dirname(HERE), "ab", "tasks2")

ALLOWED_REGISTERS = {"prose", "fact", "formulaic"}
ALLOWED_LEN = {"short", "medium", "long"}


def load_all():
    tasks = []
    for fn in sorted(os.listdir(TASKS2)):
        if fn.endswith(".json"):
            with open(os.path.join(TASKS2, fn), encoding="utf-8") as fh:
                tasks.append(json.load(fh))
    return tasks


def test_shipped_bank_schema():
    tasks = load_all()
    assert len(tasks) == 50
    for t in tasks:
        assert set(t) == {
            "id", "register", "len_class", "passage",
            "question", "expected_answer", "facts",
        }, t.get("id")
        assert t["register"] in ALLOWED_REGISTERS
        assert t["len_class"] in ALLOWED_LEN
        assert isinstance(t["passage"], str) and t["passage"]
        assert isinstance(t["question"], str) and t["question"]
        assert isinstance(t["expected_answer"], str) and t["expected_answer"]
        assert isinstance(t["facts"], list)
        assert all(isinstance(f, str) and f for f in t["facts"])


def test_bank_class_counts_and_lengths():
    tasks = load_all()
    prose = [t for t in tasks if t["register"] == "prose"]
    fact = [t for t in tasks if t["register"] == "fact"]
    formulaic = [t for t in tasks if t["register"] == "formulaic"]
    assert len(prose) == 24
    assert all(t["len_class"] == "short" for t in prose)
    med = [t for t in fact if t["len_class"] == "medium"]
    lng = [t for t in fact if t["len_class"] == "long"]
    assert len(med) >= 6 and len(lng) >= 6
    assert all(600 <= len(t["passage"]) <= 1000 for t in med)
    assert all(1400 <= len(t["passage"]) <= 2100 for t in lng)
    assert all(t["facts"] for t in fact)
    assert 12 <= len(formulaic) <= 16
    assert all(t["facts"] for t in formulaic)


def test_formulaic_recurring_schema():
    formulaic = [t for t in load_all() if t["register"] == "formulaic"]
    prefixes = [t["passage"].split()[0] for t in formulaic]
    assert set(prefixes) == {"STATUS", "STANDUP", "CHECKLIST", "CONFIG"}
    # recurrence: at least two passages share each schema family
    for prefix in prefixes:
        assert prefixes.count(prefix) >= 2


# ------- generation-verification gate (stubbed model) -------

def stub_client_generation(payload):
    user = payload["messages"][1]["content"]
    if "fact list" in user:  # generation call
        content = json.dumps({
            "passage": "The brig Foam sailed from Liverpool on 4 March with 900 tons of coal.",
            "facts": [
                "The brig Foam sailed from Liverpool on 4 March.",
                "She carried 900 tons of coal.",
            ],
        })
    else:  # verification call -> claim all facts stated
        content = "[1, 2]"
    return {
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20,
                  "total_tokens": 70},
    }


def run_gate(client, tmp_path):
    gen_log = str(tmp_path / "gen.jsonl")
    candidate = generate.generate_candidate(
        client, "m", "medium", "seed-x", 512, gen_log
    )
    if candidate is None:
        return None, gen_log
    ratio, check = generate.verify_candidate(
        client, "m", candidate, "medium", "seed-x", 512, gen_log
    )
    if ratio is None:
        return None, gen_log
    return (candidate, ratio), gen_log


def test_generation_gate_accepts_verified_candidate(tmp_path):
    result, gen_log = run_gate(stub_client_generation, tmp_path)
    assert result is not None
    candidate, ratio = result
    assert ratio == 1.0
    events = [json.loads(l) for l in open(gen_log)]
    kinds = [e["type"] for e in events]
    assert "candidate_accepted" in kinds
    # raw model answers are logged verbatim before parsing
    assert any(e["type"] == "generation_call" and "Foam" in e["raw"]
               for e in events)


def stub_client_flaky(payload):
    user = payload["messages"][1]["content"]
    if "fact list" in user:
        content = json.dumps({"passage": "p", "facts": ["f1", "f2"]})
    else:
        content = "I think maybe fact one, probably?"  # unparseable
    return {
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                  "total_tokens": 15},
    }


def test_generation_gate_rejects_on_parse_failure_with_logged_reason(tmp_path):
    result, gen_log = run_gate(stub_client_flaky, tmp_path)
    assert result is None
    events = [json.loads(l) for l in open(gen_log)]
    rejected = [e for e in events if e["type"] == "candidate_rejected"]
    assert rejected and rejected[-1]["reason"].startswith("verification parse")
    # the failing raw answer is persisted verbatim
    assert any(
        e["type"] == "verification_call" and "probably" in e["raw"]
        for e in events
    )


def test_generation_gate_rejects_below_threshold(tmp_path):
    def client(payload):
        user = payload["messages"][1]["content"]
        if "fact list" in user:
            content = json.dumps({
                "passage": "p", "facts": ["f1", "f2", "f3", "f4", "f5"],
            })
        else:
            content = "[1]"  # only 20% verify < 60% threshold
        return {
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15},
        }

    result, gen_log = run_gate(client, tmp_path)
    assert result is None
    events = [json.loads(l) for l in open(gen_log)]
    rejected = [e for e in events if e["type"] == "candidate_rejected"]
    assert "verify ratio 0.20 < 0.6" in rejected[-1]["reason"]
