"""SPEC-VALIDATION tests: grader v2 (passage-anchored deterministic
grading) and the authoring-time validation gates. Offline only — no
network, no model calls."""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ab.baseline as bl  # noqa: E402
import ab.grade as gr  # noqa: E402
from ab.expAB import Ledger  # noqa: E402

PASSAGE = (
    "The brig Aurora sailed from Bristol on 4 March 1848 with a cargo "
    "of iron. Her master was Captain Elias Hoy, and she carried a crew "
    "of nineteen. After a storm off Ushant she put into Falmouth, and "
    "later resumed her voyage to Lisbon, arriving on 2 May before "
    "continuing to Oporto, which lay north of Lisbon along the coast. "
    "The vessel displaced 1,000 tons."
)


def make_tasks_dir(tmp_path, n=2, expected="Captain Elias Hoy"):
    td = tmp_path / "tasks2"
    td.mkdir(exist_ok=True)
    for i in range(n):
        task = {
            "id": f"t{i:02d}",
            "register": ["fact", "prose"][i % 2],
            "len_class": "medium",
            "passage": PASSAGE,
            "question": "Who was the master?",
            "expected_answer": expected,
            "facts": ["The brig Aurora sailed from Bristol."],
        }
        (td / f"t{i:02d}.json").write_text(json.dumps(task),
                                           encoding="utf-8")
    return td


def run_stub(tasks_dir, results_dir, **kw):
    return bl.run_baseline(bl.make_stub_client(), "stub-model", 512,
                           str(results_dir), tasks_dir=str(tasks_dir),
                           **kw)


# ---------------- grader v2: edge cases ----------------

def test_anchored_exact_answer_correct():
    g = gr.grade_v2(PASSAGE, "Who was the master?",
                    "Captain Elias Hoy", "Captain Elias Hoy")
    assert g["anchored"] is True and g["correct"] is True
    assert g["diagnostics"]["anchor_score"] == 1.0
    assert g["diagnostics"]["match_score"] == 1.0


def test_unanchored_expected_never_correct():
    # expected answer is NOT extractable from the passage -> invalid
    # item: even a verbatim model answer is graded not-correct
    g = gr.grade_v2(PASSAGE, "What color was the unicorn?",
                    "purple elephant", "A purple elephant!")
    assert g["anchored"] is False
    assert g["correct"] is False
    assert g["diagnostics"]["anchor_score"] == 0.0


def test_paraphrase_in_passage_is_anchored():
    # every content word of the expected answer appears in the passage
    # (word-order paraphrase) -> anchored
    g = gr.grade_v2(PASSAGE, "Who led the ship?",
                    "master of the brig", "the brig's master")
    assert g["anchored"] is True


def test_number_format_canonicalization():
    g = gr.grade_v2(PASSAGE, "How many tons?",
                    "1,000 tons", "1000 tons")
    assert g["correct"] is True  # 1,000 == 1000 after normalization
    g2 = gr.grade_v2(PASSAGE, "When?", "4 March 1848",
                     "March 4th, 1848")
    assert g2["correct"] is True  # ordinal + order tolerated by tokens


def test_wrong_answer_not_correct():
    g = gr.grade_v2(PASSAGE, "Who was the master?",
                    "Captain Elias Hoy", "the cargo was iron")
    assert g["anchored"] is True and g["correct"] is False


def test_empty_inputs_are_unanchored_and_wrong():
    g = gr.grade_v2("", "q", "", "")
    assert g["anchored"] is False and g["correct"] is False


def test_determinism_same_inputs_same_verdict():
    args = (PASSAGE, "Who was the master?", "Captain Elias Hoy",
            "Her master was Captain Elias Hoy")
    a, b = gr.grade_v2(*args), gr.grade_v2(*args)
    assert a == b
    # repeated normalization is also stable
    assert gr.normalize("1,000 Tons, 3rd.") == gr.normalize(
        "1000 tons 3rd")


def test_answer_superset_of_expected_is_correct():
    g = gr.grade_v2(PASSAGE, "Who was the master?",
                    "Captain Elias Hoy",
                    "The master was Captain Elias Hoy of the Aurora.")
    assert g["correct"] is True


# ---------------- gate: bank items ----------------

def test_bank_unanchored_item_excluded_and_logged(tmp_path):
    td = tmp_path / "bank"
    td.mkdir()
    specs = [("t00", "Captain Elias Hoy"),      # anchored
             ("t01", "a purple elephant")]      # unanchored
    for tid, expected in specs:
        (td / f"{tid}.json").write_text(json.dumps({
            "id": tid, "register": "fact", "len_class": "medium",
            "passage": PASSAGE, "question": "q?",
            "expected_answer": expected, "facts": [],
        }), encoding="utf-8")
    summary = run_stub(td, tmp_path / "r")
    assert summary["passages"] == 1  # only t00 ran
    assert summary["bank_validation"] == {
        "selected": 2, "anchored": 1, "excluded_unanchored": 1}
    log = json.loads((tmp_path / "r" / "unanchored_bank_items.json")
                     .read_text())
    assert [e["task_id"] for e in log] == ["t01"]
    assert log[0]["expected_answer"] == "a purple elephant"
    asked = {json.loads(l)["task_id"] for l in
             (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()
             if json.loads(l).get("type") == "quiz_answer"}
    assert asked == {"t00"}


def test_all_bank_unanchored_is_hard_error(tmp_path):
    td = make_tasks_dir(tmp_path, 2, expected="nowhere to be found")
    with pytest.raises(bl.BaselineError):
        run_stub(td, tmp_path / "r")


# ---------------- gate: generated items ----------------

def _gen_client(items_by_half):
    """Stub client whose generation returns the given item lists
    (half1 first call, half2 second call per round)."""
    calls = {"i": 0}

    def client(payload):
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        if "quiz questions" in system:
            items = items_by_half["h1" if "fact-recall" in user
                                   else "h2"]
            answer = json.dumps(items)
        elif "judge" in system.casefold():
            answer = "wrong"
        elif "Answer:" in user:
            answer = "stub-answer"
        else:
            answer = "stub"
        return {"choices": [{"index": 0, "message": {
            "role": "assistant", "content": answer},
            "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                      "total_tokens": 14}}
    return client


def test_generated_unanchored_rejected_and_logged(tmp_path):
    td = make_tasks_dir(tmp_path, 1)

    def item(i, q_type, answer):
        return {"q": f"question {i} alpha beta gamma delta {q_type}",
                "a": answer, "type": q_type}

    h1 = [item(i, "fact", "Captain Elias Hoy") for i in range(12)]
    h1.append(item(99, "fact", "a purple elephant"))  # unanchored
    h2 = ([item(20 + i, "numeric", "nineteen") for i in range(4)]
          + [item(24 + i, "ordering", "Falmouth") for i in range(4)]
          + [item(28 + i, "inference", "Oporto") for i in range(4)])
    # rename so h1 extra lands in the fact half list
    h1 = h1[:13]

    real = bl.generate_stratified_quiz

    def gen(client, model, task, max_tokens, ledger):
        kept, dropped, unanchored = [], [], []
        for it in (h1 + h2):
            if gr.anchored(task["passage"], it["a"]):
                kept.append({"q": it["q"], "a": it["a"],
                             "q_type": it["type"]})
            else:
                unanchored.append({"q": it["q"], "a": it["a"],
                                   "q_type": it["type"]})
        return kept, dropped, unanchored

    bl.generate_stratified_quiz = gen
    try:
        summary = run_stub(td, tmp_path / "r")
    finally:
        bl.generate_stratified_quiz = real
    assert summary["generated_validation"]["rejected_unanchored"] == 1
    log = json.loads(
        (tmp_path / "r" / "unanchored_generated_items.json").read_text())
    assert log[0]["answer"] == "a purple elephant"
    assert log[0]["task_id"] == "t00"
    # the unanchored item was never asked
    asked = [json.loads(l) for l in
             (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    questions = {r["question"] for r in asked
                 if r.get("type") == "quiz_item"}
    assert all("99" not in q or "purple" not in q for q in questions)
    rejects = [r for r in asked if r.get("type") == "anchor_reject"]
    assert len(rejects) == 1 and rejects[0]["task_id"] == "t00"


def test_quota_impossible_is_hard_error(tmp_path):
    td = make_tasks_dir(tmp_path, 1)
    # only 6 anchored fact items per round -> fact quota 12 impossible
    h1 = [{"q": f"fact question w{i}a w{i}b w{i}c w{i}d",
           "a": "Captain Elias Hoy", "type": "fact"} for i in range(6)]
    h2 = [{"q": f"numeric question w{20 + i}a w{20 + i}b w{20 + i}c w{20 + i}d",
           "a": "nineteen", "type": "numeric"} for i in range(4)]
    client = _gen_client({"h1": h1, "h2": h2})
    ledger = Ledger(str(tmp_path / "led.jsonl"),
                    {"run_id": "x", "git_sha": "t"})
    task = json.loads((td / "t00.json").read_text())
    with pytest.raises(bl.BaselineError, match="quota impossible"):
        bl.generate_stratified_quiz(client, "stub-model", task, 512,
                                    ledger)
    # one retry round happened before the hard error (4 gen calls)
    calls = [json.loads(l) for l in
             (tmp_path / "led.jsonl").read_text().splitlines()]
    assert sum(1 for c in calls
               if c.get("type") == "model_call"
               and c["call"].startswith("gen_")) == 4


def test_quota_filled_from_anchored_pool(tmp_path):
    td = make_tasks_dir(tmp_path, 1)
    # 11 anchored fact + 1 unanchored fact would miss the quota...
    # but 12 anchored pass; the gate must accept a full anchored pool
    h1 = [{"q": f"fact question w{i}a w{i}b w{i}c w{i}d",
           "a": "Captain Elias Hoy", "type": "fact"} for i in range(12)]
    h2 = ([{"q": f"numeric question w{20 + i}a w{20 + i}b w{20 + i}c w{20 + i}d",
            "a": "nineteen", "type": "numeric"} for i in range(4)]
          + [{"q": f"ordering question w{24 + i}a w{24 + i}b w{24 + i}c w{24 + i}d",
              "a": "Falmouth", "type": "ordering"} for i in range(4)]
          + [{"q": f"inference question w{28 + i}a w{28 + i}b w{28 + i}c w{28 + i}d",
              "a": "Oporto", "type": "inference"} for i in range(4)])
    client = _gen_client({"h1": h1, "h2": h2})
    ledger = Ledger(str(tmp_path / "led.jsonl"),
                    {"run_id": "x", "git_sha": "t"})
    task = json.loads((td / "t00.json").read_text())
    kept, dropped, unanchored = bl.generate_stratified_quiz(
        client, "stub-model", task, 512, ledger)
    assert len(kept) == 24 and unanchored == []
    counts = {}
    for it in kept:
        counts[it["q_type"]] = counts.get(it["q_type"], 0) + 1
    assert counts == {"fact": 12, "numeric": 4, "ordering": 4,
                      "inference": 4}


# ---------------- grep-able invariant ----------------

def test_grade_module_has_no_model_calls():
    src = open(os.path.join(ROOT, "ab", "grade.py"),
               encoding="utf-8").read()
    for banned in ("call_model", "requests", "urllib", "http",
                   "default_client", "openai"):
        assert banned not in src
