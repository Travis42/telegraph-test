"""SPEC-VALIDATION §3 tests: retro-regrade tool. Fixture slice ->
expected verdicts, original summary.json untouched (hash compare)."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ab.regrade as rg  # noqa: E402

PASSAGE = (
    "The brig Aurora sailed from Bristol on 4 March 1848 with a cargo "
    "of iron. Her master was Captain Elias Hoy, and she carried a crew "
    "of nineteen."
)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def make_bank(tmp_path):
    td = tmp_path / "bank"
    td.mkdir()
    for i in range(2):
        (td / f"t{i:02d}.json").write_text(json.dumps({
            "id": f"t{i:02d}", "register": "fact", "len_class": "medium",
            "passage": PASSAGE, "question": "Who was the master?",
            "expected_answer": "Captain Elias Hoy", "facts": [],
        }), encoding="utf-8")
    return td


# (q_index, expected, model answer, old raw correct, judge verdict)
FIXTURE = [
    (0, "Captain Elias Hoy", "Her master was Captain Elias Hoy.",
     True, None),
    (1, "Captain Elias Hoy", "no idea whatsoever", False, None),
    # judge overturned the old grader's miss; grader v2 AGREES (the
    # answer is an anchored paraphrase of the expected span)
    (2, "Captain Elias Hoy", "Captain Hoy", False, "grader_wrong"),
    # judge overturned; grader v2 DISAGREES (answer not in expected)
    (3, "nineteen", "the crew was large", False, "grader_wrong"),
    (4, "nineteen", "nineteen men", True, None),
]


def make_run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    answers, judges = [], []
    for qi, expected, raw_answer, old_ok, verdict in FIXTURE:
        answers.append({
            "type": "quiz_answer", "task_id": "t00",
            "register": "fact", "len_class": "medium",
            "q_index": qi, "q_type": "fact",
            "question": f"q{qi}", "expected": expected,
            "raw_answer": raw_answer, "correct": old_ok,
        })
        if verdict:
            judges.append({"type": "judge", "task_id": "t00",
                           "q_index": qi, "sampled_hit": False,
                           "verdict": verdict})
    with open(run / "ledger.jsonl", "w", encoding="utf-8") as fh:
        for rec in answers + judges:
            fh.write(json.dumps(rec) + "\n")
    original = {"run_id": "baseline_fixture", "overall_raw": {"n": 5},
                "overall_corrected": {"n": 5}}
    with open(run / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(original, fh, indent=2)
    return run, [a["q_index"] for a in answers]


def test_regrade_fixture_verdicts_and_untouched_original(tmp_path):
    bank = make_bank(tmp_path)
    run, qis = make_run_dir(tmp_path)
    before_summary = sha(run / "summary.json")
    before_ledger = sha(run / "ledger.jsonl")

    summary = rg.regrade_run(str(run), tasks_dir=str(bank))

    # original files untouched (hash compare)
    assert sha(run / "summary.json") == before_summary
    assert sha(run / "ledger.jsonl") == before_ledger

    # expected grader v2 verdicts, per fixture spec
    regraded = [json.loads(l) for l in
                (run / "regrade_v2" / "regraded_answers.jsonl")
                .read_text().splitlines()]
    by_qi = {r["q_index"]: r for r in regraded}
    expected_verdicts = {0: True, 1: False, 2: True, 3: False, 4: True}
    for qi, verdict in expected_verdicts.items():
        assert by_qi[qi]["correct"] is verdict, qi

    # single headline = grader v2 raw; no corrected keys; judge only
    # under diagnostics/
    assert summary["overall_raw"]["n"] == 5
    assert summary["overall_raw"]["correct"] == 3
    assert summary["overall_raw"]["accuracy"] == 0.6
    for key in ("overall_corrected", "bootstrap_corrected_95",
                "per_type_corrected", "per_register_corrected", "judge",
                "miss_taxonomy"):
        assert key not in summary, key
    assert "judge" in summary["diagnostics"]

    # the 2 judge overturns were adjudicated by code, not the verdict
    assert summary["regrade"][
        "judge_overturns_adjudicated_by_code"] == 2
    assert summary["regrade"]["judge_overturns_upheld_by_grader_v2"] == 1
    diag = json.loads((run / "regrade_v2" / "diagnostics.json").read_text())
    upheld = {o["q_index"] for o in diag["judge_overturn_adjudications"]
              if o["grader_v2_correct"]}
    assert upheld == {2}

    # re-running the regrade is idempotent and still never touches
    # the original summary
    rg.regrade_run(str(run), tasks_dir=str(bank))
    assert sha(run / "summary.json") == before_summary


def test_regrade_resolves_passages_from_run_dir(tmp_path):
    bank = make_bank(tmp_path)
    run, _ = make_run_dir(tmp_path)
    with open(run / "passages.json", "w", encoding="utf-8") as fh:
        json.dump({"t00": PASSAGE}, fh)
    summary = rg.regrade_run(str(run), tasks_dir=str(bank))
    assert summary["overall_raw"]["n"] == 5


def test_regrade_missing_material_errors(tmp_path):
    bank = make_bank(tmp_path)
    # no run dir
    with pytest.raises(rg.RegradeError, match="not found"):
        rg.regrade_run(str(tmp_path / "void"), tasks_dir=str(bank))
    # run dir without a ledger
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(rg.RegradeError, match="no ledger"):
        rg.regrade_run(str(empty), tasks_dir=str(bank))
    # ledger whose task ids cannot be resolved to passages
    run = tmp_path / "orphan"
    run.mkdir()
    with open(run / "ledger.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "quiz_answer", "task_id": "zz",
                             "register": "fact", "len_class": "m",
                             "q_index": 0, "q_type": "fact",
                             "question": "q", "expected": "e",
                             "raw_answer": "a", "correct": True}) + "\n")
    with pytest.raises(rg.RegradeError, match="cannot resolve"):
        rg.regrade_run(str(run), tasks_dir=str(bank))


def test_regrade_cli(tmp_path):
    bank = make_bank(tmp_path)
    run, _ = make_run_dir(tmp_path)
    before = sha(run / "summary.json")
    out = subprocess.run(
        [sys.executable, "-m", "ab.regrade", "--run-dir", str(run),
         "--tasks-dir", str(bank)],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert (run / "regrade_v2" / "summary.json").exists()
    assert sha(run / "summary.json") == before
    # missing dir -> exit 2 with a clear error
    bad = subprocess.run(
        [sys.executable, "-m", "ab.regrade", "--run-dir",
         str(tmp_path / "void")],
        capture_output=True, text=True, cwd=ROOT)
    assert bad.returncode == 2
    assert "not found" in bad.stderr


# --- cablese cond_answer path (cbl2 v2 regrade) ------------------------------

COND_FIXTURE = {
    # condition -> list of (q_index, expected, raw answer, v2 correct)
    "A-FROM-PLAIN": [(0, "Captain Elias Hoy", "Captain Elias Hoy", True),
                     (1, "nineteen", "no idea whatsoever", False)],
    "A-FROM-CABLESE": [(0, "Captain Elias Hoy",
                        "mstr captain elias hoy", True),
                       (1, "nineteen", "crw nineteen", True)],
    "A-PLAIN-DIRECT": [(0, "Captain Elias Hoy", "Captain Hoy", True),
                       (1, "nineteen", "nineteen men", True)],
    "A-CABLESE-DIRECT": [(0, "Captain Elias Hoy",
                          "captain elias hoy telegram", True)],
    "DECODE": [(0, "Captain Elias Hoy", "somebody else entirely", False),
               (1, "nineteen", "a crew of nineteen", True)],
}


def make_cond_run(tmp_path, extra_quiz=False, orphan=False):
    run = tmp_path / "cond_run"
    run.mkdir()
    recs = []
    for cond, items in COND_FIXTURE.items():
        for qi, expected, raw, ok in items:
            recs.append({
                "type": "cond_answer", "condition": cond,
                "task_id": "t00", "register": "fact",
                "len_class": "medium", "q_index": qi, "q_type": "fact",
                "question": f"q{qi}", "expected": expected,
                "raw_answer": raw, "correct": not ok,  # stale verdicts
            })
    if orphan:
        recs.append({
            "type": "cond_answer", "condition": "DECODE",
            "task_id": "zz", "register": "fact", "len_class": "medium",
            "q_index": 0, "q_type": "fact", "question": "q",
            "expected": "e", "raw_answer": "a", "correct": True,
        })
    if extra_quiz:
        recs.append({
            "type": "quiz_answer", "task_id": "t00", "register": "fact",
            "len_class": "medium", "q_index": 0, "q_type": "fact",
            "question": "q0", "expected": "Captain Elias Hoy",
            "raw_answer": "Captain Elias Hoy", "correct": True,
        })
    # record axis: plain records are 100 tokens, cablese 40 -> 60% savings
    recs += [
        {"type": "record", "condition": "R-PLAIN", "task_id": "t00",
         "approx_tokens": 100, "text": "x" * 100},
        {"type": "record", "condition": "R-CABLESE", "task_id": "t00",
         "approx_tokens": 40, "text": "x" * 40},
    ]
    with open(run / "ledger.jsonl", "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
    with open(run / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"run_id": "cbl2_fixture"}, fh, indent=2)
    return run


def test_cond_regrade_parsing_and_per_condition_v2(tmp_path):
    bank = make_bank(tmp_path)
    run = make_cond_run(tmp_path)
    before = sha(run / "summary.json")
    report = rg.regrade_run(str(run), tasks_dir=str(bank))
    assert report["answers_regraded"] == sum(
        len(v) for v in COND_FIXTURE.values())
    # hand-computed per-condition v2 accuracies
    for cond, items in COND_FIXTURE.items():
        want = sum(1 for *_, ok in items if ok) / len(items)
        assert report["conditions"][cond]["n"] == len(items), cond
        assert report["conditions"][cond]["v2_accuracy"] == want, cond
        assert report["conditions"][cond]["wilson_ci"][0] <= want \
            <= report["conditions"][cond]["wilson_ci"][1]
    assert report["anchoring_rejections"] == 0
    # original summary untouched (hash compare)
    assert sha(run / "summary.json") == before
    # per-record detail file carries the v2 verdicts
    detail = [json.loads(l) for l in
              (run / "regrade_v2" / "regraded_cond_answers.jsonl")
              .read_text().splitlines()]
    assert len(detail) == report["answers_regraded"]
    assert all("grade_v2" in r for r in detail)


def test_cond_regrade_ratios_and_mcnemar(tmp_path):
    bank = make_bank(tmp_path)
    run = make_cond_run(tmp_path)
    report = rg.regrade_run(str(run), tasks_dir=str(bank))
    # recovery = 1.0/0.5, decode = 0.5/1.0
    assert report["recovery_ratio_v2"] == 2.0
    assert report["decode_ratio_v2"] == 0.5
    m1 = report["mcnemar_v2"]["A-FROM-CABLESE_vs_A-FROM-PLAIN"]
    assert (m1["n"], m1["a"], m1["b"], m1["c"], m1["d"]) == (2, 1, 1, 0, 0)
    m2 = report["mcnemar_v2"]["DECODE_vs_A-PLAIN-DIRECT"]
    assert (m2["n"], m2["a"], m2["b"], m2["c"], m2["d"]) == (2, 1, 0, 1, 0)
    # exact two-sided binomial on 1 discordant pair -> p = 1.0
    assert m1["p_value"] == 1.0 and m2["p_value"] == 1.0


def test_cond_regrade_record_token_accounting(tmp_path):
    bank = make_bank(tmp_path)
    run = make_cond_run(tmp_path)
    report = rg.regrade_run(str(run), tasks_dir=str(bank))
    tok = report["record_tokens_v2"]
    assert tok["R-PLAIN"]["mean_approx_tokens"] == 100.0
    assert tok["R-CABLESE"]["mean_approx_tokens"] == 40.0
    assert tok["savings_pct_v2"] in (60.0, 60.00000000000001)


def test_cond_regrade_anchoring_rejection_path(tmp_path):
    bank = make_bank(tmp_path)
    run = make_cond_run(tmp_path)
    with open(run / "ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "type": "cond_answer", "condition": "DECODE",
            "task_id": "t00", "register": "fact", "len_class": "medium",
            "q_index": 9, "q_type": "fact", "question": "q9",
            "expected": "purple magic unicorn dust", "raw_answer":
            "purple magic unicorn dust", "correct": True,
        }) + "\n")
    report = rg.regrade_run(str(run), tasks_dir=str(bank))
    # unanchored expected answer -> item invalid, never correct
    assert report["anchoring_rejections"] == 1
    cond = report["conditions"]["DECODE"]
    assert cond["n"] == 3
    assert cond["correct"] == 1  # rejection is a forced miss


def test_cond_regrade_missing_passage_guard(tmp_path):
    bank = make_bank(tmp_path)
    run = make_cond_run(tmp_path, orphan=True)
    with pytest.raises(rg.RegradeError, match="cannot resolve"):
        rg.regrade_run(str(run), tasks_dir=str(bank))


def test_cond_regrade_hash_protection_of_original_summary(tmp_path):
    bank = make_bank(tmp_path)
    run = make_cond_run(tmp_path)
    before = sha(run / "summary.json")
    report = rg.regrade_run(str(run), tasks_dir=str(bank))
    assert report["original_summary_sha256"] == before
    rg.regrade_run(str(run), tasks_dir=str(bank))  # idempotent
    assert sha(run / "summary.json") == before


def test_cond_regrade_mixed_run_both_paths(tmp_path):
    bank = make_bank(tmp_path)
    run = make_cond_run(tmp_path, extra_quiz=True)
    result = rg.regrade_run(str(run), tasks_dir=str(bank))
    assert set(result) == {"quiz_answer", "cond_answer"}
    assert result["quiz_answer"]["overall_raw"]["n"] == 1
    assert result["cond_answer"]["recovery_ratio_v2"] == 2.0
    assert (run / "regrade_v2" / "summary.json").exists()
    assert (run / "regrade_v2.json").exists()


def test_cond_regrade_cli_prints_report_path(tmp_path):
    bank = make_bank(tmp_path)
    run = make_cond_run(tmp_path)
    before = sha(run / "summary.json")
    out = subprocess.run(
        [sys.executable, "-m", "ab.regrade", "--run-dir", str(run),
         "--tasks-dir", str(bank)],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert str(run / "regrade_v2.json") in out.stdout
    assert (run / "regrade_v2.json").exists()
    assert sha(run / "summary.json") == before


def test_regrade_no_answer_records_errors(tmp_path):
    bank = make_bank(tmp_path)
    run = tmp_path / "junk"
    run.mkdir()
    with open(run / "ledger.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "model_call", "call": "x"}) + "\n")
    with pytest.raises(rg.RegradeError,
                       match="no quiz_answer or cond_answer"):
        rg.regrade_run(str(run), tasks_dir=str(bank))
