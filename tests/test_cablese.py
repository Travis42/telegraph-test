"""SPEC-CABLESE tests (offline; stub client everywhere, no network).
Covers frozen condition prompts (cablese rules, whole-content
instruction), decode prompt assembly, pairing + exact McNemar math,
Wilson sanity, record-token accounting, slicing + merge
serial-equivalence, CLI dispatch regression (expB incident 14475:
dispatch dropping slice args), summary shape, and the slurm bundle."""

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ab.cablese as cs  # noqa: E402
from tcb.tokencount import approx_count  # noqa: E402

TASK_PASSAGE = (
    "The brig Aurora sailed from Bristol on 4 March 1848 with a cargo "
    "of iron. Her master was Captain Elias Hoy, and she carried a crew "
    "of nineteen. After a storm off Ushant she put into Falmouth, and "
    "later resumed her voyage to Lisbon, arriving on 2 May before "
    "continuing to Oporto, which lay north of Lisbon along the coast."
)


def make_tasks_dir(tmp_path, n=6):
    td = tmp_path / "tasks2"
    td.mkdir(exist_ok=True)
    registers = ["fact", "prose", "formulaic"]
    for i in range(n):
        task = {
            "id": f"t{i:02d}",
            "register": registers[i % 3],
            "len_class": "medium",
            "passage": f"{TASK_PASSAGE} Passage variant {i}.",
            "question": "Who was the master?",
            "expected_answer": "Captain Elias Hoy",
            "facts": ["The brig Aurora sailed from Bristol."],
        }
        (td / f"t{i:02d}.json").write_text(json.dumps(task),
                                           encoding="utf-8")
    return td


def run_stub(tasks_dir, results_dir, **kw):
    return cs.run_cablese(cs.make_stub_client(), "stub-model", 512,
                          str(results_dir), tasks_dir=str(tasks_dir), **kw)


def cond_answer(cond, task_id, q_index, q_type, register, correct):
    return {"type": "cond_answer", "condition": cond,
            "task_id": task_id, "register": register,
            "len_class": "medium", "q_index": q_index, "q_type": q_type,
            "question": f"q{q_index}", "expected": "e",
            "raw_answer": "a", "correct_kind":
            "exact" if correct else "wrong", "correct": correct}


def judge_rec(cond, task_id, q_index, verdict, sampled_hit=False):
    return {"type": "judge", "condition": cond, "task_id": task_id,
            "q_index": q_index, "sampled_hit": sampled_hit,
            "verdict": verdict, "raw": verdict}


def record_rec(cond, task_id, tokens):
    return {"type": "record", "condition": cond, "task_id": task_id,
            "register": "fact", "len_class": "medium",
            "approx_tokens": tokens, "text": "x" * 4}


# ---------------- frozen prompts ----------------

def test_cablese_rules_present_in_both_cablese_prompts():
    for prompt in (cs.RECORD_CABLESE_SYSTEM, cs.A_CABLESE_DIRECT_SYSTEM):
        assert "telegraphese" in prompt
        assert "drop articles" in prompt
        assert "abbreviate" in prompt
        assert "telegram style" in prompt
        assert "ALL facts retained" in prompt
        assert "numbers and proper nouns verbatim" in prompt


def test_record_prompts_demand_whole_content():
    for user in (cs.RECORD_PLAIN_USER, cs.RECORD_CABLESE_USER):
        assert "whole content" in user.casefold()
        assert "nothing de-selected" in user.casefold()
    assert "normal prose" in cs.RECORD_PLAIN_USER
    assert "normal prose" in cs.RECORD_PLAIN_SYSTEM
    assert "all facts" in cs.RECORD_PLAIN_USER.casefold()


def test_plain_direct_uses_baseline_qa_prompt_verbatim():
    system, user = cs._answer_condition_prompts(
        "A-PLAIN-DIRECT",
        {"passage": "PASSAGE TEXT"},
        None, {"q": "QUESTION?"})
    from ab.run_ab import QA_PROMPT
    assert system == cs.QA_SYSTEM
    assert user == QA_PROMPT.format(context="PASSAGE TEXT",
                                    question="QUESTION?")


def test_decode_prompt_expands_only_the_answer():
    user = cs.DECODE_USER.format(answer="aurora brstl 1848 iron")
    assert "aurora brstl 1848 iron" in user
    assert "Passage:" not in user and "Record:" not in user
    assert "ONLY this answer" in user
    assert "Plain English:" in user


def test_answer_from_record_prompts_normal_english():
    system, user = cs._answer_condition_prompts(
        "A-FROM-CABLESE", {"passage": "P"}, "RCRD TEXT", {"q": "Q?"})
    assert system == cs.QA_SYSTEM
    assert "RCRD TEXT" in user and "Q?" in user
    assert "normal English" in user
    assert "Passage:" not in user  # the record, not the passage


# ---------------- stub end-to-end structure ----------------

def test_stub_run_all_conditions_24_answers_per_passage(tmp_path):
    td = make_tasks_dir(tmp_path, 1)
    summary = run_stub(td, tmp_path / "r")
    assert set(summary["conditions"]) == set(cs.ALL_CONDITIONS)
    for cond in cs.ANSWER_CONDITIONS:
        block = summary["conditions"][cond]["overall_raw"]
        assert block["n"] == 24, cond
    for cond in ("R-PLAIN", "R-CABLESE"):
        assert "overall_raw" not in summary["conditions"][cond] or \
            summary["conditions"][cond]["overall_raw"]["n"] == 0


def test_every_miss_in_every_condition_judged(tmp_path):
    td = make_tasks_dir(tmp_path, 1)
    summary = run_stub(td, tmp_path / "r")
    for cond in cs.ANSWER_CONDITIONS:
        block = summary["conditions"][cond]
        misses = block["overall_raw"]["n"] - block["overall_raw"]["correct"]
        assert misses > 0, cond  # stub answers never match
        assert block["diagnostics"]["judge"]["misses_judged"] == misses
        assert block["diagnostics"]["judge"]["parse_rate"] == 1.0


def test_decode_answers_carry_lexical_similarity(tmp_path):
    td = make_tasks_dir(tmp_path, 1)
    run_stub(td, tmp_path / "r")
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    decodes = [r for r in recs
               if r.get("type") == "cond_answer"
               and r["condition"] == "DECODE"]
    assert len(decodes) == 24
    for r in decodes:
        assert 0.0 <= r["lexical_sim_vs_plain_direct"] <= 1.0
        assert r["cablese_answer"]


# ---------------- record-token accounting ----------------

def test_record_token_accounting_positive_savings(tmp_path):
    td = make_tasks_dir(tmp_path, 2)
    summary = run_stub(td, tmp_path / "r")
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    plain = {r["task_id"]: r for r in recs
             if r.get("type") == "record" and r["condition"] == "R-PLAIN"}
    cablese = {r["task_id"]: r for r in recs
               if r.get("type") == "record"
               and r["condition"] == "R-CABLESE"}
    assert set(plain) == set(cablese) == {"t00", "t01"}
    for tid, r in plain.items():
        assert r["approx_tokens"] == approx_count(r["text"])
        assert cablese[tid]["approx_tokens"] \
            == approx_count(cablese[tid]["text"])
        assert cablese[tid]["approx_tokens"] \
            < r["approx_tokens"]  # stub cablese is shorter
    assert summary["cablese_savings_pct"] > 0
    assert summary["record_tokens"]["plain_approx_tokens_total"] \
        == sum(r["approx_tokens"] for r in plain.values())
    assert summary["record_tokens"]["cablese_approx_tokens_total"] \
        == sum(r["approx_tokens"] for r in cablese.values())
    assert len(summary["record_tokens"]["per_passage"]) == 2


def test_savings_pct_math():
    # 100 -> 50 tokens must report exactly 50.0
    s = cs.summarize_cablese({}, {}, [],
                            [record_rec("R-PLAIN", "t0", 100),
                             record_rec("R-CABLESE", "t0", 50)])
    assert s["cablese_savings_pct"] == 50.0


def test_savings_none_without_plain_tokens():
    s = cs.summarize_cablese({}, {}, [],
                             [record_rec("R-CABLESE", "t0", 50)])
    assert s["cablese_savings_pct"] is None


# ---------------- pairing + McNemar ----------------

def test_mcnemar_exact_known_value():
    # b=1, c=9 discordant pairs: exact two-sided binomial p
    # = 2 * sum_{i<=1} C(10,i) / 2^10 = 2*11/1024
    hits_a = {i: (i == 0) for i in range(10)}
    hits_b = {i: (i != 0) for i in range(10)}
    out = cs.mcnemar_exact(hits_a, hits_b)
    assert out == {"n": 10, "a": 0, "b": 1, "c": 9, "d": 0,
                   "p_value": round(22 / 1024, 6)}


def test_mcnemar_symmetry_and_degenerates():
    empty = cs.mcnemar_exact({}, {})
    assert empty["p_value"] is None and empty["n"] == 0
    same = cs.mcnemar_exact({0: True, 1: False}, {0: True, 1: False})
    assert same["p_value"] == 1.0 and same["b"] == same["c"] == 0
    one_sided = cs.mcnemar_exact({0: True}, {0: False})
    assert one_sided["p_value"] == 1.0  # 2 * 1/2
    a = cs.mcnemar_exact({0: True, 1: False}, {0: False, 1: True})
    b = cs.mcnemar_exact({0: False, 1: True}, {0: True, 1: False})
    assert a["p_value"] == b["p_value"] == 1.0  # b==c==1


def test_wilson_ci_sanity():
    # re-exported unchanged from ab/baseline.py
    lo, hi = cs.wilson_ci(8, 10)
    p = 0.8
    assert lo < p < hi
    assert cs.wilson_ci(0, 0) is None
    assert cs.wilson_ci(10, 10)[1] == 1.0


def synthetic_condition_material():
    answers = {c: [] for c in cs.ANSWER_CONDITIONS}
    # A-FROM-PLAIN: 2 raw hits + 1 grader_wrong miss -> corrected 3/4
    specs_plain = [True, True, False, False]
    for i, ok in enumerate(specs_plain):
        answers["A-FROM-PLAIN"].append(
            cond_answer("A-FROM-PLAIN", "t0", i, "fact", "fact", ok))
    # A-FROM-CABLESE: 1 raw hit + 1 grader_wrong miss -> corrected 2/4
    for i, ok in enumerate([True, False, False, False]):
        answers["A-FROM-CABLESE"].append(
            cond_answer("A-FROM-CABLESE", "t0", i, "fact", "fact", ok))
    # A-PLAIN-DIRECT: 2/4 raw, none corrected
    for i, ok in enumerate([True, True, False, False]):
        answers["A-PLAIN-DIRECT"].append(
            cond_answer("A-PLAIN-DIRECT", "t0", i, "fact", "fact", ok))
    # DECODE: 3/4 raw, none corrected
    for i, ok in enumerate([True, True, True, False]):
        answers["DECODE"].append(
            cond_answer("DECODE", "t0", i, "fact", "fact", ok))
    # A-CABLESE-DIRECT: unused in headline ratios
    for i, ok in enumerate([False] * 4):
        answers["A-CABLESE-DIRECT"].append(
            cond_answer("A-CABLESE-DIRECT", "t0", i, "fact", "fact", ok))
    judges = ([judge_rec("A-FROM-PLAIN", "t0", 2, "grader_wrong"),
               judge_rec("A-FROM-PLAIN", "t0", 3, "wrong"),
               judge_rec("A-FROM-CABLESE", "t0", 1, "grader_wrong"),
               judge_rec("A-FROM-CABLESE", "t0", 2, "wrong"),
               judge_rec("A-FROM-CABLESE", "t0", 3, "wrong"),
               judge_rec("A-PLAIN-DIRECT", "t0", 2, "wrong"),
               judge_rec("A-PLAIN-DIRECT", "t0", 3, "wrong"),
               judge_rec("DECODE", "t0", 3, "wrong"),
               judge_rec("A-CABLESE-DIRECT", "t0", 0, "wrong"),
               judge_rec("A-CABLESE-DIRECT", "t0", 1, "wrong"),
               judge_rec("A-CABLESE-DIRECT", "t0", 2, "wrong"),
               judge_rec("A-CABLESE-DIRECT", "t0", 3, "wrong")])
    judges_by_cond = {c: [j for j in judges if j["condition"] == c]
                      for c in cs.ANSWER_CONDITIONS}
    records = [record_rec("R-PLAIN", "t0", 100),
               record_rec("R-CABLESE", "t0", 40)]
    return answers, judges_by_cond, records


def test_headline_recovery_and_decode_ratios():
    answers, judges, records = synthetic_condition_material()
    answers["DECODE"][0]["lexical_sim_vs_plain_direct"] = 0.5
    answers["DECODE"][1]["lexical_sim_vs_plain_direct"] = 1.0
    s = cs.summarize_cablese(answers, judges, [], records)
    # SPEC-VALIDATION: ratios use grader v2 RAW accuracy (judge is
    # diagnostic only): from-plain 0.5, from-cablese 0.25,
    # plain-direct 0.5, decode 0.75
    assert s["recovery_ratio"] == round((0.25 / 0.5), 6)
    assert s["decode_ratio"] == 1.5
    assert s["cablese_savings_pct"] == 60.0
    assert s["decode_fidelity"] == {"n": 2,
                                    "mean_lexical_similarity": 0.75}
    # McNemar cells (raw correctness, judge corrections not folded in):
    # from-cablese [T,F,F,F] vs from-plain [T,T,F,F] -> one plain-only hit
    assert s["mcnemar"]["A-FROM-CABLESE_vs_A-FROM-PLAIN"]["b"] == 0
    assert s["mcnemar"]["A-FROM-CABLESE_vs_A-FROM-PLAIN"]["c"] == 1
    # decode vs plain-direct: q2 decode-only hit -> b=1, c=0
    assert s["mcnemar"]["DECODE_vs_A-PLAIN-DIRECT"]["b"] == 1
    assert s["mcnemar"]["DECODE_vs_A-PLAIN-DIRECT"]["c"] == 0


def test_ratios_none_on_zero_denominator():
    answers, judges, records = synthetic_condition_material()
    for r in answers["A-FROM-PLAIN"]:
        r["correct"] = False
    judges["A-FROM-PLAIN"] = [judge_rec("A-FROM-PLAIN", "t0", i, "wrong")
                              for i in range(4)]
    s = cs.summarize_cablese(answers, judges, [], records)
    assert s["recovery_ratio"] is None


def test_lexical_similarity_values():
    assert cs.lexical_similarity("aurora bristol iron",
                                 "aurora bristol iron") == 1.0
    assert cs.lexical_similarity("alpha beta", "gamma delta") == 0.0
    assert cs.lexical_similarity("alpha beta gamma",
                                 "alpha beta delta") == round(2 / 4, 6)
    assert cs.lexical_similarity("", "x") == 0.0


# ---------------- slicing + merge serial equivalence ----------------

CORE = ["conditions", "mcnemar", "record_tokens", "decode_fidelity",
        "cablese_savings_pct", "recovery_ratio", "decode_ratio"]


def test_serial_vs_sliced_serial_equivalence(tmp_path):
    td = make_tasks_dir(tmp_path, 6)
    serial = run_stub(td, tmp_path / "serial")
    par = tmp_path / "par"
    for i, (off, lim) in enumerate([(0, 2), (2, 2), (4, 2)]):
        s = run_stub(td, par, task_offset=off, task_limit=lim,
                     slice_id=f"{i:04d}")
        assert s["task_offset"] == off
        assert s["slice_id"] == f"{i:04d}"
    merged = cs.merge_cablese_slices(str(par))
    for key in CORE:
        assert merged[key] == serial[key], key
    assert merged["slices_merged"] == ["slice_0000.jsonl",
                                       "slice_0001.jsonl",
                                       "slice_0002.jsonl"]
    assert merged["passages"] == serial["passages"] == 6
    # merged ledger contains exactly the serial run's cond answers
    def answers(path):
        return sorted(
            (r["condition"], r["task_id"], r["q_index"])
            for r in map(json.loads, path.read_text().splitlines())
            if r.get("type") == "cond_answer")
    assert answers(tmp_path / "serial" / "ledger.jsonl") \
        == answers(par / "ledger.jsonl")


def test_slice_out_of_range_and_merge_errors(tmp_path):
    td = make_tasks_dir(tmp_path, 2)
    with pytest.raises(cs.CableseError):
        run_stub(td, tmp_path / "r", task_offset=5)
    with pytest.raises(cs.CableseError):
        cs.merge_cablese_slices(str(tmp_path / "nowhere"))


def test_slice_ledger_records_carry_git_sha_and_raw(tmp_path):
    td = make_tasks_dir(tmp_path, 1)
    run_stub(td, tmp_path / "r", task_offset=0, task_limit=1,
             slice_id="0042")
    path = tmp_path / "r" / "slices" / "slice_0042.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    assert recs
    for rec in recs:
        assert rec.get("git_sha")
        assert rec["run_id"].startswith("cablese_")
    assert {r["slice_id"] for r in recs} == {"0042"}
    calls = [r for r in recs if r["type"] == "model_call"]
    assert all("raw" in r and r["raw"] for r in calls)
    kinds = {r["call"] for r in calls}
    assert {"gen_half1", "gen_half2", "record_R-PLAIN",
            "record_R-CABLESE", "cond_qa", "decode", "judge"} <= kinds


# ---------------- CLI (incl. the 14475 dispatch regression) ----------------

def test_cli_dispatch_passes_offset_and_slice_args_14475(tmp_path,
                                                         monkeypatch):
    """REGRESSION (expB incident 14475): the CLI dispatch dropped the
    slice args before they reached the run function. Asserts the
    cablese dispatch passes --task-offset/--task-limit/--slice-id (and
    --passages) through, by capturing the actual call."""
    captured = {}

    def fake_run(client, model, max_tokens, results_dir, **kw):
        captured.update(kw)
        captured["client_is_stub"] = client is not None
        return {}

    monkeypatch.setattr(cs, "run_cablese", fake_run)
    rc = cs.main(["cablese", "--stub",
                  "--results-dir", str(tmp_path / "r"),
                  "--passages", "5",
                  "--task-offset", "3",
                  "--task-limit", "2",
                  "--slice-id", "0007"])
    assert rc == 0
    assert captured["task_offset"] == 3
    assert captured["task_limit"] == 2
    assert captured["slice_id"] == "0007"
    assert captured["passages"] == 5


def test_cli_dispatch_defaults_when_no_slice_args(tmp_path, monkeypatch):
    captured = {}

    def fake_run(client, model, max_tokens, results_dir, **kw):
        captured.update(kw)

    monkeypatch.setattr(cs, "run_cablese", fake_run)
    rc = cs.main(["cablese", "--stub",
                  "--results-dir", str(tmp_path / "r")])
    assert rc == 0
    assert captured["task_offset"] == 0
    assert captured["task_limit"] is None
    assert captured["slice_id"] is None


def test_cli_slice_end_to_end_creates_slice_files(tmp_path):
    td = make_tasks_dir(tmp_path, 3)
    rc = cs.main(["cablese", "--stub", "--tasks-dir", str(td),
                  "--results-dir", str(tmp_path / "r"),
                  "--task-offset", "1", "--task-limit", "1",
                  "--slice-id", "0001"])
    assert rc == 0
    sl = tmp_path / "r" / "slices" / "slice_0001.jsonl"
    su = tmp_path / "r" / "slices" / "slice_0001.summary.json"
    assert sl.exists() and su.exists()
    summary = json.loads(su.read_text())
    assert summary["passages"] == 1
    assert not (tmp_path / "r" / "ledger.jsonl").exists()


def test_cli_cablese_merge_subcommand(tmp_path):
    td = make_tasks_dir(tmp_path, 2)
    for i, (off, lim) in enumerate([(0, 1), (1, 1)]):
        assert cs.main(["cablese", "--stub", "--tasks-dir", str(td),
                        "--results-dir", str(tmp_path / "r"),
                        "--task-offset", str(off), "--task-limit", "1",
                        "--slice-id", f"{i:04d}"]) == 0
    rc = cs.main(["cablese-merge", "--results-dir", str(tmp_path / "r")])
    assert rc == 0
    merged = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert merged["passages"] == 2
    assert (tmp_path / "r" / "ledger.jsonl").exists()


def test_cli_cablese_merge_missing_dir_fails(tmp_path, capsys):
    rc = cs.main(["cablese-merge", "--results-dir",
                  str(tmp_path / "void")])
    assert rc == 2
    assert "no slices directory" in capsys.readouterr().err


# ---------------- summary shape ----------------

def test_summary_shape(tmp_path):
    td = make_tasks_dir(tmp_path, 2)
    summary = run_stub(td, tmp_path / "r")
    for key in CORE + ["run_id", "git_sha", "passages"]:
        assert key in summary, key
    for cond in cs.ANSWER_CONDITIONS:
        block = summary["conditions"][cond]
        for sub in ("overall_raw", "bootstrap_raw_95", "per_type_raw",
                    "per_register_raw", "diagnostics"):
            assert sub in block, (cond, sub)
        # SPEC-VALIDATION: no corrected metric, no top-level judge
        for sub in ("overall_corrected", "bootstrap_corrected_95",
                    "per_type_corrected", "per_register_corrected",
                    "judge", "miss_taxonomy"):
            assert sub not in block, (cond, sub)
        assert block["overall_raw"]["n"] == 48  # 2 passages x 24
    assert set(summary["mcnemar"]) == {
        "A-FROM-CABLESE_vs_A-FROM-PLAIN", "DECODE_vs_A-PLAIN-DIRECT"}
    assert summary["recovery_ratio"] is None  # stub: zero accuracy
    assert summary["decode_ratio"] is None


# ---------------- slurm bundle ----------------

def test_sbatch_passes_bash_n():
    out = subprocess.run(["bash", "-n",
                          os.path.join(ROOT, "slurm", "cablese.sbatch")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_sbatch_contract():
    text = open(os.path.join(ROOT, "slurm", "cablese.sbatch"),
                encoding="utf-8").read()
    assert "#SBATCH --array=0-23" in text
    assert "#SBATCH --export=ALL" in text
    assert "TCB_CABLESE_RUN_ID" in text
    assert "TCB_MAX_TOKENS" in text
    assert "${TCB_MAX_TOKENS:-4000}" in text
    assert "ab.cablese cablese-merge" in text
    assert "--task-offset" in text and "--task-limit" in text \
        and "--slice-id" in text
    assert "TCB_TREAT_401_TRANSIENT=1" in text
    assert "--time=04:00:00" in text


def test_sbx_llm_block():
    sbx = json.load(open(os.path.join(ROOT, "slurm", "cablese.sbx.json"),
                         encoding="utf-8"))
    llm = sbx["llm"]
    assert llm["model"] == "glm53_flash"
    assert llm["lanes"] == ["zai-max"]
    assert llm["requests"] == 7000
    assert llm["hours"] == 4
    assert llm["mode"] == "advise"


def test_sbatch_tiling_formula_covers_50_exactly():
    """Replicates the sbatch arithmetic in bash for TOTAL=50/24 slots:
    2 slots of 3 + 22 slots of 2 = 50, contiguous, no overlap."""
    script = r'''
TOTAL=50; SLOTS=24
BASE=$((TOTAL / SLOTS)); REM=$((TOTAL % SLOTS))
for ID in $(seq 0 $((SLOTS - 1))); do
  LIMIT=$((BASE + (ID < REM ? 1 : 0)))
  OFFSET=$((ID * BASE + (ID < REM ? ID : REM)))
  echo "$OFFSET $LIMIT"
done
'''
    out = subprocess.run(["bash", "-c", script], capture_output=True,
                         text=True)
    assert out.returncode == 0
    spans = [tuple(map(int, line.split()))
             for line in out.stdout.strip().splitlines()]
    assert len(spans) == 24
    assert sum(lim for _, lim in spans) == 50
    cursor = 0
    for off, lim in spans:
        assert off == cursor
        cursor += lim
    assert cursor == 50
