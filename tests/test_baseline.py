"""SPEC-BASELINE Phase 1 tests (offline; stub client everywhere, no
network). Covers stratification, dedupe incl. the 0.6 boundary, judge
verdict parsing, miss taxonomy math, Wilson CI known values, slicing +
merge serial-equivalence, the CLI dispatch regression (expB incident
14475: dispatch dropping slice args), and summary shape."""

import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ab.baseline as bl  # noqa: E402
from ab.expAB import Ledger  # noqa: E402

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
    return bl.run_baseline(bl.make_stub_client(), "stub-model", 512,
                           str(results_dir), tasks_dir=str(tasks_dir), **kw)


def answer(task_id, q_index, q_type, register, correct):
    return {"type": "quiz_answer", "task_id": task_id,
            "register": register, "len_class": "medium",
            "q_index": q_index, "q_type": q_type,
            "question": f"q{q_index}", "expected": "e",
            "raw_answer": "a", "correct_kind": "exact", "correct": correct}


# ---------------- generation + stratification ----------------

def test_stratification_counts_stub(tmp_path):
    ledger = Ledger(str(tmp_path / "led.jsonl"),
                    {"run_id": "x", "git_sha": "t"})
    task = json.loads((make_tasks_dir(tmp_path, 1) / "t00.json")
                      .read_text())
    kept, dropped, unanchored = bl.generate_stratified_quiz(
        bl.make_stub_client(), "stub-model", task, 512, ledger)
    counts = {}
    for item in kept:
        counts[item["q_type"]] = counts.get(item["q_type"], 0) + 1
    assert counts == {"fact": 12, "numeric": 4, "ordering": 4,
                      "inference": 4}
    assert len(kept) + len(dropped) == 24
    assert dropped == []  # stub questions are lexically distinct
    assert unanchored == []  # stub answers are passage spans


# ---------------- dedupe ----------------

def test_token_overlap_values():
    a = "Who was the master of the brig?"
    assert bl.token_overlap(a, a) == 1.0
    assert bl.token_overlap(a, "Completely different topic entirely") == 0.0
    ov = bl.token_overlap("alpha beta gamma delta epsilon",
                          "alpha beta gamma zeta eta theta")
    assert abs(ov - 0.6) < 1e-9  # 3 shared / min(5, 5)


def test_dedupe_drops_at_exactly_threshold():
    items = [
        {"q": "alpha beta gamma delta epsilon", "a": "x", "q_type": "fact"},
        {"q": "alpha beta gamma zeta eta theta", "a": "x",
         "q_type": "fact"},  # overlap exactly 0.6 -> dropped (>= rule)
    ]
    kept, dropped = bl.dedupe_questions(items)
    assert len(kept) == 1
    assert len(dropped) == 1
    assert dropped[0]["kept_index"] == 0
    assert abs(dropped[0]["overlap"] - 0.6) < 1e-6


def test_dedupe_keeps_below_threshold():
    items = [
        {"q": "alpha beta gamma delta epsilon", "a": "x", "q_type": "fact"},
        {"q": "alpha beta zeta eta theta iota", "a": "x",
         "q_type": "fact"},  # 2/5 = 0.4 -> kept
    ]
    kept, dropped = bl.dedupe_questions(items)
    assert len(kept) == 2 and dropped == []


def test_dedupe_drops_later_duplicate_and_records_it(tmp_path):
    items = [
        {"q": "cargo iron sailed march bristol", "a": "x", "q_type": "fact"},
        {"q": "Who was the master of the brig Aurora?",
         "a": "x", "q_type": "fact"},
        {"q": "who was the master of the brig aurora?", "a": "x",
         "q_type": "fact"},  # near-identical to item 2 -> dropped
    ]
    kept, dropped = bl.dedupe_questions(items)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert dropped[0]["item"]["q"] == items[2]["q"]
    assert dropped[0]["kept_index"] == 1  # the LATER one is dropped


def test_dedupe_run_records_drops_in_ledger(tmp_path):
    td = make_tasks_dir(tmp_path, 1)
    task = json.loads((td / "t00.json").read_text())
    real = bl.generate_stratified_quiz

    def dup_gen(client, model, t, max_tokens, ledger):
        items = [{"q": f"entry{i}a entry{i}b entry{i}c entry{i}d",
                  "a": "Captain Elias Hoy", "q_type": "fact"}
                 for i in range(3)]
        items.append({"q": "entry1a entry1b entry1c entry1d again",
                      "a": "Captain Elias Hoy", "q_type": "fact"})
        kept, dropped = bl.dedupe_questions(items)
        return kept, dropped, []

    bl.generate_stratified_quiz = dup_gen
    try:
        summary = run_stub(td, tmp_path / "r")
    finally:
        bl.generate_stratified_quiz = real
    assert summary["dedupe"]["dropped"] == 1
    assert summary["dedupe"]["kept"] == 3
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    drops = [r for r in recs if r["type"] == "dedupe_drop"]
    assert len(drops) == 1 and drops[0]["overlap"] >= 0.6


# ---------------- judge verdicts + taxonomy ----------------

def test_parse_verdict_variants():
    assert bl.parse_verdict("grader_wrong") == "grader_wrong"
    assert bl.parse_verdict("  Wrong.\n") == "wrong"
    assert bl.parse_verdict("The answer is PARTIAL here") == "partial"
    assert bl.parse_verdict("**ambiguous**") == "ambiguous"
    assert bl.parse_verdict("I think it is not grader_wrong but wrong") \
        == "grader_wrong"  # grader_wrong checked before wrong
    assert bl.parse_verdict("cannot decide at all") == "unparsed"
    assert bl.parse_verdict("") == "unparsed"


def _synthetic_answers():
    # 10 answers: 6 correct, 4 misses across 2 passages
    out = []
    specs = [("t0", "fact", True), ("t0", "fact", True),
             ("t0", "numeric", False), ("t0", "ordering", True),
             ("t0", "inference", False), ("t1", "fact", True),
             ("t1", "numeric", True), ("t1", "ordering", False),
             ("t1", "inference", False), ("t1", "fact", True)]
    for i, (tid, qt, ok) in enumerate(specs):
        out.append(answer(tid, i, qt, "fact" if tid == "t0" else "prose",
                          ok))
    return out


def test_taxonomy_covers_all_misses():
    answers = _synthetic_answers()
    judges = [
        {"type": "judge", "task_id": "t0", "q_index": 2,
         "sampled_hit": False, "verdict": "grader_wrong"},
        {"type": "judge", "task_id": "t0", "q_index": 4,
         "sampled_hit": False, "verdict": "wrong"},
        {"type": "judge", "task_id": "t1", "q_index": 7,
         "sampled_hit": False, "verdict": "partial"},
        {"type": "judge", "task_id": "t1", "q_index": 8,
         "sampled_hit": False, "verdict": "ambiguous"},
        {"type": "judge", "task_id": "t1", "q_index": 9,
         "sampled_hit": True, "verdict": "wrong"},
    ]
    s = bl.summarize_baseline(answers, judges, [])
    tax = s["diagnostics"]["miss_taxonomy"]
    n_misses = sum(1 for a in answers if not a["correct"])
    assert sum(tax.values()) == n_misses == 4
    assert tax["grader_wrong"] == 1 and tax["wrong"] == 1
    assert tax["partial"] == 1 and tax["ambiguous"] == 1
    assert tax["unjudged"] == 0 and tax["unparsed"] == 0


def test_miss_taxonomy_counts_unjudged_and_unparsed():
    answers = _synthetic_answers()
    judges = [
        {"type": "judge", "task_id": "t0", "q_index": 2,
         "sampled_hit": False, "verdict": "unparsed"},
        # t1 q8 never judged
        {"type": "judge", "task_id": "t1", "q_index": 7,
         "sampled_hit": False, "verdict": "wrong"},
        {"type": "judge", "task_id": "t0", "q_index": 4,
         "sampled_hit": False, "verdict": "wrong"},
    ]
    s = bl.summarize_baseline(answers, judges, [])
    assert s["diagnostics"]["miss_taxonomy"]["unparsed"] == 1
    assert s["diagnostics"]["miss_taxonomy"]["unjudged"] == 1
    assert sum(s["diagnostics"]["miss_taxonomy"].values()) == 4


def test_judge_demoted_out_of_truth_path():
    """SPEC-VALIDATION §1: judge verdicts are diagnostics — a
    grader_wrong overturn must NOT change any headline number."""
    answers = _synthetic_answers()
    judges = [
        {"type": "judge", "task_id": "t0", "q_index": 2,
         "sampled_hit": False, "verdict": "grader_wrong"},
        {"type": "judge", "task_id": "t1", "q_index": 7,
         "sampled_hit": False, "verdict": "wrong"},
        {"type": "judge", "task_id": "t1", "q_index": 8,
         "sampled_hit": False, "verdict": "wrong"},
        {"type": "judge", "task_id": "t0", "q_index": 4,
         "sampled_hit": False, "verdict": "grader_wrong"},
    ]
    s = bl.summarize_baseline(answers, judges, [])
    assert s["overall_raw"]["n"] == 10
    assert s["overall_raw"]["correct"] == 6
    assert s["overall_raw"]["accuracy"] == 0.6
    # the 2 judge grader_wrong overturns change NOTHING in the headline
    for key in ("overall_corrected", "bootstrap_corrected_95",
                "per_type_corrected", "per_register_corrected", "judge",
                "miss_taxonomy", "judge_correction_note"):
        assert key not in s, key
    assert s["diagnostics"]["judge"]["grader_false_negative_rate"] == 0.5
    # a sampled hit verdict still feeds only the FP estimate
    judges.append({"type": "judge", "task_id": "t1", "q_index": 9,
                   "sampled_hit": True, "verdict": "wrong"})
    s2 = bl.summarize_baseline(answers, judges, [])
    assert s2["overall_raw"]["correct"] == 6
    assert s2["diagnostics"]["judge"][
        "grader_false_positive_rate_estimate"] == 1.0


# ---------------- statistics ----------------

def test_wilson_ci_known_value():
    # independently-written arithmetic (Wilson 1927 formula, z=1.959964)
    z = 1.959964
    k, n = 59, 100
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = bl.wilson_ci(k, n)
    assert abs(lo - (center - half)) < 1e-6
    assert abs(hi - (center + half)) < 1e-6
    assert lo < p < hi


def test_wilson_ci_degenerate():
    assert bl.wilson_ci(0, 10)[0] == 0.0
    assert bl.wilson_ci(10, 10)[1] == 1.0
    assert bl.wilson_ci(0, 0) is None
    lo, hi = bl.wilson_ci(5, 5)
    assert lo > 0.5 and hi == 1.0


def test_bootstrap_deterministic_and_bracketing():
    # two passages with DIFFERENT per-passage accuracy so resampling
    # actually spreads (a constant 0.6/0.6 corpus has a degenerate CI)
    answers = ([answer("t0", i, "fact", "fact", True) for i in range(4)]
               + [answer("t1", i, "fact", "prose", False)
                  for i in range(4)])
    ci1 = bl.bootstrap_overall_ci(answers)
    ci2 = bl.bootstrap_overall_ci(answers)
    assert ci1 == ci2  # seeded -> deterministic
    raw = 4 / 8
    assert ci1[0] <= raw <= ci1[1]
    assert 0.0 <= ci1[0] < ci1[1] <= 1.0


# ---------------- slicing + merge serial equivalence ----------------

CORE = ["overall_raw", "bootstrap_raw_95", "per_type_raw",
        "per_register_raw", "diagnostics", "dedupe", "per_passage"]


def test_serial_vs_sliced_serial_equivalence(tmp_path):
    td = make_tasks_dir(tmp_path, 6)
    serial = run_stub(td, tmp_path / "serial")
    par = tmp_path / "par"
    for i, (off, lim) in enumerate([(0, 2), (2, 2), (4, 2)]):
        s = run_stub(td, par, task_offset=off, task_limit=lim,
                     slice_id=f"{i:04d}")
        assert s["task_offset"] == off
        assert s["slice_id"] == f"{i:04d}"
    merged = bl.merge_baseline_slices(str(par))
    for key in CORE:
        assert merged[key] == serial[key], key
    assert merged["slices_merged"] == ["slice_0000.jsonl",
                                       "slice_0001.jsonl",
                                       "slice_0002.jsonl"]
    assert merged["passages"] == serial["passages"] == 6
    # merged ledger contains exactly the serial run's quiz answers
    serial_answers = [json.loads(l) for l in
                      (tmp_path / "serial" / "ledger.jsonl")
                      .read_text().splitlines()]
    merged_answers = [json.loads(l) for l in
                      (par / "ledger.jsonl").read_text().splitlines()]
    key = lambda r: (r.get("type"), r.get("task_id"), r.get("q_index"))  # noqa: E731
    assert (sorted(map(str, map(key, serial_answers)))
            == sorted(map(str, map(key, merged_answers))))


def test_slice_out_of_range_and_merge_errors(tmp_path):
    td = make_tasks_dir(tmp_path, 2)
    import pytest
    with pytest.raises(bl.BaselineError):
        run_stub(td, tmp_path / "r", task_offset=5)
    with pytest.raises(bl.BaselineError):
        bl.merge_baseline_slices(str(tmp_path / "nowhere"))


def test_slice_ledger_records_carry_git_sha_and_raw(tmp_path):
    td = make_tasks_dir(tmp_path, 1)
    run_stub(td, tmp_path / "r", task_offset=0, task_limit=1,
             slice_id="0042")
    path = tmp_path / "r" / "slices" / "slice_0042.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    assert recs
    for rec in recs:
        assert rec.get("git_sha")
        assert rec["run_id"].startswith("baseline_")
    assert {r["slice_id"] for r in recs} == {"0042"}
    calls = [r for r in recs if r["type"] == "model_call"]
    assert all("raw" in r and r["raw"] for r in calls)
    kinds = {r["call"] for r in calls}
    assert {"gen_half1", "gen_half2", "quiz_qa", "judge"} <= kinds


# ---------------- CLI (incl. the 14475 dispatch regression) ----------------

def test_cli_dispatch_passes_offset_and_slice_args_14475(tmp_path,
                                                         monkeypatch):
    """REGRESSION (expB incident 14475): the CLI dispatch dropped the
    slice args before they reached the run function. Asserts the
    baseline dispatch passes --task-offset/--task-limit/--slice-id
    (and --passages) through, by capturing the actual call."""
    captured = {}

    def fake_run(client, model, max_tokens, results_dir, **kw):
        captured.update(kw)
        captured["client_is_stub"] = client is not None
        return {}

    monkeypatch.setattr(bl, "run_baseline", fake_run)
    rc = bl.main(["baseline", "--stub",
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

    monkeypatch.setattr(bl, "run_baseline", fake_run)
    rc = bl.main(["baseline", "--stub",
                  "--results-dir", str(tmp_path / "r")])
    assert rc == 0
    assert captured["task_offset"] == 0
    assert captured["task_limit"] is None
    assert captured["slice_id"] is None


def test_cli_slice_end_to_end_creates_slice_files(tmp_path):
    td = make_tasks_dir(tmp_path, 3)
    rc = bl.main(["baseline", "--stub", "--tasks-dir", str(td),
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


def test_cli_baseline_merge_subcommand(tmp_path):
    td = make_tasks_dir(tmp_path, 2)
    for i, (off, lim) in enumerate([(0, 1), (1, 1)]):
        assert bl.main(["baseline", "--stub", "--tasks-dir", str(td),
                        "--results-dir", str(tmp_path / "r"),
                        "--task-offset", str(off), "--task-limit", "1",
                        "--slice-id", f"{i:04d}"]) == 0
    rc = bl.main(["baseline-merge", "--results-dir", str(tmp_path / "r")])
    assert rc == 0
    merged = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert merged["passages"] == 2
    assert (tmp_path / "r" / "ledger.jsonl").exists()


def test_cli_baseline_merge_missing_dir_fails(tmp_path, capsys):
    rc = bl.main(["baseline-merge", "--results-dir",
                  str(tmp_path / "void")])
    assert rc == 2
    assert "no slices directory" in capsys.readouterr().err


# ---------------- end-to-end summary shape ----------------

def test_summary_shape(tmp_path):
    td = make_tasks_dir(tmp_path, 2)
    summary = run_stub(td, tmp_path / "r")
    for key in CORE + ["run_id", "git_sha", "passages",
                       "questions_per_passage_plan", "bank_validation",
                       "generated_validation"]:
        assert key in summary, key
    # SPEC-VALIDATION: no corrected metric anywhere in the headline
    for key in ("overall_corrected", "bootstrap_corrected_95",
                "per_type_corrected", "per_register_corrected",
                "judge", "miss_taxonomy", "judge_correction_note"):
        assert key not in summary, key
    assert summary["questions_per_passage_plan"] == 24
    for block in (summary["overall_raw"],):
        assert set(block) == {"n", "correct", "accuracy", "wilson_95"}
        assert block["n"] == 48  # 2 passages x 24
        assert block["wilson_95"][0] <= block["accuracy"] \
            <= block["wilson_95"][1]
    assert set(summary["per_type_raw"]) == {"fact", "numeric",
                                            "ordering", "inference"}
    assert set(summary["per_register_raw"]) == {"fact", "prose"}
    assert summary["bank_validation"] == {
        "selected": 2, "anchored": 2, "excluded_unanchored": 0}
    assert summary["generated_validation"]["rejected_unanchored"] == 0
    dj = summary["diagnostics"]["judge"]
    assert dj["parse_rate"] == 1.0  # stub verdict 'wrong'
    assert dj["misses_judged"] == 48
    assert dj["hits_sampled"] == 0  # stub: zero hits
    assert "miss_taxonomy" in summary["diagnostics"]
    assert len(summary["per_passage"]) == 2
    for pp in summary["per_passage"]:
        assert set(pp) == {"task_id", "register", "n", "raw_correct",
                           "raw_accuracy"}


# ---------------- slurm bundle ----------------

def test_sbatch_passes_bash_n():
    out = subprocess.run(["bash", "-n",
                          os.path.join(ROOT, "slurm", "baseline.sbatch")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_sbatch_contract():
    text = open(os.path.join(ROOT, "slurm", "baseline.sbatch"),
                encoding="utf-8").read()
    assert "#SBATCH --array=0-23" in text
    assert "#SBATCH --export=ALL" in text
    assert "TCB_BASELINE_RUN_ID" in text
    assert "TCB_MAX_TOKENS" in text
    assert "ab.baseline baseline-merge" in text
    assert "--task-offset" in text and "--task-limit" in text \
        and "--slice-id" in text
    assert "TCB_TREAT_401_TRANSIENT=1" in text


def test_sbx_llm_block():
    # Lane pins rot when the experiment switches provider (ds-direct ->
    # zai-max broke this pin on 2026-10-03); validate structure + live values.
    sbx = json.load(open(os.path.join(ROOT, "slurm", "baseline.sbx.json"),
                         encoding="utf-8"))
    llm = sbx["llm"]
    assert llm["model"] in ("ds-flash", "glm53_flash"), llm["model"]
    assert isinstance(llm["lanes"], list) and llm["lanes"], llm["lanes"]
    for lane in llm["lanes"]:
        assert lane in ("ds-direct", "free", "openrouter", "zai-max"), lane
    assert llm["requests"] > 0 and llm["mode"] == "advise"


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
