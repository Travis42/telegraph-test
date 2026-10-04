"""SPEC-CROSSMODEL tests (offline; stub clients everywhere, no
network). Covers the source-run guard, the SPEC-VALIDATION anchoring
gate on the reused cbl2 quiz items (logged rejections, reduced-quota
accounting), the frozen-panel 404 drop-with-note fallback,
per-reader/per-writer ratio math incl. the zero-denominator case,
McNemar reuse, tiling + merge serial equivalence, the CLI dispatch
regression for the expB 14475 class, per-call model ids in the
ledger, the v2-only summary shape (no corrected headline; judge
diagnostics isolation), the OpenRouter client contract, and the
slurm bundle."""

import json
import os
import subprocess
import sys
import urllib.error

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ab.crossmodel as xm  # noqa: E402
import ab.cablese as cs  # noqa: E402
from ab.instrument import TransientAPIError  # noqa: E402
from tcb.tokencount import approx_count  # noqa: E402

TASK_PASSAGE = (
    "The brig Aurora sailed from Bristol on 4 March 1848 with a cargo "
    "of iron. Her master was Captain Elias Hoy, and she carried a crew "
    "of nineteen. After a storm off Ushant she put into Falmouth, and "
    "later resumed her voyage to Lisbon, arriving on 2 May before "
    "continuing to Oporto, which lay north of Lisbon along the coast."
)


def make_tasks_dir(tmp_path, n=4, q_per_passage=3):
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
    return td, q_per_passage


def make_source_run(tmp_path, tasks_dir, q_per_passage=3):
    """Synthesize a cbl2-shaped source run: quiz_item + R-PLAIN +
    R-CABLESE records per passage (the record layout the cablese build
    writes; see tests/test_cablese.py and ab/cablese.py)."""
    src = tmp_path / "cbl2_source"
    src.mkdir(exist_ok=True)
    lines = [{"type": "run_meta", "run_id": "cablese_fixture"}]
    for fn in sorted(os.listdir(tasks_dir)):
        task = json.loads((tasks_dir / fn).read_text())
        for qi in range(q_per_passage):
            lines.append({
                "type": "quiz_item", "task_id": task["id"],
                "q_index": qi, "q_type": "fact",
                "question": f"q{qi} about {task['id']}?",
                "answer": "Captain Elias Hoy",
            })
        for cond, text in (
            ("R-PLAIN", "Plain record mentioning Captain Elias Hoy and "
                        "Bristol 1848 in full prose."),
            ("R-CABLESE", "rcrd: aurora brstl 1848 hoy"),
        ):
            lines.append({
                "type": "record", "condition": cond,
                "task_id": task["id"], "register": task["register"],
                "len_class": task["len_class"],
                "approx_tokens": approx_count(text), "text": text,
            })
    with open(src / "ledger.jsonl", "w", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(json.dumps(rec) + "\n")
    return str(src)


def run_stub(tasks_dir, source_run, results_dir, **kw):
    return xm.run_crossmodel(
        xm.make_stub_client(), xm.make_stub_client(), "glm-stub", 512,
        str(results_dir), source_run=source_run,
        tasks_dir=str(tasks_dir), **kw)


def cond_answer(cond, task_id, q_index, correct, panel="R1"):
    return {"type": "cond_answer", "condition": cond, "panel": panel,
            "task_id": task_id, "register": "fact",
            "len_class": "medium", "q_index": q_index,
            "q_type": "fact", "question": f"q{q_index}",
            "expected": "Captain Elias Hoy", "raw_answer": "a",
            "correct": correct}


def judge_rec(cond, task_id, q_index, verdict, sampled_hit=False):
    return {"type": "judge", "condition": cond, "task_id": task_id,
            "q_index": q_index, "sampled_hit": sampled_hit,
            "verdict": verdict, "raw": verdict}


# ---------------- source-run guard ----------------

def test_source_run_guard_missing_dir_and_ledger(tmp_path):
    with pytest.raises(xm.CrossModelError):
        xm.load_source_run(str(tmp_path / "nowhere"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(xm.CrossModelError):
        xm.load_source_run(str(empty))


def test_source_run_guard_incomplete_records(tmp_path, capsys):
    td, _ = make_tasks_dir(tmp_path, 2)
    src = make_source_run(tmp_path, td)
    lines = [json.loads(l)
             for l in open(os.path.join(src, "ledger.jsonl"))]
    # strip one passage's R-CABLESE record -> hard refuse
    kept = [r for r in lines
            if not (r.get("task_id") == "t01"
                    and r.get("condition") == "R-CABLESE")]
    with open(os.path.join(src, "ledger.jsonl"), "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    with pytest.raises(xm.CrossModelError, match="R-CABLESE"):
        xm.load_source_run(src)
    rc = xm.main(["crossmodel", "--stub", "--source-run", src,
                  "--results-dir", str(tmp_path / "r")])
    assert rc == 2
    assert "R-CABLESE" in capsys.readouterr().err


def test_source_run_guard_missing_task_in_bank(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 3)
    src = make_source_run(tmp_path, td)
    os.remove(str(td / "t02.json"))
    with pytest.raises(xm.CrossModelError, match="tasks bank"):
        run_stub(td, src, tmp_path / "r")


# ---------------- SPEC-VALIDATION anchoring gate ----------------

def make_unanchored_source(tmp_path, tasks_dir, q_per_passage=3):
    """Source run whose q_index=1 item has an answer that is NOT
    passage-anchored (grade_v2 anchor machinery must reject it)."""
    src = make_source_run(tmp_path, tasks_dir, q_per_passage)
    path = os.path.join(src, "ledger.jsonl")
    lines = [json.loads(l) for l in open(path)]
    for rec in lines:
        if rec.get("type") == "quiz_item" and rec["q_index"] == 1:
            rec["answer"] = "Zaphod Beeblebrox"  # not in any passage
    with open(path, "w", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(json.dumps(rec) + "\n")
    return src


def test_anchored_pool_filtering_with_logged_rejection(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=3)
    src = make_unanchored_source(tmp_path, td)
    summary = run_stub(td, src, tmp_path / "r")
    # rejected + logged (never silent): ledger record + sidecar file
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    rejects = [r for r in recs if r["type"] == "unanchored_reject"]
    assert len(rejects) == 1
    assert rejects[0]["answer"] == "Zaphod Beeblebrox"
    assert rejects[0]["anchor_score"] < 0.8
    sidecar = json.loads(
        (tmp_path / "r" / "unanchored_source_items.json").read_text())
    assert [r["answer"] for r in sidecar] == ["Zaphod Beeblebrox"]
    # the unanchored item is never asked; the anchored pool fills the run
    asked = {r["q_index"] for r in recs
             if r["type"] == "cond_answer"}
    assert asked == {0, 2}
    gate = summary["anchoring_gate"]
    assert gate["rejected_unanchored"] == 1
    assert gate["passages"]["t00"]["anchored_pool"] == 2
    assert summary["conditions"]["A-FROM-PLAIN/R1"]["overall_raw"]["n"] == 2


def test_reduced_quota_passage_accounting(tmp_path):
    """Pool smaller than quota -> reduced n recorded, run not failed;
    explicit quota below pool -> only the first quota items are asked."""
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    summary = run_stub(td, src, tmp_path / "r")  # default quota 24 > 2
    assert summary["anchoring_gate"]["passages"]["t00"] == {
        "quota": xm.SOURCE_QUESTIONS_PER_PASSAGE,
        "anchored_pool": 2, "n": 2}
    assert summary["conditions"]["A-FROM-PLAIN/R1"]["overall_raw"]["n"] == 2

    tmp_path.joinpath("q").mkdir()
    td3, _ = make_tasks_dir(tmp_path / "q", 1, q_per_passage=3)
    src3 = make_source_run(tmp_path / "q", td3, q_per_passage=3)
    summary3 = xm.run_crossmodel(
        xm.make_stub_client(), xm.make_stub_client(), "glm-stub", 512,
        str(tmp_path / "r3"), source_run=src3, tasks_dir=str(td3),
        quota_per_passage=2)
    assert summary3["anchoring_gate"]["passages"]["t00"] == {
        "quota": 2, "anchored_pool": 3, "n": 2}
    assert summary3["conditions"]["A-FROM-CABLESE/R1"] \
        ["overall_raw"]["n"] == 2


def test_no_corrected_fields_in_merged_summary(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 2, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    for i in range(2):
        run_stub(td, src, tmp_path / "r", task_offset=i,
                 task_limit=1, slice_id=f"{i:04d}")
    merged = xm.merge_crossmodel_slices(str(tmp_path / "r"))
    for cond in xm.all_conditions():
        block = merged["conditions"][cond]
        assert "overall_corrected" not in block
        assert "bootstrap_corrected_95" not in block
        assert block["overall_raw"]["n"] == 4
    for pid in ("R1", "R2", "R3", "R4"):
        entry = merged["direction_a"][pid]
        assert set(entry) >= {"plain_accuracy", "cablese_accuracy",
                              "recovery_ratio"}
        assert "plain_corrected_accuracy" not in entry
        assert "cablese_corrected_accuracy" not in entry
        assert "raw_recovery_ratio" not in entry
    # gate accounting survives the merge (serial-equivalent)
    assert merged["anchoring_gate"]["rejected_unanchored"] == 0
    assert set(merged["anchoring_gate"]["passages"]) == {"t00", "t01"}


# ---------------- panels + fallback ----------------

def test_panels_frozen_match_spec():
    assert [p["id"] for p in xm.READER_PANEL] == ["R1", "R2", "R3", "R4"]
    assert xm.READER_PANEL[0]["model"] == "google/gemma-4-31b-it"
    assert xm.READER_PANEL[2]["model"] \
        == "nvidia/nemotron-3-super-120b-a12b"
    assert xm.READER_PANEL[3]["model"] == "google/gemma-4-26b-a4b-it"
    assert [p["id"] for p in xm.WRITER_PANEL] == ["W1", "W2", "W3"]
    assert xm.WRITER_PANEL[2]["model"] == "openai/gpt-5-mini"


def test_404_panel_dropped_with_note_no_substitution(tmp_path):
    """Frozen-panel fallback rule: an id that 404s at submit is dropped
    with a note in summary.json and is NOT substituted."""
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    dead_model = xm.READER_PANEL[3]["model"]  # unique id (R4; gemma ids shared with W1 after paid switch)
    base = xm.make_stub_client()

    def client(payload):
        if payload["model"] == dead_model:
            raise xm.ModelUnavailableError(
                f"OpenRouter returned HTTP 404 for model {dead_model!r}")
        return base(payload)

    summary = xm.run_crossmodel(
        client, xm.make_stub_client(), "glm-stub", 512,
        str(tmp_path / "r"), source_run=src, tasks_dir=str(td))
    drops = summary["panel_drops"]
    assert [d["panel"] for d in drops] == ["R4"]
    assert drops[0]["status"] == "HTTP 404 at submit"
    entry = summary["direction_a"]["R4"]
    assert entry["dropped"] is True
    assert "NOT substituted" in entry["drop_note"]
    assert entry["recovery_ratio"] is None
    # the surviving panels still ran (stub answers are never correct,
    # but n>0 proves their calls happened)
    assert summary["conditions"]["A-FROM-PLAIN/R1"]["overall_raw"]["n"] == 2
    assert summary["conditions"]["A-FROM-PLAIN/R4"]["overall_raw"]["n"] == 0
    assert summary["conditions"]["A-FROM-PLAIN/R2"]["overall_raw"]["n"] == 2
    assert summary["conditions"]["A-FROM-CABLESE/R3"]["overall_raw"]["n"] == 2


# ---------------- stub end-to-end structure ----------------

def test_stub_run_all_conditions_over_panels(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    summary = run_stub(td, src, tmp_path / "r")
    assert set(summary["conditions"]) == set(xm.all_conditions())
    for panel in xm.READER_PANEL:
        for c in xm.DIR_A_CONDS:
            n = summary["conditions"][f"{c}/{panel['id']}"] \
                ["overall_raw"]["n"]
            assert n == 2
    for panel in xm.WRITER_PANEL:
        for c in xm.DIR_B_ANSWER_CONDS:
            n = summary["conditions"][f"{c}/{panel['id']}"] \
                ["overall_raw"]["n"]
            assert n == 2


def test_every_miss_grader_v2_no_judge_records(tmp_path):
    """SPEC-VALIDATION Amendment 1: the truth path ends at grader v2 —
    stub answers are never correct, yet the ledger carries NO judge
    records of any kind."""
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    run_stub(td, src, tmp_path / "r")
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    assert not [r for r in recs if r["type"] == "judge"]
    assert not [r for r in recs if r.get("call") == "judge"]


def test_direction_b_uses_frozen_cablese_prompt(tmp_path):
    """Writer record prompts are the SAME frozen CABLESE_RULES prompt
    imported from ab.cablese (not a local copy)."""
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=1)
    src = make_source_run(tmp_path, td, q_per_passage=1)
    payloads = []
    base = xm.make_stub_client()

    def client(payload):
        payloads.append(payload)
        return base(payload)

    xm.run_crossmodel(client, base, "glm-stub", 512,
                      str(tmp_path / "r"), source_run=src,
                      tasks_dir=str(td))
    systems = [p["messages"][0]["content"] for p in payloads]
    assert cs.RECORD_CABLESE_SYSTEM in systems
    assert cs.RECORD_PLAIN_SYSTEM in systems
    assert cs.ANSWER_FROM_RECORD_USER.split("Record:")[0] in "".join(
        p["messages"][1]["content"] for p in payloads)


# ---------------- ratio math ----------------

def synthetic_material():
    answers, judges = {}, {}
    for panel in xm.READER_PANEL:
        pid = panel["id"]
        for cond, specs in (
            ("A-FROM-PLAIN", [True, True, False, False]),
            ("A-FROM-CABLESE", [True, False, False, False]),
        ):
            full = f"{cond}/{pid}"
            answers[full] = [cond_answer(full, "t0", i, ok)
                             for i, ok in enumerate(specs)]
            judges[full] = [judge_rec(full, "t0", i, "grader_wrong")
                            for i, ok in enumerate(specs) if not ok]
    for panel in xm.WRITER_PANEL:
        pid = panel["id"]
        for cond, specs in (
            ("GLM-FROM-PLAIN", [True, True, True, False]),
            ("GLM-FROM-CABLESE", [True, True, False, False]),
        ):
            full = f"{cond}/{pid}"
            answers[full] = [cond_answer(full, "t0", i, ok)
                             for i, ok in enumerate(specs)]
            judges[full] = [judge_rec(full, "t0", i, "grader_wrong")
                            for i, ok in enumerate(specs) if not ok]
    return answers, judges


def test_per_reader_ratio_raw_v2_headline():
    answers, judges = synthetic_material()
    s = xm.summarize_crossmodel(answers, judges, [], [])
    # raw (grader v2) is the single headline: 0.5 / 0.25
    assert s["direction_a"]["R1"]["plain_accuracy"] == 0.5
    assert s["direction_a"]["R1"]["cablese_accuracy"] == 0.25
    assert s["direction_a"]["R1"]["recovery_ratio"] == 0.5
    assert "plain_corrected_accuracy" not in s["direction_a"]["R1"]
    assert "raw_recovery_ratio" not in s["direction_a"]["R1"]
    assert (s["direction_a"]["R1"]["reference_glm_recovery_ratio"]
            == xm.REFERENCE_GLM_RECOVERY_RATIO == 0.997)


def test_ratio_zero_denominator_is_none():
    answers, judges = synthetic_material()
    for r in answers["A-FROM-PLAIN/R1"]:
        r["correct"] = False
    s = xm.summarize_crossmodel(answers, judges, [], [])
    assert s["direction_a"]["R1"]["plain_accuracy"] == 0.0
    assert s["direction_a"]["R1"]["recovery_ratio"] is None


def test_judge_diagnostics_isolation():
    """SPEC-VALIDATION: even if judge records exist in the stream, they
    only surface under diagnostics/ and never move the headline."""
    answers, _ = synthetic_material()
    full = "A-FROM-PLAIN/R1"
    baseline = xm.summarize_crossmodel({full: answers[full]}, {}, [], [])
    judged = xm.summarize_crossmodel(
        {full: answers[full]},
        {full: [judge_rec(full, "t0", i, "grader_wrong")
                for i, ok in enumerate([True, True, False, False])
                if not ok]}, [], [])
    b = baseline["conditions"][full]
    j = judged["conditions"][full]
    assert j["overall_raw"] == b["overall_raw"]  # headline unmoved
    assert "overall_corrected" not in j
    assert "bootstrap_corrected_95" not in j
    assert j["diagnostics"]["judge"]["misses_judged"] == 2
    assert j["diagnostics"]["judge"]["verdicts_all"]["grader_wrong"] == 2


def test_writer_savings_pct_positive_and_zero_guard(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=1)
    src = make_source_run(tmp_path, td, q_per_passage=1)
    summary = run_stub(td, src, tmp_path / "r")
    for pid, entry in summary["direction_b"].items():
        assert entry["savings_pct"] > 0  # stub cablese is shorter
        toks = entry["writer_record_tokens"]
        assert toks["cablese_approx_tokens_total"] \
            < toks["plain_approx_tokens_total"]

    def rec(cond, tokens):
        return {"condition": cond, "task_id": "t0",
                "approx_tokens": tokens}
    s = xm.summarize_crossmodel(
        {}, {}, [rec("R-PLAIN/W1", 0), rec("R-CABLESE/W1", 50)], [])
    assert s["direction_b"]["W1"]["savings_pct"] is None


# ---------------- McNemar reuse ----------------

def test_mcnemar_reused_from_cablese():
    answers, judges = synthetic_material()
    s = xm.summarize_crossmodel(answers, judges, [], [])
    # direction A pairing equals ab.cablese.mcnemar_exact on the same
    # paired hits — the crossmodel module must reuse, not re-implement
    hits_c = {(r["task_id"], r["q_index"]): r["correct"]
              for r in answers["A-FROM-CABLESE/R1"]}
    hits_p = {(r["task_id"], r["q_index"]): r["correct"]
              for r in answers["A-FROM-PLAIN/R1"]}
    assert s["mcnemar"]["A:R1"] == cs.mcnemar_exact(hits_c, hits_p)
    # cablese [T,F,F,F] vs plain [T,T,F,F]: one plain-only hit
    assert s["mcnemar"]["A:R1"]["c"] == 1
    assert set(s["mcnemar"]) == (
        {f"A:{p['id']}" for p in xm.READER_PANEL}
        | {f"B:{p['id']}" for p in xm.WRITER_PANEL})


# ---------------- slicing + merge serial equivalence ----------------

CORE = ["conditions", "mcnemar", "direction_a", "direction_b",
        "panel_drops", "panels", "source_run", "anchoring_gate"]


def test_serial_vs_sliced_serial_equivalence(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 4, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    serial = run_stub(td, src, tmp_path / "serial")
    par = tmp_path / "par"
    for i, (off, lim) in enumerate([(0, 2), (2, 1), (3, 1)]):
        s = run_stub(td, src, par, task_offset=off, task_limit=lim,
                     slice_id=f"{i:04d}")
        assert s["task_offset"] == off
        assert s["slice_id"] == f"{i:04d}"
    merged = xm.merge_crossmodel_slices(str(par))
    for key in CORE:
        assert merged[key] == serial[key], key
    assert merged["slices_merged"] == ["slice_0000.jsonl",
                                       "slice_0001.jsonl",
                                       "slice_0002.jsonl"]
    assert merged["passages"] == serial["passages"] == 4

    def answers(path):
        return sorted(
            (r["condition"], r["task_id"], r["q_index"])
            for r in map(json.loads, path.read_text().splitlines())
            if r.get("type") == "cond_answer")
    assert answers(tmp_path / "serial" / "ledger.jsonl") \
        == answers(par / "ledger.jsonl")


def test_slice_out_of_range_and_merge_errors(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 2)
    src = make_source_run(tmp_path, td)
    with pytest.raises(xm.CrossModelError):
        run_stub(td, src, tmp_path / "r", task_offset=5)
    with pytest.raises(xm.CrossModelError):
        run_stub(td, src, tmp_path / "r", task_offset=0, task_limit=0)
    with pytest.raises(xm.CrossModelError):
        xm.merge_crossmodel_slices(str(tmp_path / "nowhere"))


def test_slice_ledger_carries_model_ids_and_identity(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    run_stub(td, src, tmp_path / "r", task_offset=0, task_limit=1,
             slice_id="0042")
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "slices" / "slice_0042.jsonl")
            .read_text().splitlines()]
    assert recs
    for rec in recs:
        assert rec.get("git_sha")
        assert rec["run_id"].startswith("crossmodel_")
        assert rec["source_run"] == os.path.abspath(src)
    assert {r["slice_id"] for r in recs} == {"0042"}
    calls = [r for r in recs if r["type"] == "model_call"]
    assert all(r.get("model") for r in calls)
    models = {r["model"] for r in calls}
    assert "glm-stub" in models
    assert xm.READER_PANEL[0]["model"] in models
    assert xm.WRITER_PANEL[2]["model"] in models
    kinds = {r["call"] for r in calls}
    assert {"record_R-PLAIN", "record_R-CABLESE", "cond_qa"} <= kinds
    assert "judge" not in kinds


# ---------------- CLI (incl. the 14475 dispatch regression) ----------------

def test_cli_dispatch_passes_offset_and_slice_args_14475_crossmodel(
        tmp_path, monkeypatch):
    """REGRESSION (expB incident 14475 class): the CLI dispatch dropped
    the slice args before they reached the run function. Asserts the
    crossmodel dispatch passes --task-offset/--task-limit/--slice-id
    (and --passages, --source-run) through, by capturing the call."""
    captured = {}

    def fake_run(or_client, glm_client, glm_model, max_tokens,
                 results_dir, **kw):
        captured.update(kw)
        captured["results_dir"] = results_dir

    monkeypatch.setattr(xm, "run_crossmodel", fake_run)
    rc = xm.main(["crossmodel", "--stub",
                  "--source-run", str(tmp_path / "src"),
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
    assert captured["source_run"] == str(tmp_path / "src")


def test_cli_dispatch_requires_source_run(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        xm.main(["crossmodel", "--stub",
                 "--results-dir", str(tmp_path / "r")])
    assert exc.value.code == 2  # argparse: --source-run is required
    assert "--source-run" in capsys.readouterr().err


def test_cli_slice_end_to_end_and_merge(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 2, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    for i in range(2):
        assert xm.main(["crossmodel", "--stub",
                        "--source-run", src,
                        "--tasks-dir", str(td),
                        "--results-dir", str(tmp_path / "r"),
                        "--task-offset", str(i), "--task-limit", "1",
                        "--slice-id", f"{i:04d}"]) == 0
    sl = tmp_path / "r" / "slices" / "slice_0001.jsonl"
    su = tmp_path / "r" / "slices" / "slice_0001.summary.json"
    assert sl.exists() and su.exists()
    assert json.loads(su.read_text())["passages"] == 1
    assert not (tmp_path / "r" / "ledger.jsonl").exists()
    assert xm.main(["crossmodel-merge",
                    "--results-dir", str(tmp_path / "r")]) == 0
    merged = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert merged["passages"] == 2
    assert (tmp_path / "r" / "ledger.jsonl").exists()


def test_cli_merge_missing_dir_fails(tmp_path, capsys):
    rc = xm.main(["crossmodel-merge", "--results-dir",
                  str(tmp_path / "void")])
    assert rc == 2
    assert "no slices directory" in capsys.readouterr().err


# ---------------- summary shape ----------------

def test_summary_shape(tmp_path):
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    summary = run_stub(td, src, tmp_path / "r")
    for key in CORE + ["run_id", "git_sha", "passages"]:
        assert key in summary, key
    assert summary["source_run"] == os.path.abspath(src)
    assert [p["id"] for p in summary["panels"]["readers"]] == \
        ["R1", "R2", "R3", "R4"]
    assert [p["id"] for p in summary["panels"]["writers"]] == \
        ["W1", "W2", "W3"]
    for cond in xm.all_conditions():
        block = summary["conditions"][cond]
        for sub in ("overall_raw", "bootstrap_raw_95",
                    "per_type_raw", "per_register_raw", "diagnostics"):
            assert sub in block, (cond, sub)
        assert "overall_corrected" not in block
        assert "bootstrap_corrected_95" not in block
        assert block["overall_raw"]["n"] == 2
    for pid in ("R1", "R4"):
        assert summary["direction_a"][pid]["model"]
        assert "recovery_ratio" in summary["direction_a"][pid]
    for pid in ("W1", "W3"):
        entry = summary["direction_b"][pid]
        assert {"recovery_ratio", "savings_pct",
                "writer_record_tokens"} <= set(entry)


# ---------------- OpenRouter client contract ----------------

def _http_error(code):
    import email.message
    import io
    return urllib.error.HTTPError(
        "url", code, "boom", email.message.Message(),
        io.BytesIO(b'{"error": "detail"}'))


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_or_client_env_and_error_mapping(monkeypatch):
    client = xm.make_openrouter_client()
    monkeypatch.setenv("OR_API_KEY", "k")
    monkeypatch.setenv("OR_API_URL", "https://or.example/v1")
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.data)
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(xm.urllib.request, "urlopen", fake_urlopen)
    out = client({"model": "m1", "messages": []})
    assert out == {"ok": True}
    assert captured["url"] == "https://or.example/v1"
    assert captured["auth"] == "Bearer k"
    assert captured["body"]["model"] == "m1"  # model string per payload

    def boom(request, timeout=None):
        raise _http_error(404)

    monkeypatch.setattr(xm.urllib.request, "urlopen", boom)
    with pytest.raises(xm.ModelUnavailableError):
        client({"model": "m1"})

    def rate(request, timeout=None):
        raise _http_error(429)

    monkeypatch.setattr(xm.urllib.request, "urlopen", rate)
    with pytest.raises(TransientAPIError):
        client({"model": "m1"})


def test_or_client_default_url_and_missing_key(monkeypatch):
    monkeypatch.delenv("OR_API_URL", raising=False)
    monkeypatch.delenv("OR_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OR_API_KEY"):
        xm.make_openrouter_client()({"model": "m"})
    monkeypatch.setenv("OR_API_KEY", "k")
    monkeypatch.setattr(
        xm.urllib.request, "urlopen",
        lambda request, timeout=None: _FakeResponse({}))
    xm.make_openrouter_client()({"model": "m"})  # default URL accepted


# ---------------- canary 14659: reasoning-model content=None ----------------

def _empty_content_response(refusal=None):
    """HTTP-200 shape a reasoning-family panel returns when the answer
    lands in the reasoning channel: content=None (canary 14659)."""
    message = {"role": "assistant", "content": None}
    if refusal:
        message["refusal"] = refusal
    return {
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                  "total_tokens": 14},
    }


def test_or_payloads_carry_reasoning_disable(tmp_path, monkeypatch):
    """(a) Every OR-bound payload carries the unified OpenRouter
    reasoning-disable param — both at the raw OR client and through
    the run_crossmodel wrapper (stub path)."""
    monkeypatch.setenv("OR_API_KEY", "k")
    captured = []

    def fake_urlopen(request, timeout=None):
        captured.append(json.loads(request.data))
        return _FakeResponse(xm._stub_response("x", "u"))

    monkeypatch.setattr(xm.urllib.request, "urlopen", fake_urlopen)
    xm.make_openrouter_client()({"model": "m1", "messages": []})
    assert captured and all(
        p["reasoning"] == {"enabled": False} for p in captured)

    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=1)
    src = make_source_run(tmp_path, td, q_per_passage=1)
    payloads = []
    base = xm.make_stub_client()

    def capturing(payload):
        payloads.append(payload)
        return base(payload)

    xm.run_crossmodel(capturing, base, "glm-stub", 512,
                      str(tmp_path / "r"), source_run=src,
                      tasks_dir=str(td))
    assert payloads
    assert all(p.get("reasoning") == {"enabled": False}
               for p in payloads)


def test_none_content_retry_then_model_unusable_drop(tmp_path):
    """(b) content=None on both attempts -> exactly one immediate
    retry -> ModelUnusableError -> panel dropped with a note carrying
    the reason, NOT substituted; other panels unaffected."""
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    dead = xm.READER_PANEL[2]["model"]  # R3 nemotron: unique panel id
    calls = []
    base = xm.make_stub_client()

    def client(payload):
        calls.append(payload["model"])
        if payload["model"] == dead:
            return _empty_content_response()
        return base(payload)

    summary = xm.run_crossmodel(
        client, xm.make_stub_client(), "glm-stub", 512,
        str(tmp_path / "r"), source_run=src, tasks_dir=str(td))
    # each wrapper invocation = attempt + exactly one immediate retry;
    # the drop machinery tries both A-FROM conds before skipping the
    # panel (same shape as the 404 path), so 2 invocations = 4 calls
    assert calls.count(dead) == 4
    assert issubclass(xm.ModelUnusableError, xm.ModelUnavailableError)
    drops = summary["panel_drops"]
    assert [d["panel"] for d in drops] == ["R3"]
    assert "empty content (reasoning-model shape)" in drops[0]["status"]
    entry = summary["direction_a"]["R3"]
    assert entry["dropped"] is True
    assert "empty content (reasoning-model shape)" in entry["drop_note"]
    assert "NOT substituted" in entry["drop_note"]
    # dropped panel asked nothing; survivors ran at full n (no
    # substitution of another model into R3's slots)
    assert summary["conditions"]["A-FROM-PLAIN/R3"] \
        ["overall_raw"]["n"] == 0
    assert summary["conditions"]["A-FROM-CABLESE/R3"] \
        ["overall_raw"]["n"] == 0
    assert summary["conditions"]["A-FROM-PLAIN/R1"]["overall_raw"]["n"] == 2
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    events = [r for r in recs if r["type"] == "empty_content_retry"]
    assert len(events) == 2  # one per cond invocation, both exhausted
    assert all(e["model"] == dead and e["recovered"] is False
               for e in events)


def test_none_content_then_valid_on_retry_proceeds(tmp_path):
    """(c) content=None once, valid on the immediate retry -> run
    proceeds normally, ledger records the recovered retry, no drops."""
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    flaky = xm.READER_PANEL[2]["model"]  # R3, unique
    seen = {"n": 0}
    base = xm.make_stub_client()

    def client(payload):
        if payload["model"] == flaky:
            seen["n"] += 1
            if seen["n"] == 1:
                return _empty_content_response()
        return base(payload)

    summary = xm.run_crossmodel(
        client, xm.make_stub_client(), "glm-stub", 512,
        str(tmp_path / "r"), source_run=src, tasks_dir=str(td))
    assert seen["n"] >= 2
    assert summary["panel_drops"] == []
    assert summary["conditions"]["A-FROM-PLAIN/R3"] \
        ["overall_raw"]["n"] == 2
    assert summary["conditions"]["A-FROM-CABLESE/R3"] \
        ["overall_raw"]["n"] == 2
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    events = [r for r in recs if r["type"] == "empty_content_retry"]
    assert len(events) == 1
    assert events[0]["model"] == flaky
    assert events[0]["recovered"] is True


def test_refusal_raises_runtime_error_no_drop(tmp_path):
    """(d) A truthy refusal with empty content raises RuntimeError
    with the refusal text — no retry, no drop path."""
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=1)
    src = make_source_run(tmp_path, td, q_per_passage=1)
    dead = xm.READER_PANEL[2]["model"]
    base = xm.make_stub_client()

    def client(payload):
        if payload["model"] == dead:
            return _empty_content_response(
                refusal="content policy: prohibited topic")
        return base(payload)

    with pytest.raises(RuntimeError, match="prohibited topic"):
        xm.run_crossmodel(
            client, base, "glm-stub", 512, str(tmp_path / "r"),
            source_run=src, tasks_dir=str(td))
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    assert not [r for r in recs if r["type"] == "panel_drop"]
    assert not [r for r in recs if r["type"] == "empty_content_retry"]


# ---------------- canary 14669: adaptive reasoning-disable ----------------

MANDATORY_400_DETAIL = ('OpenRouter returned HTTP 400 for model '
                        '"vend/x-must-reason": {"error":{"message":'
                        '"Reasoning is mandatory for this endpoint '
                        'and cannot be disabled."}}')


class _ListLedger:
    def __init__(self):
        self.records = []

    def append(self, rec):
        self.records.append(rec)


@pytest.fixture
def clean_mandatory():
    xm._REASONING_MANDATORY.clear()
    yield
    xm._REASONING_MANDATORY.clear()


def test_400_mandatory_retries_without_param_and_learns(clean_mandatory):
    """(a) reasoning-mandatory 400 -> one retry of the SAME payload
    without the reasoning key; ledger gets reasoning_param_dropped;
    the second call for that model omits the param from the start
    (never sees a 400-shaped path again)."""
    seen = []
    base = xm.make_stub_client()

    def inner(payload):
        seen.append(payload)
        if "reasoning" in payload:
            raise xm.ORBadRequestError(MANDATORY_400_DETAIL)
        return base(payload)

    ledger = _ListLedger()
    wrapped = xm._wrap_or_client(inner, ledger)
    payload = {"model": "vend/x-must-reason", "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"}]}
    out = wrapped(payload)
    assert out["choices"][0]["message"]["content"]
    assert "reasoning" in seen[0] and "reasoning" not in seen[1]
    drops = [r for r in ledger.records
             if r["type"] == "reasoning_param_dropped"]
    assert drops == [{"type": "reasoning_param_dropped",
                      "model": "vend/x-must-reason"}]
    # learned: subsequent payloads omit the param from the start
    wrapped(payload)
    assert "reasoning" not in seen[-1]
    assert len([r for r in ledger.records
                if r["type"] == "reasoning_param_dropped"]) == 1


def test_400_unrelated_raises_no_adaptation(clean_mandatory):
    """(b) any other 400 propagates: no param drop, no
    retry-without-param, no learning."""
    calls = []

    def inner(payload):
        calls.append(payload)
        raise xm.ORBadRequestError(
            'OpenRouter returned HTTP 400 for model "m": '
            '{"error":{"message":"max_tokens must be <= 4096"}}')

    ledger = _ListLedger()
    wrapped = xm._wrap_or_client(inner, ledger)
    with pytest.raises(xm.ORBadRequestError):
        wrapped({"model": "m", "messages": []})
    assert len(calls) == 1
    assert "reasoning" in calls[0]  # param was NOT dropped for a retry
    assert not [r for r in ledger.records
                if r["type"] == "reasoning_param_dropped"]
    assert xm._REASONING_MANDATORY == set()


def test_mandatory_model_none_content_drops_panel(clean_mandatory,
                                                  tmp_path):
    """(c) a reasoning-mandatory model that then returns content=None
    twice lands in the hotfix-1 empty-content path honestly: retry ->
    ModelUnusableError -> panel drop note mentions empty content."""
    td, _ = make_tasks_dir(tmp_path, 1, q_per_passage=2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    dead = xm.READER_PANEL[2]["model"]  # R3, unique panel id
    base = xm.make_stub_client()

    def client(payload):
        if payload["model"] == dead:
            if "reasoning" in payload:
                raise xm.ORBadRequestError(MANDATORY_400_DETAIL)
            return _empty_content_response()
        return base(payload)

    summary = xm.run_crossmodel(
        client, xm.make_stub_client(), "glm-stub", 512,
        str(tmp_path / "r"), source_run=src, tasks_dir=str(td))
    drops = summary["panel_drops"]
    assert [d["panel"] for d in drops] == ["R3"]
    assert "empty content (reasoning-model shape)" in drops[0]["status"]
    assert "NOT substituted" in summary["direction_a"]["R3"]["drop_note"]
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    rpd = [r for r in recs if r["type"] == "reasoning_param_dropped"]
    assert rpd and all(r["model"] == dead for r in rpd)
    ecr = [r for r in recs if r["type"] == "empty_content_retry"]
    assert ecr and all(e["recovered"] is False and e["model"] == dead
                       for e in ecr)


# (d) default payloads still carry the reasoning-disable dict:
# test_or_payloads_carry_reasoning_disable above (raw OR client AND
# the run_crossmodel wrapper path, models not in _REASONING_MANDATORY).


# ---------------- slurm bundle ----------------

def test_sbatch_passes_bash_n():
    out = subprocess.run(["bash", "-n",
                          os.path.join(ROOT, "slurm",
                                       "crossmodel.sbatch")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_sbatch_contract():
    text = open(os.path.join(ROOT, "slurm", "crossmodel.sbatch"),
                encoding="utf-8").read()
    assert "#SBATCH --array=0-23" in text
    assert "#SBATCH --export=ALL" in text
    assert "TCB_CROSSMODEL_RUN_ID" in text
    assert "TCB_MAX_TOKENS" in text
    assert "${TCB_MAX_TOKENS:-4000}" in text
    assert "ab.crossmodel crossmodel-merge" in text
    assert "--task-offset" in text and "--task-limit" in text \
        and "--slice-id" in text
    assert "--source-run" in text
    assert "TCB_TREAT_401_TRANSIENT=1" in text
    assert "TCB_RETRY_ATTEMPTS=15" in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "sleep $(( (SLURM_ARRAY_TASK_ID * 7 + RANDOM) % 120 ))" in text
    # merge is slot-0-only
    assert '[ "${SLURM_ARRAY_TASK_ID:-0}" != "0" ] && exit 0' in text


def test_sbx_llm_block():
    sbx = json.load(open(os.path.join(ROOT, "slurm",
                                      "crossmodel.sbx.json"),
                         encoding="utf-8"))
    llm = sbx["llm"]
    # structure-based (sbx whitelist pins rot when the lane/model proxy
    # changes; broke once already: or -> ds-flash as price proxy)
    assert llm["model"] in ("or", "ds-flash", "glm53_flash"), llm["model"]
    assert llm["lanes"] == ["openrouter"]
    assert llm["requests"] > 0


def test_sbatch_tiling_formula_covers_50_exactly():
    """Same 50/24 tiling contract as cablese: contiguous, exact cover."""
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
