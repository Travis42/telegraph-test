"""SPEC-RECORD-DECODE tests (offline; stub client everywhere, no
network). Covers the source-run guard, the ONLY-record decode prompt
assembly, the anchored-pool reuse (logged rejections, reduced-n
accounting), the ratio math incl. the zero-denominator case, the
source-verdict provenance (regrade artifact preferred, offline
recompute identical), tiling + merge serial equivalence, the CLI
dispatch regression for the expB 14475 class, ledger model recording,
the summary shape (grade_v2 only, no judge anywhere, token accounting
R-DECODED vs R-CABLESE vs R-PLAIN), and the slurm bundle."""

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
import ab.record_decode as rd  # noqa: E402
from tcb.tokencount import approx_count  # noqa: E402

TASK_PASSAGE = (
    "The brig Aurora sailed from Bristol on 4 March 1848 with a cargo "
    "of iron. Her master was Captain Elias Hoy, and she carried a crew "
    "of nineteen. After a storm off Ushant she put into Falmouth, and "
    "later resumed her voyage to Lisbon, arriving on 2 May before "
    "continuing to Oporto, which lay north of Lisbon along the coast."
)

R_PLAIN_TEXT = (
    "Plain record mentioning Captain Elias Hoy and Bristol 1848 in "
    "full prose, with the whole content of the passage retained in "
    "complete plain English sentences."
)
R_CABLESE_TEXT = "rcrd: aurora brstl 1848 iron hoy crw 19"


def make_tasks_dir(tmp_path, n=4):
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


def make_source_run(tmp_path, tasks_dir, q_per_passage=3):
    """Synthesize a cbl2-shaped source run: quiz_item + R-PLAIN +
    R-CABLESE records + A-FROM-PLAIN / A-FROM-CABLESE cond_answers per
    passage (the record layout ab/cablese.py writes and the regrade
    regrades). A-FROM-PLAIN answers are all v2-correct; A-FROM-CABLESE
    answers are all v2-wrong (ratio-math inputs)."""
    src = tmp_path / "cbl2_source"
    src.mkdir(exist_ok=True)
    lines = [{"type": "run_meta", "run_id": "cablese_fixture"}]
    for fn in sorted(os.listdir(tasks_dir)):
        task = json.loads((tasks_dir / fn).read_text())
        tid = task["id"]
        for qi in range(q_per_passage):
            lines.append({
                "type": "quiz_item", "task_id": tid,
                "q_index": qi, "q_type": "fact",
                "question": f"q{qi} about {tid}?",
                "answer": "Captain Elias Hoy",
            })
        for cond, text in (("R-PLAIN", R_PLAIN_TEXT),
                           ("R-CABLESE", R_CABLESE_TEXT)):
            lines.append({
                "type": "record", "condition": cond,
                "task_id": tid, "register": task["register"],
                "len_class": task["len_class"],
                "approx_tokens": approx_count(text), "text": text,
            })
        for cond, ok in (("A-FROM-PLAIN", True),
                         ("A-FROM-CABLESE", False)):
            for qi in range(q_per_passage):
                lines.append({
                    "type": "cond_answer", "condition": cond,
                    "task_id": tid, "register": task["register"],
                    "len_class": task["len_class"],
                    "q_index": qi, "q_type": "fact",
                    "question": f"q{qi} about {tid}?",
                    "expected": "Captain Elias Hoy",
                    "raw_answer": ("Captain Elias Hoy" if ok
                                   else "somebody else entirely"),
                    "correct": ok,
                })
    with open(src / "ledger.jsonl", "w", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(json.dumps(rec) + "\n")
    return str(src)


def run_stub(tasks_dir, source_run, results_dir, client=None, **kw):
    return rd.run_record_decode(
        client or rd.make_stub_client(), "glm-stub", 512,
        str(results_dir), source_run=source_run,
        tasks_dir=str(tasks_dir), **kw)


def decoded_answer(task_id, q_index, correct):
    return {"type": "cond_answer", "condition": "A-FROM-DECODED",
            "task_id": task_id, "register": "fact",
            "len_class": "medium", "q_index": q_index,
            "q_type": "fact", "question": f"q{q_index}",
            "expected": "Captain Elias Hoy", "raw_answer": "a",
            "correct": correct}


def decoded_record(task_id, tokens):
    return {"type": "record", "condition": "R-DECODED",
            "task_id": task_id, "approx_tokens": tokens}


# ---------------- source-run guard ----------------

def test_source_run_guard_missing_dir_and_ledger(tmp_path):
    with pytest.raises(rd.RecordDecodeError):
        run_stub(make_tasks_dir(tmp_path), str(tmp_path / "nowhere"),
                 tmp_path / "r")


def test_source_run_guard_incomplete_records(tmp_path, capsys):
    td = make_tasks_dir(tmp_path, 2)
    src = make_source_run(tmp_path, td)
    path = os.path.join(src, "ledger.jsonl")
    lines = [json.loads(l) for l in open(path)]
    kept = [r for r in lines
            if not (r.get("task_id") == "t01"
                    and r.get("condition") == "R-CABLESE")]
    with open(path, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    with pytest.raises(rd.RecordDecodeError, match="R-CABLESE"):
        run_stub(td, src, tmp_path / "r")
    rc = rd.main(["record-decode", "--stub", "--source-run", src,
                  "--results-dir", str(tmp_path / "r"),
                  "--tasks-dir", str(td)])
    assert rc == 2
    assert "R-CABLESE" in capsys.readouterr().err


def test_source_run_guard_missing_task_in_bank(tmp_path):
    td = make_tasks_dir(tmp_path, 3)
    src = make_source_run(tmp_path, td)
    os.remove(str(td / "t02.json"))
    with pytest.raises(rd.RecordDecodeError, match="tasks bank"):
        run_stub(td, src, tmp_path / "r")


# ---------------- ONLY-record decode prompt assembly ----------------

def test_record_decode_prompt_is_only_record(tmp_path):
    """The expansion prompt carries the source R-CABLESE record and
    NEVER the source passage; the answering prompts use cablese's
    ANSWER_FROM_RECORD_USER with record=R-DECODED."""
    td = make_tasks_dir(tmp_path, 1)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    payloads = []
    base = rd.make_stub_client()

    def client(payload):
        payloads.append(payload)
        return base(payload)

    run_stub(td, src, tmp_path / "r", client=client)
    decodes = [p for p in payloads
               if p["messages"][0]["content"]
               == rd.RECORD_DECODE_SYSTEM]
    assert len(decodes) == 1
    user = decodes[0]["messages"][1]["content"]
    assert user == rd.RECORD_DECODE_USER.format(record=R_CABLESE_TEXT)
    assert TASK_PASSAGE.split()[0] not in user  # ONLY the record
    answers = [p for p in payloads
               if "Answer the question using only the record"
               in p["messages"][1]["content"]]
    assert len(answers) == 2
    assert all(p["messages"][0]["content"] == cs.QA_SYSTEM
               or p["messages"][0]["content"] for p in answers)
    for p in answers:
        assert p["messages"][1]["content"].startswith(
            cs.ANSWER_FROM_RECORD_USER.split("Record:")[0])
    # the expanded stub record (not the cablese record, not the
    # passage) is what the answerer sees
    assert all("expanded back into" in p["messages"][1]["content"]
               for p in answers)
    assert not any(TASK_PASSAGE[:40] in p["messages"][1]["content"]
                   for p in payloads)


def test_decode_prompts_frozen_mirror_cablese_semantics():
    assert "given " in rd.RECORD_DECODE_USER
    assert "ONLY this record" in rd.RECORD_DECODE_USER
    assert "do not add information" in rd.RECORD_DECODE_USER
    assert rd.RECORD_DECODE_SYSTEM == cs.DECODE_SYSTEM  # mirror


# ---------------- anchored-pool reuse ----------------

def test_anchored_pool_reuse_with_logged_rejection(tmp_path):
    """Reused cbl2 items that are not v2-anchored to the source passage
    are rejected + logged (ledger record + sidecar), never asked; the
    quota is drawn from the anchored pool only."""
    td = make_tasks_dir(tmp_path, 1)
    src = make_source_run(tmp_path, td, q_per_passage=3)
    path = os.path.join(src, "ledger.jsonl")
    lines = [json.loads(l) for l in open(path)]
    for rec in lines:
        if rec.get("type") == "quiz_item" and rec["q_index"] == 1:
            rec["answer"] = "Zaphod Beeblebrox"  # not passage-anchored
    with open(path, "w") as fh:
        for rec in lines:
            fh.write(json.dumps(rec) + "\n")
    summary = run_stub(td, src, tmp_path / "r")
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    rejects = [r for r in recs if r["type"] == "unanchored_reject"]
    assert len(rejects) == 1
    assert rejects[0]["answer"] == "Zaphod Beeblebrox"
    assert rejects[0]["anchor_score"] < 0.8
    sidecar = json.loads(
        (tmp_path / "r" / "unanchored_source_items.json").read_text())
    assert [r["answer"] for r in sidecar] == ["Zaphod Beeblebrox"]
    asked = {r["q_index"] for r in recs
             if r["type"] == "cond_answer"}
    assert asked == {0, 2}
    gate = summary["anchoring_gate"]
    assert gate["rejected_unanchored"] == 1
    assert gate["passages"]["t00"]["anchored_pool"] == 2
    assert gate["passages"]["t00"]["n"] == 2
    assert summary["conditions"]["A-FROM-DECODED"]["overall_raw"]["n"] == 2


def test_reduced_quota_passage_accounting(tmp_path):
    """Pool smaller than quota -> reduced n recorded, run not failed."""
    td = make_tasks_dir(tmp_path, 1)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    summary = run_stub(td, src, tmp_path / "r")  # quota 24 > pool 2
    assert summary["anchoring_gate"]["passages"]["t00"] == {
        "quota": rd.SOURCE_QUESTIONS_PER_PASSAGE,
        "anchored_pool": 2, "n": 2}
    assert summary["conditions"]["A-FROM-DECODED"]["overall_raw"]["n"] == 2


# ---------------- ratio math ----------------

def synthetic_summary(decoded_correct, plain_correct):
    answers = [decoded_answer("t0", i, ok)
               for i, ok in enumerate(decoded_correct)]
    source_hits = {
        "A-FROM-PLAIN": {("t0", i): ok
                         for i, ok in enumerate(plain_correct)},
        "A-FROM-CABLESE": {("t0", i): False
                           for i in range(len(plain_correct))},
    }
    return rd.summarize_record_decode(
        answers, [decoded_record("t0", 100)],
        source_hits, {("t0", "R-PLAIN"): 100,
                      ("t0", "R-CABLESE"): 50}, {"t0"})


def test_decoded_recovery_ratio_and_mcnemar_math():
    s = synthetic_summary([True, True, False, False],
                          [True, True, True, False])
    # decoded 0.5 / plain 0.75; cablese 0.0
    assert s["decoded_recovery_ratio"] == round(0.5 / 0.75, 6)
    # cablese accuracy 0.0 -> record-comparison ratio undefined
    assert s["decode_ratio_rec"] is None
    assert s["source_accuracies_v2"]["A-FROM-PLAIN"] == 0.75
    assert s["source_accuracies_v2"]["A-FROM-CABLESE"] == 0.0
    # decoded [T,T,F,F] vs plain [T,T,T,F]: one plain-only hit (c=1)
    assert s["mcnemar"]["A-FROM-DECODED_vs_A-FROM-PLAIN"]["c"] == 1
    assert s["mcnemar"]["A-FROM-DECODED_vs_A-FROM-PLAIN"] == (
        cs.mcnemar_exact(
            {("t0", i): ok for i, ok in enumerate([True, True, False, False])},
            {("t0", i): ok for i, ok in enumerate([True, True, True, False])}))


def test_ratio_zero_denominator_is_none():
    s = synthetic_summary([False, False], [False, False])
    assert s["decoded_recovery_ratio"] is None  # 0.0/0.0 denominator
    s2 = synthetic_summary([True, True], [False, False])
    assert s2["decoded_recovery_ratio"] is None


def test_token_accounting_decoded_vs_cablese_vs_plain():
    s = synthetic_summary([True], [True])
    toks = s["record_tokens"]
    assert toks["R-DECODED"]["mean_approx_tokens"] == 100
    assert toks["R-CABLESE"]["mean_approx_tokens"] == 50
    assert toks["R-PLAIN"]["mean_approx_tokens"] == 100
    assert toks["decoded_vs_plain_ratio"] == 1.0  # ~1x R-PLAIN expected
    assert toks["decoded_vs_cablese_ratio"] == 2.0


# ---------------- source-verdict provenance ----------------

def test_source_v2_hits_prefers_regrade_artifact(tmp_path):
    """The regrade_v2 per-record verdicts are the v2 truth of record:
    when present they win over a ledger recompute."""
    td = make_tasks_dir(tmp_path, 1)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    from ab.run_v21 import load_tasks
    bank = {t["id"]: t for t in load_tasks(str(td))}
    hits = rd.load_source_v2_hits(src, tasks_by_id=bank)  # recompute path
    assert all(v for v in hits["A-FROM-PLAIN"].values())
    assert not any(v for v in hits["A-FROM-CABLESE"].values())
    out = os.path.join(src, "regrade_v2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "regraded_cond_answers.jsonl"),
              "w", encoding="utf-8") as fh:
        for i in range(2):
            fh.write(json.dumps({
                "type": "cond_answer", "condition": "A-FROM-PLAIN",
                "task_id": "t00", "q_index": i, "correct": False,
            }) + "\n")
    hits2 = rd.load_source_v2_hits(src, tasks_by_id=bank)
    assert hits2["A-FROM-PLAIN"] == {("t00", 0): False, ("t00", 1): False}
    summary = run_stub(td, src, tmp_path / "r")
    assert summary["source_accuracies_v2"]["A-FROM-PLAIN"] == 0.0
    assert summary["decoded_recovery_ratio"] is None  # 0 denominator


# ---------------- stub run + summary shape ----------------

def test_stub_run_summary_shape_and_no_judge(tmp_path):
    td = make_tasks_dir(tmp_path, 2, )
    src = make_source_run(tmp_path, td, q_per_passage=2)
    summary = run_stub(td, src, tmp_path / "r")
    assert summary["passages"] == 2
    block = summary["conditions"]["A-FROM-DECODED"]
    assert block["overall_raw"]["n"] == 4
    assert block["overall_raw"]["accuracy"] == 0.0  # stub never matches
    assert "overall_corrected" not in block
    assert set(summary["mcnemar"]) == {"A-FROM-DECODED_vs_A-FROM-PLAIN"}
    # stub expansion is longer than the cablese record, ~ R-PLAIN scale
    toks = summary["record_tokens"]
    assert toks["R-DECODED"]["mean_approx_tokens"] \
        > toks["R-CABLESE"]["mean_approx_tokens"]
    assert toks["decoded_vs_cablese_ratio"] > 1.0
    assert toks["decoded_vs_plain_ratio"] >= 1.0
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    assert not [r for r in recs if r["type"] == "judge"]
    assert not [r for r in recs if r.get("call") == "judge"]
    kinds = {r["call"] for r in recs if r["type"] == "model_call"}
    assert kinds == {"record_decode", "cond_qa"}
    assert summary["source_run"] == os.path.abspath(src)


def test_ledger_records_model_ids_and_identity(tmp_path):
    td = make_tasks_dir(tmp_path, 1)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    run_stub(td, src, tmp_path / "r", task_offset=0, task_limit=1,
             slice_id="0042")
    recs = [json.loads(l) for l in
            (tmp_path / "r" / "slices" / "slice_0042.jsonl")
            .read_text().splitlines()]
    assert recs
    for rec in recs:
        assert rec.get("git_sha")
        assert rec["run_id"].startswith("record_decode_")
        assert rec["source_run"] == os.path.abspath(src)
    assert {r["slice_id"] for r in recs} == {"0042"}
    calls = [r for r in recs if r["type"] == "model_call"]
    assert calls and all(r["model"] == "glm-stub" for r in calls)


# ---------------- slicing + merge serial equivalence ----------------

CORE = ["conditions", "mcnemar", "decoded_recovery_ratio",
        "decode_ratio_rec", "source_accuracies_v2", "record_tokens",
        "anchoring_gate", "source_run"]


def test_serial_vs_sliced_serial_equivalence(tmp_path):
    td = make_tasks_dir(tmp_path, 4)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    serial = run_stub(td, src, tmp_path / "serial")
    par = tmp_path / "par"
    for i, (off, lim) in enumerate([(0, 2), (2, 1), (3, 1)]):
        s = run_stub(td, src, par, task_offset=off, task_limit=lim,
                     slice_id=f"{i:04d}")
        assert s["task_offset"] == off
        assert s["slice_id"] == f"{i:04d}"
    merged = rd.merge_record_decode_slices(str(par), tasks_dir=str(td))
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
    td = make_tasks_dir(tmp_path, 2)
    src = make_source_run(tmp_path, td)
    with pytest.raises(rd.RecordDecodeError):
        run_stub(td, src, tmp_path / "r", task_offset=5)
    with pytest.raises(rd.RecordDecodeError):
        run_stub(td, src, tmp_path / "r", task_offset=0, task_limit=0)
    with pytest.raises(rd.RecordDecodeError):
        rd.merge_record_decode_slices(str(tmp_path / "nowhere"))


# ---------------- CLI (incl. the 14475 dispatch regression) ----------------

def test_cli_dispatch_passes_offset_and_slice_args_14475_record_decode(
        tmp_path, monkeypatch):
    """REGRESSION (expB incident 14475 class): the CLI dispatch dropped
    the slice args before they reached the run function. Asserts the
    record-decode dispatch passes --task-offset/--task-limit/--slice-id
    (and --passages, --source-run) through, by capturing the call."""
    captured = {}

    def fake_run(client, model, max_tokens, results_dir, **kw):
        captured.update(kw)
        captured["results_dir"] = results_dir

    monkeypatch.setattr(rd, "run_record_decode", fake_run)
    rc = rd.main(["record-decode", "--stub",
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


def test_cli_source_run_default_is_cbl2():
    """--source-run defaults to results/cablese_cbl2 (spec)."""
    import ab.record_decode as mod
    assert mod.DEFAULT_SOURCE_RUN == os.path.join(
        mod._ROOT, "results", "cablese_cbl2")


def test_cli_slice_end_to_end_and_merge(tmp_path):
    td = make_tasks_dir(tmp_path, 2)
    src = make_source_run(tmp_path, td, q_per_passage=2)
    for i in range(2):
        assert rd.main(["record-decode", "--stub",
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
    assert rd.main(["record-decode-merge",
                    "--tasks-dir", str(td),
                    "--results-dir", str(tmp_path / "r")]) == 0
    merged = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert merged["passages"] == 2
    assert (tmp_path / "r" / "ledger.jsonl").exists()


def test_cli_merge_missing_dir_fails(tmp_path, capsys):
    rc = rd.main(["record-decode-merge", "--results-dir",
                  str(tmp_path / "void")])
    assert rc == 2
    assert "no slices directory" in capsys.readouterr().err


# ---------------- slurm bundle ----------------

def test_sbatch_passes_bash_n():
    out = subprocess.run(["bash", "-n",
                          os.path.join(ROOT, "slurm",
                                       "record_decode.sbatch")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_sbatch_contract():
    text = open(os.path.join(ROOT, "slurm", "record_decode.sbatch"),
                encoding="utf-8").read()
    assert "#SBATCH --export=ALL" in text
    assert "TCB_RECORD_DECODE_RUN_ID" in text
    assert "TCB_MAX_TOKENS" in text
    assert "${TCB_MAX_TOKENS:-4000}" in text
    assert "--source-run" in text
    assert "results/cablese_cbl2" in text  # default source run
    assert "--task-offset" in text and "--task-limit" in text \
        and "--slice-id" in text
    assert "TCB_TREAT_401_TRANSIENT=1" in text
    assert "TCB_RETRY_ATTEMPTS=15" in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "ab.record_decode record-decode-merge" in text
    # merge is slot-0-only
    assert '[ "${SLURM_ARRAY_TASK_ID:-0}" != "0" ] && exit 0' in text


def test_sbx_llm_block():
    sbx = json.load(open(os.path.join(ROOT, "slurm",
                                      "record_decode.sbx.json"),
                         encoding="utf-8"))
    llm = sbx["llm"]
    assert llm["model"] == "glm53_flash"
    assert llm["lanes"] == ["zai-max"]
    assert llm["requests"] == 800
