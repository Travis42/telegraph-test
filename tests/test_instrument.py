"""Hardened instrument: parse success/failure paths, verbatim raw
persistence, no code path returning a default on parse failure."""

import json

import pytest

from ab import instrument


def make_client(answer, prompt_tokens=10, completion_tokens=3):
    def client(payload):
        return {
            "choices": [
                {"index": 0,
                 "message": {"role": "assistant", "content": answer},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        }

    return client


# ---------------- parse_json_block ----------------

def test_parse_plain_array():
    obj, err = instrument.parse_json_block("[1, 2, 3]")
    assert err is None and obj == [1, 2, 3]


def test_parse_prose_wrrapped_and_fenced():
    text = 'Here you go:\n```json\n{"a": [1, 2], "b": "x"}\n```\nDone.'
    obj, err = instrument.parse_json_block(text)
    assert err is None and obj == {"a": [1, 2], "b": "x"}


def test_parse_nested_braces_inside_strings():
    obj, err = instrument.parse_json_block('sure ["a{b}c", "] not a close"]')
    assert err is None and obj == ["a{b}c", "] not a close"]


def test_parse_failure_returns_none_and_error_never_default():
    for bad in ("no json here", "", "[1, 2", "[unterminated", None):
        obj, err = instrument.parse_json_block(bad)
        assert obj is None, bad
        assert isinstance(err, str) and err, bad


def test_parse_failure_is_explicit_not_empty_result():
    obj, err = instrument.parse_json_block("the model rambled")
    assert obj is None
    assert "no JSON" in err


# ---------------- call_model ----------------

def test_call_model_returns_raw_verbatim_and_parsed():
    raw_text = 'Answer: ["q"] end'
    res = instrument.call_model(
        make_client(raw_text), "zai/glm-5.3-flash", "s", "u", 64,
        parse=instrument.parse_json_block,
    )
    assert res["raw"] == raw_text
    assert res["parsed"] == ["q"]
    assert res["parse_error"] is None
    assert res["usage"]["total_tokens"] == 13
    assert res["latency_ms"] >= 0


def test_call_model_parse_failure_is_flagged_not_defaulted():
    res = instrument.call_model(
        make_client("I cannot answer in JSON"), "m", "s", "u", 64,
        parse=instrument.parse_json_block,
    )
    assert res["raw"] == "I cannot answer in JSON"
    assert res["parsed"] is None
    assert res["parse_error"]


def test_call_model_missing_usage_raises_explicitly():
    def bad_client(payload):
        return {"choices": [{"message": {"content": "x"}}]}

    with pytest.raises(instrument.InstrumentError):
        instrument.call_model(bad_client, "m", "s", "u", 64)


def test_call_model_missing_content_raises_explicitly():
    def bad_client(payload):
        return {"choices": [], "usage": {"prompt_tokens": 1,
                                         "completion_tokens": 1,
                                         "total_tokens": 2}}

    with pytest.raises(instrument.InstrumentError):
        instrument.call_model(bad_client, "m", "s", "u", 64)


# ---------------- fact_check (FactCheck result type) ----------------

def test_fact_check_ok_shape():
    check = instrument.fact_check(
        make_client("[1, 3]"), "m", "ctx", ["f1", "f2", "f3"]
    )
    assert check["facts_total"] == 3
    assert check["facts_matched"] == [1, 3]
    assert check["parse_ok"] is True
    assert check["raw"] == "[1, 3]"
    assert instrument.recall_of(check) == pytest.approx(2 / 3)


def test_fact_check_zero_matched_distinct_from_parse_failure():
    zero = instrument.fact_check(make_client("[]"), "m", "c", ["f1"])
    assert zero["parse_ok"] is True
    assert zero["facts_matched"] == []
    assert instrument.recall_of(zero) == 0.0

    broken = instrument.fact_check(make_client("none stated"), "m", "c", ["f1"])
    assert broken["parse_ok"] is False
    assert broken["parse_error"]
    assert instrument.recall_of(broken) is None  # NOT silently 0.0


def test_fact_check_rejects_out_of_range_ids_as_parse_error():
    check = instrument.fact_check(make_client("[1, 9]"), "m", "c", ["f1", "f2"])
    assert check["parse_ok"] is False
    assert "out of range" in check["parse_error"]


def test_fact_check_rejects_non_integer_entries():
    check = instrument.fact_check(make_client('["1", 2]'), "m", "c", ["f1", "f2"])
    assert check["parse_ok"] is False
    assert "non-integer" in check["parse_error"]


def test_persisted_record_carries_raw(tmp_path):
    """Raw answers survive verbatim into a JSONL record (no truncation,
    no re-encoding)."""
    raw_text = '["multibyte ✓ ünïcode", 2]'
    res = instrument.call_model(
        make_client(raw_text), "m", "s", "u", 64,
        parse=instrument.parse_json_block,
    )
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps({"raw": res["raw"]}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["raw"] == raw_text
