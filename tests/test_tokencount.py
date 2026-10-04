"""Approximate counter: monotonic sanity + approximation labeling."""

from tcb import tokencount


def test_labeled_approx():
    out = tokencount.approx_count_labeled("hello world")
    assert out["approx"] is True
    assert isinstance(out["count"], int) and out["count"] > 0


def test_empty():
    assert tokencount.approx_count("") == 0
    assert tokencount.approx_count("   \n ") == 0


def test_monotonic_under_concatenation():
    a = "the weather was quiet"
    b = "and the operator answered"
    c = a + " " + b
    assert tokencount.approx_count(c) >= tokencount.approx_count(a)
    assert tokencount.approx_count(c) >= tokencount.approx_count(b)


def test_monotonic_under_prefixing():
    for text in ("message", "the amount was paid", "73 best regards"):
        assert tokencount.approx_count("x " + text) >= tokencount.approx_count(text)


def test_longer_word_costs_not_less():
    assert tokencount.approx_count("internationalization") > tokencount.approx_count("a")


def test_punctuation_counts():
    assert tokencount.approx_count("word.") > tokencount.approx_count("word")
