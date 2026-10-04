#!/usr/bin/env python3
"""Wave 4A corpus expansion (SPEC-WAVE4A §1).

Reuses the V2.1 probe chain and frame extractor from ab/envelope_corpus.py
(read-only CLI export first, parameterized session count; target 50
sessions, accept whatever exists). The ONLY new behavior is the split:
SESSION-LEVEL train/test split (even-index sessions -> TRAIN, odd-index ->
TEST) so that frame recurrence never crosses the split (zero leakage,
unlike V2.1's frame-level half split).

Output: results/wave4a/corpus_train.jsonl, corpus_test.jsonl,
corpus_stats.json.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ab import envelope_corpus  # noqa: E402

DEFAULT_SESSIONS = 50
WAVE4A_DIR = os.path.join(_ROOT, "results", "wave4a")


def session_order(frames):
    """Session ids in first-occurrence order (deterministic for a given
    corpus file order)."""
    seen = []
    for frame in frames:
        session = frame["session"]
        if session not in seen:
            seen.append(session)
    return seen


def split_by_session(frames):
    """Even-index sessions -> TRAIN, odd-index -> TEST (session-level
    split, zero cross-leakage of frame recurrence)."""
    order = session_order(frames)
    train_sessions = {s for i, s in enumerate(order) if i % 2 == 0}
    train = [f for f in frames if f["session"] in train_sessions]
    test = [f for f in frames if f["session"] not in train_sessions]
    meta = {
        "split": "session_level_even_odd",
        "sessions_total": len(order),
        "train_sessions": sorted(train_sessions),
        "test_sessions": sorted(s for s in order if s not in train_sessions),
        "train_frames": len(train),
        "test_frames": len(test),
        "zero_leakage": not (train_sessions & set(order) - train_sessions),
    }
    meta["zero_leakage"] = not (
        {f["session"] for f in train} & {f["session"] for f in test}
    )
    return train, test, meta


def build_and_write(sessions=DEFAULT_SESSIONS, out_dir=WAVE4A_DIR):
    """Probe the V2.1 source chain, extract frames, split by session,
    write corpus + stats. Returns (train, test, meta, trail)."""
    frames, trail = envelope_corpus.build_corpus(sessions)
    train, test, meta = split_by_session(frames)
    os.makedirs(out_dir, exist_ok=True)
    for name, subset in (
        ("corpus_train.jsonl", train),
        ("corpus_test.jsonl", test),
    ):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            for frame in subset:
                fh.write(
                    json.dumps(frame, sort_keys=True, ensure_ascii=False) + "\n"
                )
    meta = dict(meta)
    meta.update(
        {
            "type": "wave4a_corpus",
            "spec": "SPEC-WAVE4A.md",
            "target_sessions": sessions,
            "frames_total": len(frames),
            "source_trail": trail,
        }
    )
    stats_path = os.path.join(out_dir, "corpus_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    return train, test, meta, trail


def load_jsonl(path):
    frames = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def corpus_paths(out_dir=WAVE4A_DIR):
    return (
        os.path.join(out_dir, "corpus_train.jsonl"),
        os.path.join(out_dir, "corpus_test.jsonl"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--results-dir", default=WAVE4A_DIR)
    args = parser.parse_args(argv)
    train, test, meta, _ = build_and_write(args.sessions, args.results_dir)
    train_path, test_path = corpus_paths(args.results_dir)
    print(f"train: {len(train)} frames / {len(meta['train_sessions'])} sessions -> {train_path}")
    print(f"test:  {len(test)} frames / {len(meta['test_sessions'])} sessions -> {test_path}")
    print(f"zero_leakage={meta['zero_leakage']}")
    return 0 if meta["zero_leakage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
