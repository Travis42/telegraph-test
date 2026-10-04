#!/usr/bin/env python3
"""Wave 4A lossless dictionary builder v2 (SPEC-WAVE4A §2).

Stage 1 (keep): whole-frame exact match (V2.1 behavior) — every unique
TRAIN frame text gets a short code.

Stage 2 (new): segment mining on TRAIN —
  - JSON-ish tokenizer (strings / punctuation / numbers / literals) over
    the canonical frame texts; spans of >= 2 tokens become candidates;
    non-JSON texts fall back to plain recurring-substring mining;
  - candidates: length >= 12 chars OR >= 3 JSON fields, TRAIN frequency
    (frames containing) >= 5;
  - greedy selection longest-first (length desc, then lexicographic —
    deterministic), non-overlapping per text; occurrences come from a
    ONE-pass multi-pattern scan (Aho-Corasick automaton, stdlib only)
    over the unique texts — no per-candidate corpus rescans;
  - collision-free, tokenizer-friendly short ASCII codes:
    frames `za0001`..., segments `zc01`..`zc99` then `zd001`...;
  - economy rule: a segment enters only if
    (TEST-projected frequency x per-occurrence token saving) >
    amortized legend cost share (the tokens of its own legend line).

Legend: one text block (code -> segment, plus the frame table), emitted
once; its size is counted in all net-savings math (by the caller).

HARD PROPERTY: decode(encode(x)) == x for EVERY frame, verified at build
time; ANY violation, duplicate code, or code collision aborts the build
(DictionaryBuildError — no partial dictionaries are ever returned).

Operational guards: a heartbeat line on stderr every 10s per stage
(silence must be visible), and a wall-clock budget (--max-build-seconds,
default 600) whose expiry aborts the build with a clear error.
"""

import bisect
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tcb.tokencount import approx_count  # noqa: E402

MIN_CHARS = 12
MIN_FIELDS = 3
MIN_FREQ = 5
MAX_CANDIDATES = 20000
MAX_SPAN_TOKENS = 10
MAX_SUBSTR_LEN = 64
MAX_SEGMENT_CODES = 99 + 999  # zc01..zc99, zd001..zd999
MAX_FRAME_CODES = 9999
MAX_BUILD_SECONDS = 600
HEARTBEAT_INTERVAL = 10.0

_TOKEN_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"'                # JSON strings
    r"|[{}\[\]:,]"                      # structural punctuation
    r"|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"  # numbers
    r"|true|false|null"                 # literals
)


class DictionaryBuildError(Exception):
    """Losslessness / uniqueness violation — the build aborts."""


# --------------------------------------------------------------------------
# tokenization


def tokenize(text):
    """JSON-ish tokens covering the WHOLE text; None when the text has
    gaps (bare characters outside JSON syntax => non-JSON text)."""
    tokens = []
    pos = 0
    for match in _TOKEN_RE.finditer(text):
        if match.start() != pos:
            return None
        tokens.append(match.group(0))
        pos = match.end()
    if pos != len(text):
        return None
    return tokens


def field_count(tokens):
    return sum(1 for t in tokens if t == ":")


# --------------------------------------------------------------------------
# code allocation (collision-free by construction, checked anyway)


class CodeAllocator:
    def __init__(self):
        self.used = set()
        self.frame_i = 0
        self.seg_i = 0

    def _issue(self, code):
        if code in self.used:
            raise DictionaryBuildError(f"code collision on {code!r}")
        self.used.add(code)
        return code

    def next_frame_code(self):
        if self.frame_i >= MAX_FRAME_CODES:
            raise DictionaryBuildError("frame code space exhausted")
        self.frame_i += 1
        return self._issue(f"za{self.frame_i:04d}")

    def peek_segment_code(self):
        i = self.seg_i + 1
        if i <= 99:
            return f"zc{i:02d}"
        if i <= 99 + 999:
            return f"zd{i - 99:03d}"
        raise DictionaryBuildError("segment code space exhausted")

    def commit_segment_code(self):
        if self.seg_i >= MAX_SEGMENT_CODES:
            raise DictionaryBuildError("segment code space exhausted")
        self.seg_i += 1
        code = self.peek_segment_code()
        return self._issue(code)


# --------------------------------------------------------------------------
# candidate mining


def _token_span_candidates(texts, min_chars):
    """Candidates from JSON-tokenizable texts: joins of contiguous token
    spans (2..MAX_SPAN_TOKENS tokens). Returns {segment: set(text_idx)}."""
    containment = {}
    for idx, text in enumerate(texts):
        tokens = tokenize(text)
        if tokens is None:
            continue
        n = len(tokens)
        for i in range(n):
            for span in range(2, min(MAX_SPAN_TOKENS, n - i) + 1):
                seg = "".join(tokens[i : i + span])
                if len(seg) < min_chars and field_count(tokens[i : i + span]) < MIN_FIELDS:
                    continue
                containment.setdefault(seg, set()).add(idx)
    return containment


def _substring_candidates(texts, min_chars):
    """Bounded fallback for non-JSON texts: only substrings that occur as
    the PREFIX or SUFFIX of a text (shared boilerplate lives at frame
    edges) are candidates — O(texts x MAX_SUBSTR_LEN) window generations,
    never all substrings."""
    containment = {}
    for idx, text in enumerate(texts):
        if tokenize(text) is not None:
            continue
        capped = text[:4000]
        seen_here = set()
        for length in range(min_chars, min(MAX_SUBSTR_LEN, len(capped)) + 1):
            seen_here.add(capped[:length])
            seen_here.add(capped[-length:])
        for seg in seen_here:
            containment.setdefault(seg, set()).add(idx)
    return containment


def _find_occurrences(text, seg):
    """All start offsets of seg in text (non-step-1 scanning is fine;
    occurrences are checked for overlap against occupied intervals).

    Kept ONLY for reference/testing — the build itself never calls this
    (it uses the one-pass _AhoCorasick occurrence index below)."""
    out = []
    start = text.find(seg)
    while start != -1:
        out.append(start)
        start = text.find(seg, start + 1)
    return out


class _AhoCorasick:
    """Minimal multi-pattern occurrence scanner (stdlib only).

    build: trie + BFS fail links (with output inheritance). scan: one
    pass over each text reports EVERY (overlapping) occurrence of every
    pattern — the single pass that replaces per-candidate rescans."""

    def __init__(self, patterns):
        self.lens = [len(p) for p in patterns]
        self.goto = [{}]          # node -> {char: node}
        self.fail = [0]
        self.out = [()]           # node -> tuple(pattern indexes)
        for pi, pat in enumerate(patterns):
            node = 0
            for ch in pat:
                nxt = self.goto[node].get(ch)
                if nxt is None:
                    nxt = len(self.goto)
                    self.goto.append({})
                    self.fail.append(0)
                    self.out.append(())
                    self.goto[node][ch] = nxt
                node = nxt
            self.out[node] = self.out[node] + (pi,)
        # BFS fail links; outputs inherited along fail edges
        from collections import deque
        queue = deque(self.goto[0].values())
        while queue:
            u = queue.popleft()
            for ch, v in self.goto[u].items():
                queue.append(v)
                f = self.fail[u]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[v] = self.goto[f].get(ch, 0)
                if self.fail[v] == v:  # depth-1 self-reference guard
                    self.fail[v] = 0
                self.out[v] = self.out[v] + self.out[self.fail[v]]


def _scan_occurrences(texts, patterns, heartbeat, deadline):
    """ONE pass over `texts` with the automaton over `patterns`.
    Returns, per pattern index, a list of packed ints
    (text_idx << 32 | start_offset) — every overlapping occurrence."""
    ac = _AhoCorasick(patterns)
    goto, fail, out, lens = ac.goto, ac.fail, ac.out, ac.lens
    root = goto[0]
    occ = [[] for _ in patterns]
    total = 0
    for idx, text in enumerate(texts):
        heartbeat.tick("occurrence-index", frames=idx + 1,
                       patterns=len(patterns), occurrences=total)
        _check_deadline(deadline, "occurrence-index")
        node = 0
        for pos, ch in enumerate(text):
            nxt = goto[node].get(ch)
            if nxt is None:
                while node:
                    node = fail[node]
                    nxt = goto[node].get(ch)
                    if nxt is not None:
                        break
                if nxt is None:
                    nxt = root.get(ch)
            if nxt is not None:
                node = nxt
                hits = out[node]
                if hits:
                    end_pos = pos + 1
                    for pi in hits:
                        occ[pi].append((idx << 32) | (end_pos - lens[pi]))
                    total += len(hits)
    return occ


class _SpanTable:
    """Per-text occupied intervals, kept sorted and disjoint —
    O(log n) overlap checks via bisect (intervals never overlap each
    other, so only the neighbors of the probe can collide)."""

    __slots__ = ("table",)

    def __init__(self):
        self.table = {}  # text_idx -> ([starts], [ends]) sorted, disjoint

    def free(self, idx, start, end):
        entry = self.table.get(idx)
        if entry is None:
            return True
        starts, ends = entry
        i = bisect.bisect_right(starts, start)
        if i and ends[i - 1] > start:
            return False
        if i < len(starts) and starts[i] < end:
            return False
        return True

    def add(self, idx, start, end):
        starts, ends = self.table.get(idx) or ([], [])
        i = bisect.bisect_right(starts, start)
        starts.insert(i, start)
        ends.insert(i, end)
        self.table[idx] = (starts, ends)


class _Heartbeat:
    """Progress line to stderr every HEARTBEAT_INTERVAL seconds while a
    build is running — silence must be visible (test-monitors rule)."""

    def __init__(self, interval=HEARTBEAT_INTERVAL, stream=None):
        self.interval = interval
        self.stream = stream if stream is not None else sys.stderr
        self._last = time.monotonic()

    def tick(self, stage, **counters):
        now = time.monotonic()
        if now - self._last >= self.interval:
            parts = " ".join(f"{k}={v}" for k, v in counters.items())
            print(f"[envelope_dict2] {stage}: {parts} "
                  f"(+{now - self._last:.0f}s)", file=self.stream)
            self._last = now


def _check_deadline(deadline, stage):
    if deadline is not None and time.monotonic() > deadline:
        raise DictionaryBuildError(
            f"build exceeded --max-build-seconds during '{stage}' — "
            f"aborted, no partial dictionary returned"
        )


# --------------------------------------------------------------------------
# encode / decode

def _seg_scan_rx(payload):
    """Lazily compiled alternation of all segment texts, longest-first
    (re alternation is ordered, so the longest match at a position
    wins — exactly the old per-position longest-first loop, in C)."""
    rx = payload.get("_seg_rx")
    if rx is None:
        segs = payload["segment_codes"]
        if segs:
            order = sorted(segs, key=len, reverse=True)
            rx = re.compile("|".join(re.escape(s) for s in order))
        else:
            rx = False
        payload["_seg_rx"] = rx
    return rx or None


def _code_sub_rx(payload):
    """Lazily compiled alternation of all codes (segments + frames),
    longest-first, plus the code->text reverse map. Codes are pairwise
    non-substrings (validate_dictionary), so one simultaneous pass is
    identical to sequential longest-first replacement."""
    rx = payload.get("_code_rx")
    if rx is None:
        rev = {}
        for mapping in (payload["segment_codes"], payload["frame_codes"]):
            for seg, code in mapping.items():
                rev[code] = seg
        if rev:
            rx = re.compile("|".join(
                re.escape(c) for c in sorted(rev, key=len, reverse=True)))
        else:
            rx = False
        payload["_code_rx"] = rx
        payload["_code_map"] = rev
    return rx or None


def encode_frame(payload, text):
    """Lossless encode: whole-frame code first, else left-to-right
    longest-segment substitution."""
    frame_code = payload["frame_codes"].get(text)
    if frame_code is not None:
        return frame_code
    segs = payload["segment_codes"]
    if not segs:
        return text
    rx = _seg_scan_rx(payload)
    out = []
    last = 0
    for m in rx.finditer(text):
        out.append(text[last:m.start()])
        out.append(segs[m.group(0)])
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def decode_frame(payload, text):
    """Inverse of encode_frame. Codes are fixed-width per family with
    distinct prefixes, so no code is a substring of another and a single
    longest-first simultaneous substitution pass is exact."""
    rx = _code_sub_rx(payload)
    if rx is None:
        return text
    rev = payload["_code_map"]
    return rx.sub(lambda m: rev[m.group(0)], text)


# --------------------------------------------------------------------------
# validation


def validate_dictionary(payload):
    """Raise DictionaryBuildError on duplicate codes, code-substring
    collisions, or non-ASCII/unfriendly codes. No partial results."""
    codes = {}
    for mapping, label in (
        (payload.get("segment_codes", {}), "segment"),
        (payload.get("frame_codes", {}), "frame"),
    ):
        for key, code in mapping.items():
            if code in codes:
                raise DictionaryBuildError(
                    f"duplicate code {code!r} ({label} vs {codes[code]})"
                )
            codes[code] = label
            if not code.isascii() or not code.isalnum():
                raise DictionaryBuildError(f"non-friendly code {code!r}")
    ordered = sorted(codes)
    for a, b in zip(ordered, ordered[1:]):
        if b.startswith(a):
            raise DictionaryBuildError(f"code {a!r} is a prefix of {b!r}")
    return True


# --------------------------------------------------------------------------
# build


def build_dictionary(train_frames, n_test=0, min_chars=MIN_CHARS,
                     min_freq=MIN_FREQ, max_candidates=MAX_CANDIDATES,
                     max_build_seconds=MAX_BUILD_SECONDS):
    """Build the v2 dictionary from TRAIN frames only. `n_test` is the
    TEST frame count used for the economy projection (a count, never the
    test content itself — no leakage). Candidate generation is bounded
    (prefix/suffix mining for non-JSON texts, hard cap on candidate count
    at `max_candidates`). Selection runs purely on a ONE-pass occurrence
    index (Aho-Corasick over the capped candidates) — never a per-
    candidate corpus rescan. A heartbeat line goes to stderr every 10s;
    exceeding `max_build_seconds` aborts with DictionaryBuildError.
    Verifies decode(encode(x)) == x for EVERY train frame; any violation
    aborts (raises)."""
    deadline = (time.monotonic() + max_build_seconds
                if max_build_seconds else None)
    heartbeat = _Heartbeat()
    allocator = CodeAllocator()

    # ---- stage 1: whole-frame exact match (V2.1 behavior) --------------
    frame_codes = {}
    for fi, frame in enumerate(train_frames):
        if fi % 500 == 0:
            heartbeat.tick("frame-codes", frames=fi,
                           codes=len(frame_codes))
            _check_deadline(deadline, "frame-codes")
        text = frame["frame_text"]
        if text not in frame_codes:
            frame_codes[text] = allocator.next_frame_code()

    # ---- stage 2: segment mining ----------------------------------------
    # dedupe texts (frequency = frames containing, weights per unique text)
    unique_texts = []
    weights = []
    index_of = {}
    for frame in train_frames:
        text = frame["frame_text"]
        idx = index_of.get(text)
        if idx is None:
            index_of[text] = len(unique_texts)
            unique_texts.append(text)
            weights.append(1)
        else:
            weights[idx] += 1

    n_train = len(train_frames)
    containment = _token_span_candidates(unique_texts, min_chars)
    for seg, idxs in _substring_candidates(unique_texts, min_chars).items():
        containment.setdefault(seg, set()).update(idxs)
    _check_deadline(deadline, "candidate-mining")

    candidates = []
    for seg, idxs in containment.items():
        freq = sum(weights[i] for i in idxs)
        if len(seg) < min_chars:
            toks = tokenize(seg)
            if toks is None or field_count(toks) < MIN_FIELDS:
                continue
        if freq < min_freq:
            continue
        candidates.append((seg, freq))
    # hard cap: keep the top by (length x freq); ties broken
    # lexicographically (determinism) — never OOM
    if len(candidates) > max_candidates:
        candidates.sort(key=lambda c: (-(len(c[0]) * c[1]), c[0]))
        dropped = len(candidates) - max_candidates
        candidates = candidates[:max_candidates]
        print(
            f"WARNING: candidate segments exceeded --max-candidates "
            f"({max_candidates}); kept top by length*freq, dropped {dropped}",
            file=sys.stderr,
        )
    # greedy longest-first; ties broken lexicographically (determinism)
    candidates.sort(key=lambda c: (-len(c[0]), c[0]))

    # ---- ONE-pass occurrence index over the capped candidates ---------
    # (Aho-Corasick automaton over the candidate patterns, scanned once
    # per unique text — replaces the per-candidate corpus rescans that
    # made selection O(candidates x frames x scan))
    occ_index = _scan_occurrences(
        unique_texts, [c[0] for c in candidates], heartbeat, deadline)

    occupied = _SpanTable()
    segment_codes = {}
    selection_log = []
    for ci, (seg, _freq) in enumerate(candidates):
        if ci % 500 == 0:
            heartbeat.tick("selection", candidates=ci,
                           selected=len(segment_codes))
            _check_deadline(deadline, "selection")
        seg_len = len(seg)
        usable = []
        frames_hit = set()
        for packed in occ_index[ci]:
            idx = packed >> 32
            start = packed & 0xFFFFFFFF
            end = start + seg_len
            if occupied.free(idx, start, end):
                usable.append((idx, start, end))
                frames_hit.add(idx)
        if not usable:
            continue
        freq = sum(weights[i] for i in frames_hit)
        if freq < min_freq:
            continue
        code = allocator.peek_segment_code()
        per_occ_saving = approx_count(seg) - approx_count(code)
        legend_share = approx_count(f"{code}\t{seg}\n")
        projected = freq * (n_test / n_train) if n_train else 0.0
        if projected * per_occ_saving <= legend_share:
            continue  # economy rule: cannot amortize its own legend line
        code = allocator.commit_segment_code()
        segment_codes[seg] = code
        for idx, start, end in usable:
            occupied.add(idx, start, end)
        selection_log.append(
            {
                "code": code,
                "segment_chars": len(seg),
                "train_freq": freq,
                "per_occ_saving_approx_tokens": per_occ_saving,
                "legend_share_approx_tokens": legend_share,
                "test_projected_freq": round(projected, 4),
            }
        )

    seg_lens = [len(s) for s in segment_codes]
    legend_text = _legend_text(frame_codes, segment_codes)
    payload = {
        "version": 2,
        "spec": "SPEC-WAVE4A.md",
        "frame_codes": frame_codes,
        "segment_codes": segment_codes,
        "legend_text": legend_text,
        "legend_tokens": approx_count(legend_text),
        "frame_legend_tokens": approx_count(_frame_legend_text(frame_codes)),
        "segment_legend_tokens": approx_count(_segment_legend_text(segment_codes)),
        "thresholds": {
            "min_chars": min_chars,
            "min_fields": MIN_FIELDS,
            "min_freq": min_freq,
            "max_candidates": max_candidates,
        },
        "economy": {
            "rule": (
                "test_projected_freq * per_occurrence_saving > "
                "amortized legend cost share (own legend line)"
            ),
            "projection": "train_freq * (n_test_frames / n_train_frames)",
            "n_test_frames": n_test,
            "n_train_frames": n_train,
        },
        "selection_log": selection_log,
        "_max_seg_len": max(seg_lens) if seg_lens else 0,
        "_min_seg_len": min(seg_lens) if seg_lens else 1,
    }

    validate_dictionary(payload)
    verify_lossless(payload, train_frames, heartbeat=heartbeat,
                    deadline=deadline)
    return payload


def public_payload(payload):
    """Strip runtime caches for serialization (deterministic bytes)."""
    return {
        k: v for k, v in payload.items() if not k.startswith("_")
    }


def hydrate(payload):
    """Recompute runtime caches after loading from disk."""
    seg_lens = [len(s) for s in payload["segment_codes"]]
    payload["_max_seg_len"] = max(seg_lens) if seg_lens else 0
    payload["_min_seg_len"] = min(seg_lens) if seg_lens else 1
    return payload


def _segment_legend_text(segment_codes):
    if not segment_codes:
        return ""
    lines = [
        f"{code}\t{seg}"
        for seg, code in sorted(segment_codes.items(), key=lambda kv: kv[1])
    ]
    return "=== segment legend ===\n" + "\n".join(lines) + "\n"


def _frame_legend_text(frame_codes):
    if not frame_codes:
        return ""
    lines = [
        f"{code}\t{text}"
        for text, code in sorted(frame_codes.items(), key=lambda kv: kv[1])
    ]
    return "=== frame legend ===\n" + "\n".join(lines) + "\n"


def _legend_text(frame_codes, segment_codes):
    return _segment_legend_text(segment_codes) + _frame_legend_text(frame_codes)


def verify_lossless(payload, frames, heartbeat=None, deadline=None):
    """decode(encode(x)) == x for EVERY frame or the build aborts."""
    if heartbeat is None:
        heartbeat = _Heartbeat()
    for fi, frame in enumerate(frames):
        if fi % 500 == 0:
            heartbeat.tick("verify-lossless", frames=fi, total=len(frames))
            _check_deadline(deadline, "verify-lossless")
        text = frame["frame_text"]
        decoded = decode_frame(payload, encode_frame(payload, text))
        if decoded != text:
            raise DictionaryBuildError(
                f"round-trip failed for frame {frame.get('session')}:{frame.get('seq')}"
            )
    return True


def serialize(payload):
    return json.dumps(public_payload(payload), sort_keys=True,
                      indent=2, ensure_ascii=False) + "\n"


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="corpus_train.jsonl")
    parser.add_argument("--test", required=True, help="corpus_test.jsonl")
    parser.add_argument("--out", required=True, help="dictionary.json output")
    parser.add_argument("--min-seg-chars", type=int, default=MIN_CHARS,
                        help="min segment length in chars (default %(default)s)")
    parser.add_argument("--min-freq", type=int, default=MIN_FREQ,
                        help="min TRAIN frequency (frames containing) "
                             "(default %(default)s)")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES,
                        help="hard cap on mined candidates; top by "
                             "length*freq are kept (default %(default)s)")
    parser.add_argument("--max-build-seconds", type=float,
                        default=MAX_BUILD_SECONDS,
                        help="wall-clock budget for the build; exceeding "
                             "it aborts with a clear error "
                             "(default %(default)s)")
    args = parser.parse_args(argv)

    def read_jsonl(path):
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]

    train = read_jsonl(args.train)
    test = read_jsonl(args.test)
    payload = build_dictionary(train, n_test=len(test),
                               min_chars=args.min_seg_chars,
                               min_freq=args.min_freq,
                               max_candidates=args.max_candidates,
                               max_build_seconds=args.max_build_seconds)
    # losslessness proven on the FULL corpus (train + test apply-time too);
    # the same budget covers the apply-time verification pass
    verify_lossless(payload, test,
                    deadline=time.monotonic() + args.max_build_seconds)
    out = dict(public_payload(payload))
    out["lossless_proven"] = True
    out["frames_verified"] = len(train) + len(test)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(out, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    print(
        f"dictionary: {len(out['frame_codes'])} frame codes, "
        f"{len(out['segment_codes'])} segment codes, legend "
        f"{out['legend_tokens']} approx tokens; lossless over "
        f"{out['frames_verified']} frames -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
