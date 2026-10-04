#!/usr/bin/env python3
"""Experiment A (stable bundle compression) + Experiment B (quiz-braked
compressor) per SPEC-EXPAB.md.

Rules honored throughout:
- workspace files under /root/clawd are READ-ONLY (never written);
- every model call goes through ab/instrument.py (raw answer persisted
  VERBATIM to the ledger before any parsing is trusted; no silent
  fallbacks — parse failures become explicit parse_error records);
- heartbeats every 10s on long loops; --max-seconds wall-clock guard
  (default 600) on every all-loops stage;
- Python 3 stdlib only;
- tool-schema extraction from the installed openclaw package is a
  bounded-effort attempt; when the dist contains no statically
  extractable JSON tool definitions it SKIPs with an explicit note
  (never fabricates schemas).

CLI:
  python3 -m ab.expAB expa [--model M] [--max-seconds S] ...
  python3 -m ab.expAB expb [--model M] [--max-seconds S] ...
  TCB_EXPAB_STUB=1 (or --stub) runs the declared deterministic stub
  client (smoke mode, recorded as client='stub' in the ledger).
"""

import argparse
import datetime
import difflib
import json
import math
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ab import cipher as cipher_mod  # noqa: E402
from ab import envelope_dict2 as dict2  # noqa: E402
from ab.instrument import (  # noqa: E402
    call_model,
    default_client,
    parse_json_block,
)
from ab.run_ab import QA_PROMPT, score_answer  # noqa: E402
from tcb.tokencount import approx_count  # noqa: E402

CLAWD_DIR = os.environ.get("TCB_CLAWD_DIR", "/root/clawd")
OPENCLAW_DIST = os.environ.get(
    "TCB_OPENCLAW_DIST", "/usr/lib/node_modules/openclaw/dist"
)
BUNDLE_FILES = [
    "AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "MEMORY.md", "IDENTITY.md",
]
DEFAULT_MODEL = os.environ.get("TCB_MODEL", "zai/glm-5.3-flash")
DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_SECONDS = 600.0
HEARTBEAT_INTERVAL = 10.0

# SPEC-EXPAB §A3 honest caveat, recorded verbatim in expA results.
QUIZ_CAVEAT = (
    "quiz measures recall of bundle content, not instruction-following "
    "compliance; a high score does not prove the compressed bundle still "
    "steers behavior the same way"
)

EXPB_STYLES = ["S1", "S2"]
EXPB_MODES = ["M0", "M1"]
EXPB_LEVELS = ["L1", "L2", "L3"]
CIPHER_CONTROL_TASKS = 5
QUIZ_QUESTIONS_PER_TASK = 6
BRAKE_SLACK = 1  # accept while score >= baseline - 1


class ExpABError(Exception):
    """Explicit experiment failure (missing material, losslessness gate)."""


class DeadlineExceeded(ExpABError):
    """A loop stage exceeded its --max-seconds wall-clock budget."""


class Heartbeat:
    """One progress line to stderr every HEARTBEAT_INTERVAL seconds —
    silence must be visible."""

    def __init__(self, interval=HEARTBEAT_INTERVAL, stream=None):
        self.interval = interval
        self.stream = stream if stream is not None else sys.stderr
        self._last = time.monotonic()

    def tick(self, stage, **counters):
        now = time.monotonic()
        if now - self._last >= self.interval:
            parts = " ".join(f"{k}={v}" for k, v in counters.items())
            print(f"[expAB] {stage}: {parts} (+{now - self._last:.0f}s)",
                  file=self.stream)
            self._last = now


def check_deadline(deadline, stage):
    if deadline is not None and time.monotonic() > deadline:
        raise DeadlineExceeded(
            f"stage '{stage}' exceeded --max-seconds — aborted, "
            f"no partial results hidden"
        )


class Ledger:
    """Append-only JSONL ledger; every record stamped with run identity."""

    def __init__(self, path, run_meta):
        self.path = path
        self.run_meta = run_meta
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def append(self, record):
        rec = dict(self.run_meta)
        rec.update(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


def git_sha():
    override = os.environ.get("TCB_EXPAB_GIT_SHA")
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "-C", _ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unavailable (not a git checkout)"


def make_stub_client():
    """Declared deterministic stub (TCB_EXPAB_STUB=1): answers carry a
    recognizable marker; the ledger records client='stub'. A declared
    smoke mode, never a silent fallback."""

    def client(payload):
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        if "quiz questions" in system:
            text = user.rsplit("Text:\n", 1)[1]
            words = text.split()
            anchor = " ".join(words[:3])
            n = 6
            m = re.search(r"Write (\d+) questions", user)
            if m:
                n = int(m.group(1))
            answer = json.dumps(
                [{"q": f"stub question {i+1}?", "a": anchor}
                 for i in range(n)]
            )
        elif "fact-heavy" in system:
            lines = [l for l in user.splitlines() if re.match(r"^\d+\.", l)]
            answer = json.dumps(["fact-heavy"] * len(lines))
        elif "Answer:" in user:
            answer = "stub-answer"
        else:
            answer = "stub-compressed"
        pt = approx_count(user) + 8
        return {
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": 4,
                      "total_tokens": pt + 4},
        }

    return client


# ==========================================================================
# Experiment A — the stable per-turn bundle
# ==========================================================================

def capture_bundle(clawd_dir=CLAWD_DIR, files=BUNDLE_FILES):
    """READ-ONLY capture of the workspace bundle. Missing file => explicit
    error (never an empty silent stand-in)."""
    bundle = []
    for name in files:
        path = os.path.join(clawd_dir, name)
        if not os.path.isfile(path):
            raise ExpABError(f"bundle file missing: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        bundle.append({
            "name": name,
            "chars": len(text),
            "approx_tokens": approx_count(text),
            "text": text,
        })
    return bundle


def structural_strip(text):
    """V1: deterministic markdown -> compact-text collapse. Headers and
    bullet/list markers are dropped, decoration lines removed, whitespace
    collapsed. Pure function (same input => same output, always)."""
    lines = text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[-=_*]{3,}", stripped):
            continue  # horizontal rules / underline decoration
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)      # ATX headers
        stripped = re.sub(r"^[-*+]\s+", "", stripped)       # bullets
        stripped = re.sub(r"^\d+[.)]\s+", "", stripped)     # list numbers
        stripped = stripped.replace("**", "").replace("__", "")
        stripped = re.sub(r"^>\s?", "", stripped)           # blockquotes
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped:
            kept.append(stripped)
    return "\n".join(kept)


# --- tool-definition extraction (bounded effort, honest skip) ------------

_TOOL_JSON_RE = re.compile(
    r'\{\s*"name"\s*:\s*"[A-Za-z0-9_.-]+"\s*,\s*"description"\s*:\s*"[^"]*"'
    r'(?=[^{}]*"(?:parameters|inputSchema)")',
    re.DOTALL,
)


def _extract_balanced(text, start):
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_tool_defs(dist_dir=OPENCLAW_DIST, max_files=400,
                      max_file_bytes=3_000_000, max_seconds=120):
    """Bounded-effort static extraction of tool JSON-schema definitions
    from the installed openclaw package. Looks for JSON object literals
    carrying name + description + parameters/inputSchema. Returns a
    status dict; on failure the status is 'skipped' with an explicit
    note (file-bundle-alone is a valid result per SPEC-EXPAB §A1).
    Never fabricates schemas."""
    if not os.path.isdir(dist_dir):
        return {
            "status": "skipped",
            "tools": [],
            "approx_tokens": 0,
            "files_scanned": 0,
            "note": f"openclaw dist directory not found: {dist_dir}",
        }
    deadline = time.monotonic() + max_seconds
    names = sorted(n for n in os.listdir(dist_dir) if n.endswith(".js"))
    tools = []
    seen = set()
    scanned = 0
    for fname in names:
        if scanned >= max_files or time.monotonic() > deadline:
            break
        path = os.path.join(dist_dir, fname)
        if not os.path.isfile(path):
            continue
        scanned += 1
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read(max_file_bytes)
        except OSError:
            continue
        for match in _TOOL_JSON_RE.finditer(content):
            blob = _extract_balanced(content, match.start())
            if blob is None:
                continue
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            schema = obj.get("parameters") or obj.get("inputSchema")
            if (isinstance(obj.get("name"), str)
                    and isinstance(obj.get("description"), str)
                    and isinstance(schema, dict)
                    and obj["name"] not in seen):
                seen.add(obj["name"])
                tools.append(obj)
    if len(tools) < 3:
        return {
            "status": "skipped",
            "tools": [],
            "approx_tokens": 0,
            "files_scanned": scanned,
            "note": (
                f"no statically extractable JSON tool definitions found in "
                f"{dist_dir} ({scanned} .js files scanned within the "
                f"bounded effort): openclaw builds model-facing tool "
                f"schemas at runtime (zod -> JSON projection in "
                f"dist/tool-schema-projection-*, bash-tools.descriptions-* "
                f"assemble descriptions in code), so there is no static "
                f"schema JSON to grep. Skipped honestly per SPEC-EXPAB "
                f"§A1; the file bundle alone is the reported result."
            ),
        }
    serialized = json.dumps(tools, sort_keys=True, ensure_ascii=False)
    return {
        "status": "ok",
        "tools": tools,
        "approx_tokens": approx_count(serialized),
        "files_scanned": scanned,
        "note": f"{len(tools)} tool definitions extracted from {scanned} files",
    }


# --- V3: fragment dictionary across bundle files --------------------------

def build_bundle_dictionary(texts_by_name, min_freq=2,
                            max_build_seconds=120):
    """Run ab/envelope_dict2.py's segment miner ACROSS the bundle files
    (paragraph frames pooled over all files). Returns the segment-only
    payload (whole-frame codes are useless for a per-turn bundle), the
    encoded texts, and the segment legend. HARD GATE: decode(encode(x))
    == x for every input text or ExpABError (losslessness, no partial).    Bundle files repeat themselves line-by-line, so frames are non-empty
    LINES pooled across all files. Returns the segment-only payload
    (whole-frame codes are useless for a per-turn bundle), the encoded
    texts, and the segment legend. HARD GATE: decode(encode(x)) == x for
    every input text or ExpABError (losslessness, no partial)."""
    frames = []
    for name, text in texts_by_name.items():
        for line in text.splitlines():
            if line.strip():
                frames.append({"frame_text": line, "file": name})
    payload = dict2.build_dictionary(
        frames, n_test=len(frames), min_freq=min_freq,
        max_build_seconds=max_build_seconds,
    )
    seg_payload = {"frame_codes": {}, "segment_codes": payload["segment_codes"]}
    encoded = {}
    for name, text in texts_by_name.items():
        enc = dict2.encode_frame(seg_payload, text)
        if dict2.decode_frame(seg_payload, enc) != text:
            raise ExpABError(f"dictionary round-trip failed for {name}")
        encoded[name] = enc
    legend = dict2._segment_legend_text(payload["segment_codes"])
    return {
        "dictionary": dict2.public_payload(payload),
        "segment_payload": seg_payload,
        "encoded": encoded,
        "legend_text": legend,
        "legend_tokens": approx_count(legend),
        "frames": len(frames),
        "segment_count": len(payload["segment_codes"]),
    }


# --- quiz ------------------------------------------------------------------

QUIZ_GEN_SYSTEM = (
    "You write reading-comprehension quiz questions about a text. Reply "
    "ONLY with a JSON array of objects {\"q\": question, \"a\": answer}, "
    "where each answer is a short verbatim span of the text. "
    "No other output."
)

QA_SYSTEM = "You answer reading-comprehension tasks faithfully and briefly."


def _norm(s):
    s = s.casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def answer_verified(answer, source_text):
    """Deterministic verification: the answer must be recoverable from the
    original text (normalized substring or near-exact span match)."""
    na, nt = _norm(answer), _norm(source_text)
    if not na:
        return False
    if na in nt:
        return True
    return difflib.SequenceMatcher(None, na, nt).ratio() >= 0.98


def generate_quiz(client, model, source_name, source_text, n_questions,
                  max_tokens, ledger, call_kind="quiz_gen"):
    """Generate n questions answerable ONLY from source_text; verify each
    answer against the original; return only verified items (every
    generated item is ledgered, verified or not)."""
    user = (
        f"Write {n_questions} questions answerable ONLY from this text, "
        f"covering its rules, aliases and facts.\n\nText:\n{source_text}"
    )
    result = call_model(client, model, QUIZ_GEN_SYSTEM, user, max_tokens,
                        parse=parse_json_block)
    ledger.append({
        "type": "model_call",
        "exp": "A" if call_kind == "quiz_gen" else "B",
        "call": call_kind,
        "source": source_name,
        "raw": result["raw"],  # VERBATIM, persisted before parsing
        "parse_error": result["parse_error"],
        "usage": result["usage"],
        "latency_ms": result["latency_ms"],
    })
    items = []
    if result["parse_error"] is not None:
        ledger.append({
            "type": "parse_error", "call": call_kind,
            "source": source_name, "parse_error": result["parse_error"],
            "raw": result["raw"],
        })
    elif not isinstance(result["parsed"], list):
        ledger.append({
            "type": "parse_error", "call": call_kind,
            "source": source_name,
            "parse_error": f"parsed JSON is {type(result['parsed']).__name__},"
                           f" expected array",
            "raw": result["raw"],
        })
    else:
        for entry in result["parsed"]:
            if (isinstance(entry, dict) and isinstance(entry.get("q"), str)
                    and isinstance(entry.get("a"), str) and entry["q"] and entry["a"]):
                items.append({"q": entry["q"], "a": entry["a"]})
    verified = []
    for item in items[:n_questions]:
        ok = answer_verified(item["a"], source_text)
        ledger.append({
            "type": "quiz_item", "source": source_name,
            "question": item["q"], "answer": item["a"], "verified": ok,
        })
        if ok:
            verified.append({"source": source_name, **item})
    return verified


def run_quiz(client, model, questions, context_text, max_tokens, ledger,
             exp, unit_tags):
    """Ask every question against context_text (temperature 0 fixed in the
    instrument); grade deterministically via score_answer; return score,
    accuracy and summed usage. Raw answers persisted per call."""
    correct = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for qi, item in enumerate(questions):
        result = call_model(
            client, model, QA_SYSTEM,
            QA_PROMPT.format(context=context_text, question=item["q"]),
            max_tokens,
        )
        kind = score_answer(result["raw"], item["a"])
        ok = kind in ("exact", "contains", "fuzzy")
        correct += ok
        usage["prompt_tokens"] += result["usage"]["prompt_tokens"]
        usage["completion_tokens"] += result["usage"]["completion_tokens"]
        usage["total_tokens"] += result["usage"]["total_tokens"]
        ledger.append({
            "type": "model_call", "exp": exp, "call": "quiz_qa",
            "raw": result["raw"], "usage": result["usage"],
            "latency_ms": result["latency_ms"],
            "correct_kind": kind, "expected": item["a"],
            "question_index": qi, **unit_tags,
        })
    return {
        "score": correct,
        "total": len(questions),
        "accuracy": round(correct / len(questions), 4) if questions else None,
        "usage": usage,
    }


# --- V2 model rewrite -------------------------------------------------------

# One worked example embedded (examples beat rules). Reused as expB style
# S1 ("S1 strict-short-label outline (as V2 above)").
STRICT_OUTLINE_SYSTEM = (
    "You compress text into a strict short-label outline. One fact per "
    "line, format 'label: value' with short lowercase labels (e.g. rule, "
    "role, tool, alias, fact, pref, tone, src). Keep every name, number, "
    "path, command and URL verbatim. No prose sentences.\n"
    "Example:\n"
    "Text: The agent must always ask the user before deleting any file. "
    "Deletion requests go to the review queue and are logged to "
    "/var/log/review.log.\n"
    "Outline:\n"
    "rule: ask user before deleting any file\n"
    "rule: deletion requests -> review queue\n"
    "fact: deletions logged /var/log/review.log\n"
    "Output ONLY the outline."
)

FREEFORM_SYSTEM = (
    "You compress text into the most compact form you choose yourself — "
    "outline, telegraphic notes, table, any format. KEEP EVERY FACT: no "
    "information may be lost; keep every name, number, path and command "
    "verbatim.\n"
    "Example:\n"
    "Text: The agent must always ask the user before deleting any file. "
    "Deletion requests go to the review queue and are logged to "
    "/var/log/review.log.\n"
    "Compressed:\n"
    "delete file: ask user first; request -> review queue; log "
    "/var/log/review.log\n"
    "Output ONLY the compressed text."
)


def rewrite_file(client, model, text, max_tokens, ledger, source_name):
    """V2: one-shot strict-short-label outline rewrite of one bundle file."""
    result = call_model(client, model, STRICT_OUTLINE_SYSTEM, text,
                        max_tokens)
    ledger.append({
        "type": "model_call", "exp": "A", "call": "v2_rewrite",
        "source": source_name, "raw": result["raw"],  # VERBATIM
        "usage": result["usage"], "latency_ms": result["latency_ms"],
    })
    return result["raw"], result["usage"]["total_tokens"]


# --- variant accounting ------------------------------------------------------

def variant_accounting(v0_tokens, variant_tokens, transform_cost_tokens):
    """Per-variant math (deterministic): tokens, per-turn saving, and the
    break-even turn = ceil(cost / saving). saving <= 0 => break-even None
    (the variant never pays for itself)."""
    saving = v0_tokens - variant_tokens
    if saving > 0 and transform_cost_tokens > 0:
        break_even = math.ceil(transform_cost_tokens / saving)
    elif saving > 0:
        break_even = 1
    else:
        break_even = None
    return {
        "bundle_approx_tokens": variant_tokens,
        "per_turn_saving_approx_tokens": saving,
        "transform_cost_tokens": transform_cost_tokens,
        "break_even_turns": break_even,
    }


def assemble_bundle_text(texts_by_name):
    return "\n\n".join(
        f"=== {name} ===\n{texts_by_name[name]}" for name in texts_by_name
    )


def run_expA(client, model, max_tokens, results_dir, clawd_dir=CLAWD_DIR,
             dist_dir=OPENCLAW_DIST, max_seconds=DEFAULT_MAX_SECONDS,
             quiz_per_file=9):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_dir, "expA")
    os.makedirs(out_dir, exist_ok=True)
    run_meta = {"run_id": f"expA_{stamp}", "git_sha": git_sha(),
                "model": model, "client": client.__name__ if hasattr(client, "__name__") else "stub"}
    ledger = Ledger(os.path.join(out_dir, "ledger.jsonl"), run_meta)
    deadline = time.monotonic() + max_seconds
    hb = Heartbeat()

    # A1: materials (read-only capture) + tool definitions
    bundle = capture_bundle(clawd_dir)
    tool_defs = extract_tool_defs(dist_dir)
    ledger.append({"type": "tool_extraction", **{
        k: v for k, v in tool_defs.items() if k != "tools"}})
    capture = {
        "clawd_dir": clawd_dir,
        "read_only": True,
        "files": [{k: v for k, v in f.items() if k != "text"} for f in bundle],
        "bundle_approx_tokens": sum(f["approx_tokens"] for f in bundle),
        "tool_definitions": {k: v for k, v in tool_defs.items()
                             if k != "tools"},
    }
    with open(os.path.join(out_dir, "capture.json"), "w",
              encoding="utf-8") as fh:
        json.dump(capture, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")

    texts0 = {f["name"]: f["text"] for f in bundle}
    v0_tokens = sum(f["approx_tokens"] for f in bundle)
    if tool_defs["status"] == "ok":
        v0_tokens += tool_defs["approx_tokens"]

    # A3: quiz generation (verified against the original text)
    questions = []
    for f in bundle:
        check_deadline(deadline, "expA-quiz-gen")
        hb.tick("expA-quiz-gen", file=f["name"], questions=len(questions))
        questions.extend(generate_quiz(
            client, model, f["name"], f["text"], quiz_per_file,
            max_tokens, ledger))
    if not questions:
        raise ExpABError("quiz generation produced 0 verified questions — "
                         "cannot measure quality; aborting (no silent "
                         "fallback to no-quiz)")

    # A2: variants
    variants = {}

    # V0 baseline
    variants["V0"] = {
        "texts": texts0,
        "extra_tokens": tool_defs["approx_tokens"]
        if tool_defs["status"] == "ok" else 0,
        "transform_cost_tokens": 0,
        "method": "as-is",
    }

    # V1 structural strip (deterministic)
    v1_texts = {name: structural_strip(text) for name, text in texts0.items()}
    variants["V1"] = {
        "texts": v1_texts, "extra_tokens": 0,
        "transform_cost_tokens": 0,
        "method": "deterministic markdown->compact-text strip",
    }

    # V2 model rewrite (one shot per file)
    v2_texts = {}
    v2_cost = 0
    for f in bundle:
        check_deadline(deadline, "expA-v2")
        hb.tick("expA-v2", file=f["name"])
        out, cost = rewrite_file(client, model, f["text"], max_tokens,
                                 ledger, f["name"])
        v2_texts[f["name"]] = out
        v2_cost += cost
    variants["V2"] = {
        "texts": v2_texts, "extra_tokens": 0,
        "transform_cost_tokens": v2_cost,
        "method": "one-shot strict-short-label outline rewrite per file",
    }

    # V3 fragment dictionary across files (lossless, verified)
    v3 = build_bundle_dictionary(texts0)
    variants["V3"] = {
        "texts": v3["encoded"], "extra_tokens": v3["legend_tokens"],
        "transform_cost_tokens": 0,
        "method": "segment dictionary mined across bundle files "
                  "(lossless; markers + legend counted)",
        "dictionary_stats": {
            "frames": v3["frames"],
            "segments": v3["segment_count"],
            "legend_tokens": v3["legend_tokens"],
        },
    }
    ledger.append({
        "type": "dictionary_build", "exp": "A",
        "frames": v3["frames"], "segments": v3["segment_count"],
        "legend_tokens": v3["legend_tokens"],
        "lossless": True,
    })

    # V2+V3 combined: dictionary over the rewritten texts
    v23 = build_bundle_dictionary(v2_texts)
    v23_cost = v2_cost  # dictionary build is deterministic (no model calls)
    variants["V2+V3"] = {
        "texts": v23["encoded"], "extra_tokens": v23["legend_tokens"],
        "transform_cost_tokens": v23_cost,
        "method": "V2 rewrite then cross-file dictionary",
        "dictionary_stats": {
            "frames": v23["frames"],
            "segments": v23["segment_count"],
            "legend_tokens": v23["legend_tokens"],
        },
    }

    # A3: quiz under each variant + accounting (all variants measured on
    # the same footing: the assembled context text, headers included)
    table = []
    contexts = {}
    for name, var in variants.items():
        check_deadline(deadline, "expA-quiz")
        hb.tick("expA-quiz", variant=name)
        context = (assemble_bundle_text(var["texts"])
                   + ("\n\n" + v3["legend_text"] if name == "V3" else "")
                   + ("\n\n" + v23["legend_text"] if name == "V2+V3" else ""))
        contexts[name] = (approx_count(context) + var["extra_tokens"])
    v0_total = contexts["V0"]  # baseline = as-is assembled bundle
    for name, var in variants.items():
        check_deadline(deadline, "expA-quiz")
        hb.tick("expA-quiz", variant=name)
        context = (assemble_bundle_text(var["texts"])
                   + ("\n\n" + v3["legend_text"] if name == "V3" else "")
                   + ("\n\n" + v23["legend_text"] if name == "V2+V3" else ""))
        variant_tokens = contexts[name]
        quiz = run_quiz(client, model, questions, context, max_tokens,
                        ledger, "A", {"variant": name})
        acc = variant_accounting(v0_total, variant_tokens,
                                 var["transform_cost_tokens"])
        row = {
            "variant": name,
            "method": var["method"],
            "quiz_score": quiz["score"],
            "quiz_total": quiz["total"],
            "quiz_accuracy": quiz["accuracy"],
            "quiz_usage": quiz["usage"],
            **acc,
        }
        if "dictionary_stats" in var:
            row["dictionary_stats"] = var["dictionary_stats"]
        table.append(row)
        ledger.append({"type": "variant_summary", "exp": "A", **row})

    summary = {
        "type": "expA_summary",
        "run_id": run_meta["run_id"],
        "bundle_approx_tokens": v0_total,
        "tool_definitions_status": tool_defs["status"],
        "tool_definitions_note": tool_defs["note"],
        "caveat": QUIZ_CAVEAT,
        "questions_verified": len(questions),
        "variants": table,
    }
    with open(os.path.join(out_dir, "variants.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"expA: bundle {v0_total} approx tokens (raw files: {v0_tokens}); "
          f"{len(questions)} verified questions; "
          f"tool definitions: {tool_defs['status']}")
    for row in table:
        print(f"  {row['variant']}: {row['bundle_approx_tokens']} tokens "
              f"(-{row['per_turn_saving_approx_tokens']}/turn), quiz "
              f"{row['quiz_score']}/{row['quiz_total']}, break-even "
              f"{row['break_even_turns']} turns")
    print(f"  caveat: {QUIZ_CAVEAT}")
    return summary


# ==========================================================================
# Experiment B — quiz-braked compressor
# ==========================================================================

LEVEL_INSTRUCTIONS = {
    "S1": {
        "L1": "LIGHT: drop filler and redundancy only; keep nearly every "
              "outline line.",
        "L2": "MEDIUM: merge related lines and shorten labels further.",
        "L3": "DENSE: maximum compression; telegraphic labels, shortest "
              "lines that still carry every fact.",
    },
    "S2": {
        "L1": "LIGHT: drop filler and redundancy only.",
        "L2": "MEDIUM: aggressively shorten, any format.",
        "L3": "DENSE: maximum compression, any format.",
    },
}

LABEL_SYSTEM = (
    "You classify paragraphs of a passage. Reply ONLY with a JSON array "
    "with exactly one label per numbered paragraph; each label is exactly "
    "\"fact-heavy\" or \"narrative\". No other output."
)


def split_paragraphs(text):
    """Paragraph units for M1 selective mode; single-block passages are
    chunked into ~2-sentence groups so labeling still has units."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:
        sentences = [s.strip() for s in
                     re.findall(r"[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$", text)
                     if s.strip()]
        if len(sentences) >= 4:
            paras = [" ".join(sentences[i:i + 2])
                     for i in range(0, len(sentences), 2)]
        elif sentences:
            paras = sentences
    return paras


def label_paragraphs(client, model, paragraphs, max_tokens, ledger,
                     unit_tags):
    """M1: model labels each paragraph fact-heavy vs narrative.
    Returns list of labels or None after an EXPLICIT parse_error record
    (no silent fallback to a default labeling)."""
    listing = "\n".join(f"{i+1}. {p}" for i, p in enumerate(paragraphs))
    result = call_model(client, model, LABEL_SYSTEM, listing, max_tokens,
                        parse=parse_json_block)
    ledger.append({
        "type": "model_call", "exp": "B", "call": "label_paragraphs",
        "raw": result["raw"], "usage": result["usage"],
        "latency_ms": result["latency_ms"], **unit_tags,
    })
    parsed, err = result["parsed"], result["parse_error"]
    if err is None:
        if not isinstance(parsed, list) or len(parsed) != len(paragraphs):
            err = (f"label array length {len(parsed) if isinstance(parsed, list) else 'n/a'}"
                   f" != paragraph count {len(paragraphs)}")
        elif not all(x in ("fact-heavy", "narrative") for x in parsed):
            err = "label array contains values other than fact-heavy/narrative"
    if err is not None:
        ledger.append({
            "type": "parse_error", "exp": "B", "call": "label_paragraphs",
            "parse_error": err, "raw": result["raw"], **unit_tags,
        })
        return None
    return list(parsed)


def transform_passage(client, model, style, level, text, max_tokens,
                      ledger, unit_tags):
    system = (STRICT_OUTLINE_SYSTEM if style == "S1" else FREEFORM_SYSTEM)
    user = (f"Aggressiveness: {LEVEL_INSTRUCTIONS[style][level]}\n\n"
            f"Text:\n{text}")
    result = call_model(client, model, system, user, max_tokens)
    ledger.append({
        "type": "model_call", "exp": "B", "call": "transform",
        "style": style, "mode": unit_tags.get("mode"),
        "level": level, "task_id": unit_tags.get("task_id"),
        "raw": result["raw"], "usage": result["usage"],
        "latency_ms": result["latency_ms"],
    })
    return result["raw"], result["usage"]["total_tokens"]


def run_brake(task, style, mode, client, model, max_tokens, ledger,
              deadline=None, hb=None, cipher_table=None,
              quiz_runner=None):
    """One (task, style, mode) brake run per SPEC-EXPAB §B3.
    quiz_runner is injectable for tests (defaults to the real quiz)."""
    hb = hb or Heartbeat()
    quiz_runner = quiz_runner or run_quiz
    unit_tags = {"task_id": task["id"], "style": style, "mode": mode}
    passage = task["passage"]
    passage_tokens = approx_count(passage)

    # 1. questions from the ORIGINAL passage, verified
    questions = generate_quiz(client, model, task["id"], passage,
                              QUIZ_QUESTIONS_PER_TASK, max_tokens, ledger,
                              call_kind="b_quiz_gen")
    if not questions:
        rec = {"type": "b_unit", "state": "skipped_no_verified_questions",
               **unit_tags}
        ledger.append(rec)
        return rec

    # 2. baseline quiz on the original
    baseline = quiz_runner(client, model, questions, passage, max_tokens,
                           ledger, "B", unit_tags)
    threshold = baseline["score"] - BRAKE_SLACK

    # M1: label paragraphs on the ORIGINAL once (explicit skip on failure)
    labels = None
    if mode == "M1":
        paragraphs = split_paragraphs(passage)
        labels = label_paragraphs(client, model, paragraphs, max_tokens,
                                  ledger, unit_tags)
        if labels is None:
            rec = {"type": "b_unit", "state": "skipped_label_parse_error",
                   **unit_tags, "baseline_score": baseline["score"]}
            ledger.append(rec)
            return rec
        ledger.append({"type": "b_labels", "labels": labels,
                       "paragraphs": len(paragraphs), **unit_tags})

    # 3. ladder L1 -> L3 with the brake
    levels = {}
    chosen = "L0"
    chosen_savings = 0
    brake_events = []
    cipher_mode = cipher_table is not None
    for level in EXPB_LEVELS:
        if deadline is not None:
            check_deadline(deadline, f"expB-ladder-{task['id']}-{style}-{mode}")
        hb.tick("expB-ladder", task=task["id"], style=style, mode=mode,
                level=level)
        if cipher_mode:
            # cipher control: deterministic transform, no model call
            compressed = cipher_mod.encode(passage, cipher_table)
            cost = 0
        elif mode == "M1":
            fact_paras = [p for p, lab in zip(split_paragraphs(passage),
                                              labels)
                          if lab == "fact-heavy"]
            narrative = [p for p, lab in zip(split_paragraphs(passage),
                                             labels)
                         if lab == "narrative"]
            joined = "\n\n".join(fact_paras)
            rewritten, cost = transform_passage(
                client, model, style, level, joined, max_tokens, ledger,
                dict(unit_tags, level=level))
            compressed = rewritten + "\n\n" + "\n\n".join(narrative)
        else:
            compressed, cost = transform_passage(
                client, model, style, level, passage, max_tokens, ledger,
                dict(unit_tags, level=level))
        comp_tokens = approx_count(compressed)
        quiz = quiz_runner(client, model, questions, compressed,
                           max_tokens, ledger, "B",
                           dict(unit_tags, level=level))
        levels[level] = {
            "score": quiz["score"],
            "compressed_approx_tokens": comp_tokens,
            "savings_approx_tokens": passage_tokens - comp_tokens,
            "transform_cost_tokens": cost,
            "quiz_usage": quiz["usage"],
        }
        if quiz["score"] >= threshold:
            chosen = level
            chosen_savings = passage_tokens - comp_tokens
        else:
            brake_events.append({
                "stopped_at": level,
                "score": quiz["score"], "threshold": threshold,
            })
            break  # brake: stop, keep last accepted (L0 = original)

    rec = {
        "type": "b_unit",
        "state": "done",
        "register": task.get("register"),
        "len_class": task.get("len_class"),
        "passage_approx_tokens": passage_tokens,
        "baseline_score": baseline["score"],
        "threshold": threshold,
        "questions": len(questions),
        "levels": levels,
        "chosen_level": chosen,
        "tokens_saved_at_chosen": chosen_savings,
        "pct_saved_at_chosen": round(100.0 * chosen_savings / passage_tokens, 2)
        if passage_tokens else 0.0,
        "brake_events": brake_events,
        "cipher_control": cipher_mode,
        **unit_tags,
    }
    ledger.append(rec)
    return rec


def summarize_expB(units, cipher_units=None):
    """Frontier per style x mode: max mean savings at quiz-parity
    (mean level score >= mean baseline - 1), chosen-level stats, brake
    stats, per-content-type (register) breakdown."""
    done = [u for u in units if u.get("state") == "done"]
    frontier = {}
    for style in EXPB_STYLES:
        for mode in EXPB_MODES:
            combo = [u for u in done
                     if u["style"] == style and u["mode"] == mode]
            if not combo:
                continue
            per_level = {}
            for level in EXPB_LEVELS:
                reached = [u for u in combo if level in u["levels"]]
                if not reached:
                    continue
                mean_score = sum(u["levels"][level]["score"] for u in reached) / len(reached)
                mean_base = sum(u["baseline_score"] for u in reached) / len(reached)
                mean_sav = sum(u["levels"][level]["savings_approx_tokens"]
                               for u in reached) / len(reached)
                per_level[level] = {
                    "tasks_reached": len(reached),
                    "mean_score": round(mean_score, 4),
                    "quiz_parity": mean_score >= mean_base - BRAKE_SLACK,
                    "mean_savings_approx_tokens": round(mean_sav, 2),
                    "mean_pct_saved": round(
                        sum(u["levels"][level]["savings_approx_tokens"]
                            / max(u["passage_approx_tokens"], 1)
                            for u in reached) / len(reached) * 100, 2),
                }
            parity_levels = [l for l, s in per_level.items() if s["quiz_parity"]]
            best = max(parity_levels, key=lambda l: per_level[l]["mean_savings_approx_tokens"]) if parity_levels else None
            frontier[f"{style}/{mode}"] = {
                "units": len(combo),
                "frontier_level": best,
                "frontier_mean_savings_approx_tokens": (
                    per_level[best]["mean_savings_approx_tokens"] if best else None),
                "frontier_mean_pct_saved": (
                    per_level[best]["mean_pct_saved"] if best else None),
                "per_level": per_level,
                "mean_chosen_level": (
                    sum(int(u["chosen_level"][1]) for u in combo) / len(combo)),
                "mean_pct_saved_at_chosen": round(
                    sum(u["pct_saved_at_chosen"] for u in combo) / len(combo), 2),
                "mean_baseline_score": round(
                    sum(u["baseline_score"] for u in combo) / len(combo), 4),
                "mean_score_at_chosen": round(
                    sum(u["levels"][u["chosen_level"]]["score"]
                        for u in combo if u["chosen_level"] != "L0") / max(
                        len([u for u in combo if u["chosen_level"] != "L0"]), 1), 4),
                "brake_events": sum(len(u["brake_events"]) for u in combo),
                "stopped_at_L0": sum(u["chosen_level"] == "L0" for u in combo),
                "per_register": _per_register(combo),
            }
    cipher_out = None
    if cipher_units is not None:
        cipher_done = [u for u in cipher_units if u.get("state") == "done"]
        cipher_out = {
            "expected": "cipher must fail the brake at L1 (chosen L0)",
            "units": len(cipher_done),
            "all_failed_at_L1": all(u["chosen_level"] == "L0" for u in cipher_done),
            "chosen_levels": {u["task_id"]: u["chosen_level"] for u in cipher_done},
        }
    return {"type": "expB_summary", "frontier": frontier,
            "cipher_control": cipher_out}


def _per_register(combo):
    out = {}
    for reg in sorted({u["register"] for u in combo if u.get("register")}):
        sub = [u for u in combo if u.get("register") == reg]
        out[reg] = {
            "units": len(sub),
            "mean_pct_saved_at_chosen": round(
                sum(u["pct_saved_at_chosen"] for u in sub) / len(sub), 2),
            "mean_chosen_level": (
                sum(int(u["chosen_level"][1]) for u in sub) / len(sub)),
        }
    return out


def plain_summary_line(key, stats):
    """Plain-English three-number summary per style x mode: how much
    smaller / quality / worth it."""
    if not stats:
        return f"{key}: no units"
    pct = stats.get("frontier_mean_pct_saved")
    base = stats.get("mean_baseline_score")
    score = stats.get("mean_score_at_chosen")
    worth = (
        "worth it" if (pct is not None and pct >= 10
                       and stats.get("frontier_level")) else "not worth it"
    )
    return (f"{key}: {pct}% smaller at frontier level "
            f"{stats.get('frontier_level')} / quiz {score} vs baseline "
            f"{base} (mean of {stats.get('units')} units, "
            f"{stats.get('brake_events')} brake events) / {worth}")


def load_expB_tasks(tasks_dir):
    """Reuse ab/tasks2 fact + prose tasks (skip formulaic — already
    answered by V2.1)."""
    from ab.run_v21 import load_tasks
    return [t for t in load_tasks(tasks_dir) if t["register"] != "formulaic"]


def run_expB(client, model, max_tokens, results_dir, tasks_dir=None,
             max_seconds=DEFAULT_MAX_SECONDS, task_limit=None,
             task_offset=0, slice_id=None):
    """task_offset/task_limit select tasks[offset:offset+limit] of the
    FULL task list (slice math is over the unsliced list, so parallel
    slices tile the serial run exactly). slice_id, when set, redirects
    output to results/expB/slices/slice_<id>.* instead of the single
    ledger/summary; every record keeps the identical shape."""
    from ab.run_v21 import TASKS2_DIR
    tasks_dir = tasks_dir or TASKS2_DIR
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_dir, "expB")
    os.makedirs(out_dir, exist_ok=True)
    if slice_id:
        out_dir = os.path.join(out_dir, "slices")
        os.makedirs(out_dir, exist_ok=True)
        ledger_path = os.path.join(out_dir, f"slice_{slice_id}.jsonl")
        summary_path = os.path.join(out_dir, f"slice_{slice_id}.summary.json")
    else:
        ledger_path = os.path.join(out_dir, "ledger.jsonl")
        summary_path = os.path.join(out_dir, "summary.json")
    run_meta = {"run_id": f"expB_{stamp}", "git_sha": git_sha(),
                "model": model}
    if slice_id:
        run_meta["slice_id"] = slice_id
    ledger = Ledger(ledger_path, run_meta)
    deadline = time.monotonic() + max_seconds
    hb = Heartbeat()

    all_tasks = load_expB_tasks(tasks_dir)
    if task_offset < 0 or task_offset > len(all_tasks):
        raise ExpABError(
            f"--task-offset {task_offset} out of range "
            f"(0..{len(all_tasks)} tasks)")
    end = None if task_limit is None else task_offset + task_limit
    tasks = all_tasks[task_offset:end]
    if not tasks:
        raise ExpABError(f"no fact/prose tasks found in {tasks_dir}")

    cipher_table = cipher_mod.load_table()
    units = []
    idx = 0
    for local_idx, task in enumerate(tasks):
        for style in EXPB_STYLES:
            for mode in EXPB_MODES:
                check_deadline(deadline, "expB-units")
                hb.tick("expB-units", task=idx, total=len(tasks) * 4)
                idx += 1
                units.append(run_brake(
                    task, style, mode, client, model, max_tokens, ledger,
                    deadline=deadline, hb=hb))

    # B4: cipher control on the FIRST CIPHER_CONTROL_TASKS tasks of the
    # FULL list — a slice runs it only for the tasks it owns, so slices
    # tile the serial cipher subset without duplication.
    cipher_units = []
    for local_idx, task in enumerate(tasks):
        if task_offset + local_idx >= CIPHER_CONTROL_TASKS:
            break
        check_deadline(deadline, "expB-cipher")
        hb.tick("expB-cipher", task=task["id"])
        cipher_units.append(run_brake(
            task, "S1", "M0", client, model, max_tokens, ledger,
            deadline=deadline, hb=hb, cipher_table=cipher_table))

    summary = summarize_expB(units, cipher_units)
    summary["run_id"] = run_meta["run_id"]
    summary["tasks"] = len(tasks)
    if slice_id:
        summary["slice_id"] = slice_id
        summary["task_offset"] = task_offset
    summary["without_brake_comparison"] = (
        "V2.1 numbers cited from results/v21_* (not rerun here)"
    )
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    for key, stats in sorted(summary["frontier"].items()):
        print(plain_summary_line(key, stats))
    cc = summary["cipher_control"]
    if cc:
        print(f"cipher control: failed at L1 on all {cc['units']} tasks = "
              f"{cc['all_failed_at_L1']}")
    return summary


def merge_expB_slices(results_dir):
    """Concatenate results/expB/slices/slice_*.jsonl, recompute the
    per-style×mode frontier from ALL units, and write the combined
    results/expB/ledger.jsonl + summary.json (deterministic, sorted)."""
    slices_dir = os.path.join(results_dir, "expB", "slices")
    if not os.path.isdir(slices_dir):
        raise ExpABError(f"no slices directory at {slices_dir}")
    names = sorted(n for n in os.listdir(slices_dir)
                   if re.fullmatch(r"slice_.+\.jsonl", n))
    if not names:
        raise ExpABError(f"no slice_*.jsonl files in {slices_dir}")
    records, units, cipher_units = [], [], []
    for name in names:
        with open(os.path.join(slices_dir, name), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append(rec)
                if rec.get("type") == "b_unit":
                    if rec.get("cipher_control"):
                        cipher_units.append(rec)
                    else:
                        units.append(rec)
    records.sort(key=lambda r: json.dumps(r, sort_keys=True,
                                          ensure_ascii=False))
    ledger_path = os.path.join(results_dir, "expB", "ledger.jsonl")
    with open(ledger_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True,
                                ensure_ascii=False) + "\n")
    summary = summarize_expB(units, cipher_units)
    summary["run_id"] = "expB_merged"
    summary["tasks"] = len({u["task_id"] for u in units})
    summary["slices_merged"] = names
    summary["without_brake_comparison"] = (
        "V2.1 numbers cited from results/v21_* (not rerun here)"
    )
    with open(os.path.join(results_dir, "expB", "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    for key, stats in sorted(summary["frontier"].items()):
        print(plain_summary_line(key, stats))
    cc = summary["cipher_control"]
    if cc:
        print(f"cipher control: failed at L1 on all {cc['units']} tasks = "
              f"{cc['all_failed_at_L1']}")
    return summary


# ==========================================================================
# CLI
# ==========================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="exp", required=True)
    for name in ("expa", "expb", "expb-merge"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
        p.add_argument("--max-seconds", type=float,
                       default=DEFAULT_MAX_SECONDS,
                       help="wall-clock guard on all-loops stages "
                            "(default %(default)s)")
        p.add_argument("--results-dir", default=os.path.join(_ROOT, "results"))
        p.add_argument("--stub", action="store_true",
                       help="declared deterministic stub client (smoke)")
        if name == "expa":
            p.add_argument("--clawd-dir", default=CLAWD_DIR)
            p.add_argument("--dist-dir", default=OPENCLAW_DIST)
            p.add_argument("--quiz-per-file", type=int, default=9,
                           help="8-10 per file -> 48-60 questions "
                                "(default %(default)s)")
        if name == "expb":
            p.add_argument("--tasks-dir", default=None)
            p.add_argument("--task-limit", type=int, default=None,
                           help="cap tasks (keeps call volume within the "
                                "SPEC <1k budget on full runs)")
            p.add_argument("--task-offset", type=int, default=0,
                           help="run tasks[offset:offset+limit] of the "
                                "FULL task list (parallel slicing)")
            p.add_argument("--slice-id", default=None,
                           help="write results/expB/slices/slice_<id>.* "
                                "instead of the single ledger/summary")
        if name == "expb-merge":
            pass  # --results-dir already added for all subcommands above
    args = parser.parse_args(argv)

    if args.exp == "expb-merge":
        try:
            merge_expB_slices(args.results_dir)
        except ExpABError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    stub = args.stub or os.environ.get("TCB_EXPAB_STUB") == "1"
    client = make_stub_client() if stub else default_client

    try:
        if args.exp == "expa":
            run_expA(client, args.model, args.max_tokens, args.results_dir,
                     clawd_dir=args.clawd_dir, dist_dir=args.dist_dir,
                     max_seconds=args.max_seconds,
                     quiz_per_file=args.quiz_per_file)
        else:
            run_expB(client, args.model, args.max_tokens, args.results_dir,
                     tasks_dir=args.tasks_dir, max_seconds=args.max_seconds,
                     task_limit=args.task_limit,
                     task_offset=args.task_offset,
                     slice_id=args.slice_id)
    except DeadlineExceeded as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ExpABError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
