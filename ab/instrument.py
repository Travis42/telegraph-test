#!/usr/bin/env python3
"""Hardened instrument layer for V2.1 (SPEC-V21.md §1).

Design rule: NO silent fallbacks anywhere.
- Every model answer is returned as `raw` VERBATIM; the caller persists it
  to the run JSONL before any parsing is trusted.
- `parse_json_block` NEVER defaults: it returns (None, error_string) on any
  failure; there is no code path that invents an empty result.
- A failed parse is surfaced as `parse_ok: False` plus a non-empty
  `parse_error` string, so "0 facts matched" (parse_ok True) and
  "extraction failed" (parse_ok False) are distinct, distinguishable states.
- A missing `usage` block is an API contract violation and raises
  InstrumentError rather than being papered over.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request


class InstrumentError(Exception):
    """Explicit instrument failure (API shape violation, unreachable API)."""


class TransientAPIError(InstrumentError):
    """429/5xx from the API — retryable. Carries retry_after when provided."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


API_URL_DEFAULT = "https://api.z.ai/api/paas/v4/chat/completions"
TEMPERATURE = 0


def default_client(payload):
    """Minimal stdlib HTTP client for the ZAI (OpenAI-compatible) endpoint.

    The endpoint URL is taken from TCB_ZAI_API_URL (coding endpoint) when
    set; the key must be in ZAI_API_KEY. Aborts loudly when unset.
    """
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        sys.exit(
            "error: ZAI_API_KEY is not set in the environment; "
            "export ZAI_API_KEY=<your key> and re-run."
        )
    url = os.environ.get("TCB_ZAI_API_URL", API_URL_DEFAULT)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        # TCB_TREAT_401_TRANSIENT=1: z.ai coding endpoint intermittently
        # returns false "token expired" 401s with a valid key (observed
        # 2026-09-30: same key fails one minute, passes the next, from the
        # same host). Retry only when explicitly opted in.
        transient_401 = (
            exc.code == 401
            and os.environ.get("TCB_TREAT_401_TRANSIENT") == "1"
        )
        if exc.code == 429 or transient_401 or 500 <= exc.code < 600:
            raise TransientAPIError(
                f"API returned HTTP {exc.code}: {detail}",
                retry_after=exc.headers.get("Retry-After") if exc.headers else None,
            )
        sys.exit(f"error: API returned HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"error: API unreachable: {exc}")


def parse_json_block(text):
    """Extract the outermost JSON array/object from a model answer.

    Returns (obj, None) on success or (None, error_string) on failure.
    NEVER returns a default value. Handles prose-wrapped and fenced
    (```...```) answers by locating the first '[' or '{' and matching its
    balancing close, honoring string escapes.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "model output is empty or not a string"
    start = None
    opener = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start, opener = i, ch
            break
    if start is None:
        return None, "no JSON array or object found in model output"
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
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate), None
                except json.JSONDecodeError as exc:
                    return None, f"JSON decode failed: {exc}"
    return None, "unbalanced JSON block (no matching closing bracket)"


def _extract_usage(response):
    if not isinstance(response, dict) or "usage" not in response:
        raise InstrumentError(
            f"API response has no usage block: {json.dumps(response)[:300]}"
        )
    usage = response["usage"]
    try:
        details = usage.get("completion_tokens_details") or {}
        return {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            # Billed-but-invisible thinking: absent in frozen cbl2-era
            # ledgers (extractor predates the field), present from now on.
            "reasoning_tokens": details.get("reasoning_tokens"),
        }
    except (KeyError, TypeError) as exc:
        raise InstrumentError(
            f"API usage block malformed ({exc!r}): {json.dumps(usage)[:300]}"
        ) from exc


def _call_with_backoff(client, payload, max_attempts=None):
    import os
    if max_attempts is None:
        max_attempts = int(os.environ.get("TCB_RETRY_ATTEMPTS", "6"))
    """Call client with exponential backoff + jitter on 429/5xx.

    Honors Retry-After when present. Raises the last exception otherwise.
    No silent fallback: a rate-limited call either succeeds or raises.
    """
    import random
    last = None
    for attempt in range(max_attempts):
        try:
            return client(payload)
        except TransientAPIError as exc:
            last = exc
            delay = min(2 ** attempt + random.uniform(0, 1.5), 60)
            if getattr(exc, "retry_after", None):
                delay = max(delay, float(exc.retry_after))
            time.sleep(delay)
    raise last


def call_model(client, model, system, user, max_tokens, parse=None):
    """One model call through the hardened instrument.

    Returns {"raw": verbatim answer, "parsed": obj_or_None,
             "parse_error": str_or_None, "usage": {...}, "latency_ms": float}.

    `raw` is always present and verbatim; the CALLER persists it before
    trusting any parsed value. If `parse` is None, parsed/parse_error are
    None (the call is free-text by design, e.g. transforms).
    """
    # zai/ prefix strips for the z.ai coding endpoint; other provider ids
    # (e.g. openrouter "deepseek/deepseek-v4.1-flash") pass through intact.
    payload = {
        "model": model[4:] if model.startswith("zai/") else model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if os.environ.get("TCB_REASONING_OFF") == "1":
        # zai coding endpoint ignores the OpenAI-compat `reasoning`
        # object (verified 2026-10-05: reasoning:{enabled:false} still
        # billed 168 reasoning tokens on a 4-token answer;
        # thinking.type=disabled bills 0). Use the native param for
        # zai/ models; OpenRouter keeps the reasoning object it honors.
        if model.startswith("zai/"):
            payload["thinking"] = {"type": "disabled"}
        else:
            # openrouter reasoning models can return content=null with
            # the answer in reasoning_content; opt-out instead.
            payload["reasoning"] = {"enabled": False}
    started = time.monotonic()
    response = _call_with_backoff(client, payload)
    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
    usage = _extract_usage(response)
    try:
        raw = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InstrumentError(
            f"API response has no choices[0].message.content ({exc!r}): "
            f"{json.dumps(response)[:300]}"
        ) from exc
    if not isinstance(raw, str):
        raise InstrumentError(
            f"model content is not a string: {raw!r:.200}"
        )
    parsed = None
    parse_error = None
    if parse is not None:
        parsed, parse_error = parse(raw)
    return {
        "raw": raw,
        "parsed": parsed,
        "parse_error": parse_error,
        "usage": usage,
        "latency_ms": latency_ms,
    }


COVERAGE_SYSTEM = (
    "You check fact coverage. You are given a context and then a numbered "
    "fact list. Reply ONLY with a JSON array of the numbers (integers) of "
    "the facts that ARE stated in the context. An empty array [] means "
    "none of the facts are stated. Output nothing else."
)


def coverage_user(context, facts):
    listing = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
    return f"Context:\n{context}\n\nFacts:\n{listing}"


def fact_check(client, model, context, facts, max_tokens=512):
    """FactCheck result type per SPEC-V21 §1:

    {"facts_total": n, "facts_matched": [ids], "parse_ok": bool, "raw": "..."}

    parse_ok=False (with parse_error set) marks extraction failure;
    parse_ok=True with facts_matched==[] marks a checked zero. These are
    distinct states and are never conflated. The raw answer is carried
    verbatim so the caller can persist it before trusting the ids.
    """
    facts = list(facts)
    result = call_model(
        client,
        model,
        COVERAGE_SYSTEM,
        coverage_user(context, facts),
        max_tokens,
        parse=parse_json_block,
    )
    check = {
        "facts_total": len(facts),
        "facts_matched": [],
        "parse_ok": False,
        "raw": result["raw"],
        "parse_error": None,
    }
    if result["parse_error"] is not None:
        check["parse_error"] = result["parse_error"]
    elif not isinstance(result["parsed"], list):
        check["parse_error"] = (
            f"parsed JSON is {type(result['parsed']).__name__}, expected array"
        )
    elif not all(isinstance(x, int) and not isinstance(x, bool) for x in result["parsed"]):
        check["parse_error"] = "parsed array contains non-integer entries"
    elif any(x < 1 or x > len(facts) for x in result["parsed"]):
        check["parse_error"] = "parsed array contains fact ids out of range"
    else:
        check["parse_ok"] = True
        check["facts_matched"] = sorted(set(result["parsed"]))
    check["usage"] = result["usage"]
    check["latency_ms"] = result["latency_ms"]
    return check


def recall_of(check):
    """Recall from a FactCheck; None when the check failed to parse
    (explicitly NOT treated as 0.0 — that would be a silent fallback)."""
    if not check.get("parse_ok"):
        return None
    if check["facts_total"] == 0:
        return None
    return len(check["facts_matched"]) / check["facts_total"]
