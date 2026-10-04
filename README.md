# The Telegraph Test

**Can language models write in telegraphese — and be understood?**

Instruct an LLM to write its records in *telegraphese* (drop articles and filler, abbreviate, keep every fact, number, and proper noun) and it emits **~55% fewer output tokens**. The evidence in this repo says downstream models — including *other model families* — read those compressed records **as well as or better than** plain-English records.

## See it in 30 seconds

A real pair from the runs (a passage about the 1859 *Royal Charter* wreck inquiry — the actual telegraph era):

**Plain record (259 tokens mean / this one: 2470 chars):**
> **Record of the Royal Charter Commission Text** — The text describes an official inquiry into the wreck of the Royal Charter, a steam clipper ship. A commission was appointed...

**Cablese record (117 tokens mean / this one: 1412 chars):**
> COMMISSION APPOINTED INQUIRE LOSS ROYAL CHARTER STEAM CLIPPER. SITTINGS LIVERPOOL FROM 6 FEBRUARY 1860 BEFORE MR TREMENHEERE ASSISTED CAPTAINS HARRIS...

Same facts. Half the words. Victorian telegraph operators called this register *telegraphese* (the cable operators' own word for their traffic was *cablese* — we keep it as the package name); they used it for the same reason it matters now: **metered words are expensive, and output tokens are the expensive half of every LLM bill.**

## Headline results (all deterministically graded — no LLM judges anywhere in the truth path)

| Measurement | Result |
|---|---|
| Token savings, telegraphese records (GLM-5.3-Flash) | **54.6%** (ledger-recomputed) |
| Foreign readers on GLM cablese (Gemma / Qwen / Nemotron / Gemma-26B) | recovery **1.00–1.10** (cablese ≥ plain, all pairs) |
| GLM reading foreign cablese (Gemma / Qwen / **GPT-5-mini**) | recovery **1.09 / 1.10 / 0.99** (parity at 27% savings) |
| In-family: answer-from-telegraphese vs answer-from-plain | 82.5% vs 75.8% (McNemar p=3e-06) |
| Decode answer → plain (per-answer transcription) | 0.86 ratio — real, honest cost |
| Decode record → plain, then answer (storage loop) | **1.08 ratio** (p=0.031); decoded archive stays 40% leaner |
| Per-model compressibility under identical instruction | 27% (GPT-5-mini) … 48% (Qwen) … 55% (GLM) — a property, not noise |

**Verdict: telegraphese is a shared register across model families, not one model's idiolect.** Run `python3 verify_headlines.py` to recompute every number above from the frozen data in `data/runs/` — zero API calls.

## Try it

**→ [Open the demo notebook in Colab](notebooks/telegraph_test.ipynb)** — frozen results replay (no key, no cost) + optional live mode (bring an OpenRouter key, compress your own text).

## Repo layout

- `ab/` — the harness: question generation & anchoring gates (`baseline`), the telegraphese experiment (`cablese`), cross-family matrix (`crossmodel`), record-decode loop (`record_decode`), **grader v2** — deterministic, passage-anchored grading (`grade`), retro-regrade tool (`regrade`), passage bank (`tasks2/`)
- `data/runs/` — frozen artifacts of the four definitive runs: full per-call ledgers, summaries, per-record graded verdicts
- `notebooks/` — the demo
- `docs/` — the findings narrative as it evolved (including the self-caught judge-inflation correction)
- `verify_headlines.py` — recomputes the README table from frozen data

## Reproduce

```bash
pip install pytest  # stdlib-only harness otherwise
python3 -m pytest tests/ -q      # 200+ offline tests, no network
python3 -m ab.regrade --run-dir data/runs/cablese_cbl2   # re-derive the v2 numbers from the ledger
```

New runs need an OpenAI-compatible endpoint (env contract in the module docstrings; the paper trail for every number is the ledger, not anyone's testimony).

## Method in one paragraph

50 passages, 24 stratified questions each; every question **anchored** — its expected answer must be deterministically extractable from the source passage, or it is rejected before being asked. Every comparison is **paired** (same questions, both conditions) and assessed with **McNemar's test** on the discordant pairs. Grading is pure code: fixed normalization, containment, anchor checks. An earlier version of this experiment used an LLM to judge its own grading edge cases; the models graded their own homework generously, we caught it, deleted every judge-derived number, and rebuilt grading deterministically. That story is in `docs/`. **No fidelity claim in this repo survives on a model's own testimony.**

## License

Code: Apache-2.0 (`LICENSE`). Data and benchmark artifacts: CC-BY-4.0 (`DATA_LICENSE.md`).

## Context

This benchmark accompanies the write-up *"The Telegraph Test: What 19th-Century Cable Codes Know About LLM Token Economics."* The name credits the telegraph operators who ran this experiment first, for the same economic reason, ~150 years ago.
