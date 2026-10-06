#!/usr/bin/env python3
"""Recompute the README headline numbers from the frozen run artifacts.

Zero API calls — pure recompute from data/runs/. Exits non-zero on any mismatch
with the README table (within rounding tolerance).
"""
import json, math, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "data", "runs")
TOL = 0.005  # half a percent tolerance for rounding in the README

checks, failures = [], []


def check(name, got, claim):
    ok = got is not None and abs(got - claim) <= TOL
    checks.append((name, got, claim, ok))
    if not ok:
        failures.append(name)


def load(path):
    with open(os.path.join(RUNS, path), encoding="utf-8") as f:
        return json.load(f)


# --- 1. cbl2 v2 regrade: savings + in-family ratios -------------------------
rg = load("cablese_cbl2/regrade_v2.json")
check("cbl2 savings_pct_v2 (README 54.6%)", rg["record_tokens_v2"]["savings_pct_v2"] / 100, 0.546)
check("recovery_ratio_v2 (README 1.089)", rg["recovery_ratio_v2"], 0.825 / 0.758)
check("A-FROM-PLAIN acc (README 75.8%)", rg["conditions"]["A-FROM-PLAIN"]["v2_accuracy"], 0.758)
check("A-FROM-CABLESE acc (README 82.5%)", rg["conditions"]["A-FROM-CABLESE"]["v2_accuracy"], 0.825)
check("decode_ratio_v2 (README 0.862)", rg["decode_ratio_v2"], 0.788 / 0.915)

# --- 2. crossmodel xm4: family ratios + per-model savings -------------------
xm = load("crossmodel_xm4/summary.json")
readers = {"R1": 1.095, "R2": 1.097, "R3": 1.004, "R4": 1.069}
for rid, claim in readers.items():
    v = xm["direction_a"][rid]["recovery_ratio"]
    check(f"xm4 reader {rid} ({xm['direction_a'][rid]['model']})", v, claim)
writers = {"W1": 1.087, "W2": 1.099, "W3": 0.992}
for wid, claim in writers.items():
    v = xm["direction_b"][wid]["recovery_ratio"]
    check(f"xm4 writer {wid} ({xm['direction_b'][wid]['model']}) ratio", v, claim)
check("xm4 W3 gpt-5-mini savings (README 27%)", xm["direction_b"]["W3"]["savings_pct"] / 100, 0.267)
check("xm4 W2 qwen savings (README 48%)", xm["direction_b"]["W2"]["savings_pct"] / 100, 0.475)

# --- 3. record_decode rd1: the storage loop ---------------------------------
rd = load("record_decode_rd1/summary.json")
check("rd1 A-FROM-DECODED (README 81.6%)", rd["conditions"]["A-FROM-DECODED"]["overall_raw"]["accuracy"], 0.816)
check("rd1 decoded_recovery_ratio (README 1.08)", rd["decoded_recovery_ratio"], 1.08)
check("rd1 decode_ratio_rec (README 0.976)", rd["decode_ratio_rec"], 0.976)
rt = rd["record_tokens"]
check("rd1 decoded stays 40% leaner (README 0.596)", rt["decoded_vs_plain_ratio"], 0.596)

# --- 4. meter-basis savings (2026-10-05 probes + xm4 usage recompute) -------
import statistics as _st
def _med(vals):
    vals = [v for v in vals if v is not None]
    return _st.median(vals) if vals else None

glm = load("billing_reasoning_off_20261005/summary.json")
check("GLM meter savings, thinking off (README 33.9%)",
      glm["savings_pct_billing_basis"] / 100, 0.339)

g5 = load("billing_gpt5mini_effortlow_20261005/summary.json")
check("gpt-5-mini meter savings, effort=low (README -99.8%)",
      g5["savings_pct_billing_basis"] / 100, -0.998)

def _probe_visible(path):
    vis = {"R-PLAIN": [], "R-CABLESE": []}
    for line in open(path):
        r = json.loads(line)
        if r.get("type") != "billing_record":
            continue
        u = r["usage"]
        vis[r["condition"]].append(u["completion_tokens"] - (u.get("reasoning_tokens") or 0))
    return vis
v5 = _probe_visible(os.path.join(RUNS, "billing_gpt5mini_effortlow_20261005/ledger.jsonl"))
check("gpt-5-mini visible-text savings (README 22.6%)",
      1 - _med(v5["R-CABLESE"]) / _med(v5["R-PLAIN"]), 0.226)

xm_billed = {}
for line in open(os.path.join(RUNS, "crossmodel_xm4/ledger.jsonl")):
    r = json.loads(line)
    if r.get("type") != "model_call" or not r.get("call", "").startswith("record_"):
        continue
    xm_billed.setdefault((r.get("model"), r["call"]), []).append(r["usage"]["completion_tokens"])
gm = _med(xm_billed[("google/gemma-4-31b-it", "record_R-CABLESE")]) / _med(xm_billed[("google/gemma-4-31b-it", "record_R-PLAIN")])
check("gemma meter savings (README 25.0%)", 1 - gm, 0.250)
qm = _med(xm_billed[("qwen/qwen3.8-27b", "record_R-CABLESE")]) / _med(xm_billed[("qwen/qwen3.8-27b", "record_R-PLAIN")])
check("qwen meter savings (README 29.8%)", 1 - qm, 0.298)

lc = load("billing_glm_lowercase_20261005/summary.json")
check("GLM meter savings, lowercase cablese (README 48.4%)",
      lc["savings_pct_billing_basis"] / 100, 0.484)

lc = load("lc_readability_20261005/summary.json")
check("GLM lowercase readability vs plain (README 1.089)",
      lc["recovery_vs_plain"], 1.089)
for tag, claim in [("gemma", 1.018), ("qwen", 1.0144), ("gpt5mini", 1.0362)]:
    s = load(f"lc_read_{tag}_20261006/summary.json")
    check(f"{tag} reads GLM lowercase records (README {claim})",
          s["recovery_vs_plain"], claim)
for tag, claim in [("gemma", 0.404), ("qwen", 0.489)]:
    s = load(f"billing_{tag}_lowercase_20261005/summary.json")
    check(f"{tag} lowercase writer meter savings (README {claim:.1%})".replace("40.4%", "40.4%"),
          s["savings_pct_billing_basis"] / 100, claim)

def _probe_visible(path):
    vis = {"R-PLAIN": [], "R-CABLESE": []}
    for line in open(path):
        r = json.loads(line)
        if r.get("type") != "billing_record":
            continue
        u = r["usage"]
        vis[r["condition"]].append(u["completion_tokens"] - (u.get("reasoning_tokens") or 0))
    return vis
vl = _probe_visible(os.path.join(RUNS, "billing_gpt5mini_lowercase_20261005/ledger.jsonl"))
check("gpt-5-mini lowercase visible-text savings (README 17.7%)",
      1 - _med(vl["R-CABLESE"]) / _med(vl["R-PLAIN"]), 0.177)

# --- report ------------------------------------------------------------------
print(f"{'check':55s} {'computed':>10s} {'README':>8s}  ok")
print("-" * 84)
for name, got, claim, ok in checks:
    g = f"{got:.4f}" if got is not None else "MISSING"
    print(f"{name:55s} {g:>10s} {claim:8.3f}  {'OK' if ok else 'FAIL'}")
print("-" * 84)
print(f"{len(checks) - len(failures)}/{len(checks)} checks pass")
if failures:
    print("FAILURES:", ", ".join(failures))
    sys.exit(1)
print("All README headline numbers reproduce from frozen artifacts.")
