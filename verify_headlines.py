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
