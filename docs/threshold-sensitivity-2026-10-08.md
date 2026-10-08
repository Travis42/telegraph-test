# Threshold sensitivity sweep (2026-10-08)

Question raised by a reader: what settled the 0.8 overlap cutoff, and how sensitive are recovery ratios to it?

**Provenance of 0.8:** design inheritance, not calibration. `ab/grade.py` (grader v2, 2026-10-03) sets `ANCHOR_THRESHOLD = MATCH_THRESHOLD = 0.8`, reusing the containment formula |A∩B|/min(|A|,|B|) from the baseline machinery. One value gates item authoring (expected answers ≥0.8 extractable from passage) and grading.

**Sweep:** recomputed from `data/runs/crossmodel_xm4/ledger.jsonl` — 9,716 stored `match_score` values, re-thresholded, zero API calls. Recovery ratios (cablese / plain, n=694 paired questions per panel):

| threshold | gemma-26b | gemma-31b | nemotron | qwen |
|---|---|---|---|---|
| 0.50 | 1.026 | 1.020 | 1.008 | 1.028 |
| 0.60 | 1.043 | 1.053 | 0.993 | 1.043 |
| 0.70 | 1.052 | 1.082 | 0.998 | 1.072 |
| 0.75 | 1.054 | 1.080 | 0.998 | 1.070 |
| 0.80 (published) | 1.072 | 1.093 | 1.008 | 1.098 |
| 0.85 | 1.064 | 1.095 | 1.010 | 1.099 |
| 0.90 | 1.060 | 1.090 | 1.008 | 1.091 |

**Findings:** every ratio stays ≥0.99 across 0.5–0.9; max swing ±0.04 vs published; panel ordering never flips; 0.8 is not a favorable corner (0.85–0.9 score same or higher for three of four panels). The cutoff shifts absolute accuracy but cancels in paired ratios — parity is cutoff-independent.
