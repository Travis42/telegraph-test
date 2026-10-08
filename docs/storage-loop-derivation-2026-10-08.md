# Storage-loop "one-third" derivation (2026-10-08)

Question raised by a reader: what absolute token counts established "expansion gives back about a third of the savings"?

Basis: mean stored-text tokens (deterministic tokenization of record texts, `approx_tokens`), n=50 records, glm-5.3-flash writes and expands. Frozen in `data/runs/record_decode_rd1/summary.json` (`record_tokens`) over `cablese_cbl2` records:

- R-PLAIN: **259.9** mean tokens
- R-CABLESE: **116.5** mean tokens → initial savings 143.4 tokens (~55%)
- R-DECODED (expanded archive): **154.8** mean tokens → expansion consumed 38.3 of the 143.4 saved = **26.7% ≈ "about a third"**
- Decoded archive vs plain: 154.8/259.9 = 0.595 → still **~40% leaner than plaintext**

Separately, the expansion *call's* billing (`record_decode` model_call medians, thinking-default-on protocol of that run): input 157 tokens (the cablese record), output 708.5 billed. The post's storage-loop claim is archive arithmetic (stored text), deliberately not the call meter — write-side savings (40–49%) are the meter-measured numbers.

## Lifecycle accounting (added after author challenge, same day)

The one-third figure is stored-text arithmetic only. The expansion call is additive: ~157 input (cablese record + instruction) + ~155 output (~312 total; the thinking-on run billed 708 out). Full lifecycle vs writing plain, in output-equivalents at 4:1 pricing:

- plain: 260 out + ~65 per machine read
- cablese + one expansion: ~311 out-equiv + ~29 per machine read

Break-even ≈ 1–2 machine reads. On output tokens alone (272 vs 260) the loop never wins — its advantage is entirely cheaper machine reads (117 vs 260 input each). One-write-one-human-look with zero machine reads is a net loss; humans as primary reader → write plain.
