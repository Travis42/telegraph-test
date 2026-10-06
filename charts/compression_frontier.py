#!/usr/bin/env python3
"""Compression/auditability frontier chart (dark theme, site palette).

2026-10-05 correction: cablese position updated to meter-basis savings
(both stylings on provider meters: lowercase 40-49%, default 25-34%).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, FG, ACC, DIM = "#1a1b2e", "#dadadb", "#8f7bd8", "#5a5b6e"

fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=170)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# naive tradeoff: more compression assumed to cost auditability
xs = np.linspace(0.045, 1.02, 300)
ax.plot(xs, xs ** 0.5 * 0.97, ls="--", color=DIM, lw=1.3, alpha=0.8)
ax.text(0.52, 0.72, "assumed tradeoff", color=DIM, fontsize=9,
        rotation=17, ha="center", style="italic")

# Cablese occupies TWO verified positions: default ALL-CAPS styling
# (25-34% meter savings across families) and the lowercase instruction
# (40-49%). Casing is free tokenizer efficiency; the register is constant.
cablese_pts = [(0.70, 0.90, "Cablese (default styling)"),
               (0.55, 0.94, "Cablese (lowercase)")]
for x, y, label in cablese_pts:
    ax.scatter([x], [y], s=170, color=ACC, zorder=3,
               edgecolor=FG, linewidths=1.2)
    ax.annotate(label, (x, y), xytext=(x, y + 0.045),
                color=FG, fontsize=10, weight="bold", ha="center")

ax.annotate("", xy=(0.565, 0.935), xytext=(0.685, 0.902),
            arrowprops=dict(arrowstyle="->", color=ACC, lw=1.2, alpha=0.85))
ax.text(0.625, 0.845, "one instruction word", color=ACC, fontsize=8.5,
        ha="center", style="italic")

points = [
    ("Plaintext",        1.00, 0.97, DIM,  (0, -16)),
    ("BabelTele\n(published claim)", 0.28, 0.30, DIM, (10, 4)),
    ("Emergent\nprotocols", 0.17, 0.20, DIM, (8, 2)),
    ("Raw\nembeddings",  0.05, 0.05, DIM, (6, 4)),
]
for label, x, y, color, (dx, dy) in points:
    ax.scatter([x], [y], s=90, color=color, zorder=3)
    ax.annotate(label, (x, y), xytext=(x + dx * 0.012, y + dy * 0.012),
                color=DIM, fontsize=9, ha="center")

ax.annotate("40-49% lowercase / 25-34% default\non the providers' own meters",
            xy=(0.55, 0.94), xytext=(0.42, 0.66),
            color=ACC, fontsize=9.5, ha="center",
            arrowprops=dict(arrowstyle="-", color=ACC, lw=1.0, alpha=0.7))
ax.text(0.10, 0.62, "external claims shown are\nlength-basis, unverified on meters",
        color=DIM, fontsize=7.5, style="italic", ha="left")

ax.set_xlim(0, 1.06)
ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("Relative record size  (provider-meter tokens; lower = more compressed)",
              color=FG, fontsize=10)
ax.set_ylabel("Auditability (human-readable, stable, decodable)",
              color=FG, fontsize=10)
ax.tick_params(colors=DIM)
for spine in ax.spines.values():
    spine.set_color(DIM)

ax.set_title("The compression/auditability frontier, meter-basis",
             color=FG, fontsize=12, pad=12)

fig.tight_layout()
fig.savefig("charts/compression-frontier.png", facecolor=BG)
print("wrote charts/compression-frontier.png")
