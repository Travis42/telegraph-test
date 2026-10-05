#!/usr/bin/env python3
"""Compression/auditability frontier chart (dark theme, site palette).

2026-10-05 correction: cablese position updated to meter-basis savings
(25-34% visible-token, provider meters). Character-count measures
suggest roughly 2x more; the chart annotates both.
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

points = [
    ("Plaintext",        1.00, 0.97, DIM,  (0, -16)),
    ("Cablese",          0.70, 0.92, ACC,  (0, 10)),
    ("BabelTele\n(published claim)", 0.28, 0.30, DIM, (10, 4)),
    ("Emergent\nprotocols", 0.17, 0.20, DIM, (8, 2)),
    ("Raw\nembeddings",  0.05, 0.05, DIM, (6, 4)),
]
for label, x, y, color, (dx, dy) in points:
    big = color == ACC
    ax.scatter([x], [y], s=170 if big else 90, color=color,
               zorder=3, edgecolor=FG if big else "none", linewidths=1.2)
    ax.annotate(label, (x, y), xytext=(x + dx * 0.012, y + dy * 0.012),
                color=FG if big else DIM, fontsize=10.5 if big else 9,
                weight="bold" if big else "normal", ha="center")

ax.annotate("25-34% on the API's own meter\n(character counts suggest ~2x)",
            xy=(0.70, 0.92), xytext=(0.66, 0.60),
            color=ACC, fontsize=9.5, ha="center",
            arrowprops=dict(arrowstyle="-", color=ACC, lw=1.0, alpha=0.7))

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
