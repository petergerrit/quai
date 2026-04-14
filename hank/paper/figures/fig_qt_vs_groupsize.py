"""Generate fig:qt_vs_groupsize.pdf — QT/Q_opt vs |C| for every Tier-1
(group, extension) pair. Uses the d3_tier1 summary data directly.

Renders with LaTeX fonts (requires system latex) for consistency with the
paper body. Two-column figure width.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LaTeX-style fonts; works with the pgf backend or via usetex=True.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# Data from sweep_runs/d3_tier1_summary.txt — Tier 1 @ t=5.
# Keys are element counts (|C|); group labels use Sigma(m×3) where |C| = 3m.
data = {
    216:  {"size": 216,  "m": 72,
           "HV(1,1,0) Campbell": 13.750,
           "HV(1,2,0) Eq(27)":  13.750,
           "HV(2,1,0)":          7.755,
           r"$P(2\pi/9)$":      43.208,
           r"$P(\pi/9)$":      175.561,
           r"$P(2\pi/5)$":      12.681,
           "rnd_batch1":         3.885,
           "rnd_batch2":         3.960,
           "Popt":               2.692},
    648:  {"size": 648,  "m": 216,
           "HV(1,1,0) Campbell": 16.561,
           "HV(1,2,0) Eq(27)":  16.561,
           "HV(2,1,0)":          9.340,
           r"$P(2\pi/9)$":      52.039,
           r"$P(\pi/9)$":      211.443,
           r"$P(2\pi/5)$":      15.273,
           "rnd_batch1":         3.369,
           "rnd_batch2":         3.565,
           "Popt":               2.544},
    1080: {"size": 1080, "m": 360,
           "HV(1,1,0) Campbell": 17.867,
           "HV(1,2,0) Eq(27)":  17.867,
           "HV(2,1,0)":         12.419,
           r"$P(2\pi/9)$":      56.145,
           r"$P(\pi/9)$":      228.127,
           r"$P(2\pi/5)$":      16.478,
           "rnd_batch1":         3.308,
           "rnd_batch2":         3.096,
           "Popt":               2.495},
}

sizes_list = [216, 648, 1080]

ext_style = [
    ("HV(1,1,0) Campbell",      dict(marker="o",  color="#1f77b4",
                                     label=r"$\mathrm{HV}(1,1,0)$")),
    ("HV(1,2,0) Eq(27)",        dict(marker="s",  color="#ff7f0e",
                                     label=r"$\mathrm{HV}(1,2,0)$")),
    ("HV(2,1,0)",               dict(marker="D",  color="#2ca02c",
                                     label=r"$\mathrm{HV}(2,1,0)$")),
    (r"$P(2\pi/9)$",             dict(marker="^",  color="#d62728",
                                     label=r"$P(2\pi/9)$")),
    (r"$P(\pi/9)$",              dict(marker="v",  color="#9467bd",
                                     label=r"$P(\pi/9)$")),
    (r"$P(2\pi/5)$",             dict(marker="<",  color="#8c564b",
                                     label=r"$P(2\pi/5)$")),
]

# Two-column figure: 7 in × 4 in, with a dedicated legend panel on the right.
fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(7.0, 4.0),
                                    gridspec_kw={"width_ratios": [3.2, 1.0]})

# Kesten-McKay floor
popt = [data[s]["Popt"] for s in sizes_list]
ax.plot(sizes_list, popt, "--", color="0.3", lw=1.5,
        label=r"$Q_{\mathrm{opt}}$ (Kesten--McKay floor)")

# Fixed extensions
for key, style in ext_style:
    qt = [data[s][key] for s in sizes_list]
    ax.plot(sizes_list, qt, linestyle="-", markersize=9, linewidth=1.2, **style)

# Haar-random (stars), one per batch per group
rnd_label_shown = False
for s in sizes_list:
    for bkey in ("rnd_batch1", "rnd_batch2"):
        qt_val = data[s][bkey]
        if not rnd_label_shown:
            ax.plot(s, qt_val, marker="*", color="k", markersize=14, zorder=5,
                    label=r"Haar-random (best of 10, per batch)")
            rnd_label_shown = True
        else:
            ax.plot(s, qt_val, marker="*", color="k", markersize=14, zorder=5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$|\mathcal{C}|$")
ax.set_ylabel(r"$Q_T$ (best-sample, $t=5$)")
ax.set_xticks(sizes_list)
ax.set_xticklabels([r"$\Sigma(72{\times}3)$",
                    r"$\Sigma(216{\times}3)$",
                    r"$\Sigma(360{\times}3)$"])
ax.minorticks_off()
ax.set_yticks([2, 3, 5, 10, 20, 50, 100, 200])
ax.set_yticklabels([r"$2$", r"$3$", r"$5$", r"$10$",
                    r"$20$", r"$50$", r"$100$", r"$200$"])
ax.grid(True, which="both", alpha=0.3)

# Dedicated legend axis (no axis artifacts)
ax_legend.axis("off")
handles, labels = ax.get_legend_handles_labels()
ax_legend.legend(handles, labels, loc="center left", frameon=False,
                 fontsize=9, handlelength=1.8, handletextpad=0.6)

fig.tight_layout()

out = Path(__file__).resolve().parent / "fig_qt_vs_groupsize.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
