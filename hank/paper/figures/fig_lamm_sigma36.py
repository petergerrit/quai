"""Visualize the DSA $\\Sigma(36\\times3)$ target-family coverage table.

Plots hits/58 versus mean projective distance (log) for each
(base, extension) pair in ``tab:lamm_sigma36``. The Clifford+T anchor
sits bottom-right (few hits, largest mean distance) and the
rational-phase winners cluster top-left (many hits, smallest mean
distance). Markers are colored by the cost method (``rs_exact`` vs
``bfs_estimate``) so certified-synthesis rows stand out.

Data here are hardcoded from ``tab:lamm_sigma36`` in the paper. Update
the dict below when the table updates.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


# (label, hits, mean_dist, cost_method).
# Short labels chosen to fit tight layouts.
_ROWS = [
    (r"Cl$+P(\pi/18)$",           34, 0.0104, "bfs"),
    (r"hu$+P(\pi/18)$",           35, 0.0112, "bfs"),
    (r"Cl$+P(\pi/9)$",            27, 0.0120, "bfs"),
    (r"Cl$+P(2\pi/9)$",           27, 0.0122, "bfs"),
    (r"Cl$+P(\pi/8)$",             7, 0.0221, "bfs"),
    (r"Cl$+T_{24}$",               6, 0.0228, "bfs"),
    (r"Cl$+T_{12}$",               8, 0.0233, "bfs"),
    (r"hu$+T_{12}$",               0, 0.0313, "bfs"),
    (r"Cl$+P(\pi/4)$",             5, 0.0833, "rs"),
]

_COLORS = {"bfs": "#1f77b4", "rs": "#d62728"}
_LABELS = {"bfs": r"\texttt{bfs}", "rs": r"\texttt{rs}"}


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.6))  # single-column width

    # Data points.
    for label, hits, mean_dist, kind in _ROWS:
        ax.plot(mean_dist, hits, marker="o", markersize=7,
                color=_COLORS[kind], linestyle="none",
                markeredgecolor="black", markeredgewidth=0.4,
                zorder=3)

    # Label placement: explicit (x_abs, y_abs, ha, va) per point.
    # Chosen so no two labels overlap and all leader lines are clean.
    label_positions = [
        # (x_abs, y_abs, ha, va)
        (0.0085, 37.0, "right", "center"),   # Cl+P(π/18)   → top-left
        (0.0130, 35.5, "left",  "center"),   # hu+P(π/18)   → top-right
        (0.0098, 24.0, "right", "center"),   # Cl+P(π/9)
        (0.0148, 27.0, "left",  "center"),   # Cl+P(2π/9)   → right
        (0.0285, 11.0, "left",  "center"),   # Cl+P(π/8)    → right
        (0.0175,  3.5, "right", "center"),   # Cl+T_24      → lower-left
        (0.0295,  5.0, "left",  "center"),   # Cl+T_12      → right
        (0.0250, -4.0, "center","top"),      # hu+T_12      → below its point (0 hits)
        (0.0600,  8.5, "right", "center"),   # Cl+T         → left of its point
    ]

    for (label, _hits, _mean, _kind), (xt, yt, ha, va) in zip(_ROWS, label_positions):
        mean_dist = _mean
        hits = _hits
        ax.annotate(label, xy=(mean_dist, hits), xytext=(xt, yt),
                    fontsize=7.5, ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", color="0.5",
                                    lw=0.35, shrinkA=1, shrinkB=3))

    ax.set_xscale("log")
    ax.set_xlabel(r"mean projective distance on $\mathcal{T}_{36}$")
    ax.set_ylabel(r"hits / $58$")
    ax.set_xlim(0.006, 0.14)
    ax.set_ylim(-8, 44)
    ax.grid(True, which="both", alpha=0.22)
    ax.axhline(0, color="0.7", lw=0.6, ls="--")

    # Legend.
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=c, marker="o", markersize=7,
                      markeredgecolor="black", markeredgewidth=0.4,
                      linestyle="none", label=_LABELS[k])
        for k, c in _COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True,
              framealpha=0.95, edgecolor="0.8", fontsize=7)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fig_lamm_sigma36.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
