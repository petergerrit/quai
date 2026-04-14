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


# (label, hits, mean_dist, cost_method)
_ROWS = [
    (r"$\mathrm{Clifford}{+}P(\pi/18)$",    34, 0.0104, "bfs"),
    (r"$\texttt{hurwitz}{+}P(\pi/18)$",     35, 0.0112, "bfs"),
    (r"$\mathrm{Clifford}{+}P(\pi/9)$",     27, 0.0120, "bfs"),
    (r"$\mathrm{Clifford}{+}P(2\pi/9)$",    27, 0.0122, "bfs"),
    (r"$\mathrm{Clifford}{+}P(\pi/8)$",      7, 0.0221, "bfs"),
    (r"$\mathrm{Clifford}{+}T_{24}$",        6, 0.0228, "bfs"),
    (r"$\mathrm{Clifford}{+}T_{12}$",        8, 0.0233, "bfs"),
    (r"$\texttt{hurwitz}{+}T_{12}$",         0, 0.0313, "bfs"),
    (r"$\mathrm{Clifford}{+}T\ (P(\pi/4))$", 5, 0.0833, "rs"),
]


_COLORS = {"bfs": "#1f77b4", "rs": "#d62728"}
_LABELS = {"bfs": r"\texttt{bfs\_estimate}", "rs": r"\texttt{rs\_exact}"}

# Manual label offset hints (dx multiplier on x-axis, absolute dy).
# Since x is log, dx is expressed as a multiplicative factor.
_LABEL_OFFSETS = {
    0: (1.08, 3.0),   # Clifford+P(π/18)      → top cluster
    1: (1.08, -1.5),  # hurwitz+P(π/18)       → top cluster, below 0
    2: (0.72, -0.5),  # Clifford+P(π/9)       → left of point
    3: (1.10, 0.5),   # Clifford+P(2π/9)      → right of point
    4: (1.10, 0.0),   # Clifford+P(π/8)
    5: (1.10, 1.5),   # Clifford+T_24         → up-right
    6: (1.10, -1.5),  # Clifford+T_12         → down-right
    7: (1.10, 0.0),   # hurwitz+T_12
    8: (0.60, 2.0),   # Clifford+T            → far left to clear its own zone
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for i, (label, hits, mean_dist, kind) in enumerate(_ROWS):
        ax.plot(mean_dist, hits, marker="o", markersize=9,
                color=_COLORS[kind], linestyle="none",
                markeredgecolor="black", markeredgewidth=0.5,
                zorder=3)
        x_mult, dy = _LABEL_OFFSETS.get(i, (1.10, 1.0))
        xtext = mean_dist * x_mult
        ytext = hits + dy
        ha = "left" if x_mult > 1.0 else "right"
        ax.annotate(label, xy=(mean_dist, hits),
                    xytext=(xtext, ytext),
                    fontsize=8, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="0.6",
                                    lw=0.4, shrinkA=2, shrinkB=4))

    ax.set_xscale("log")
    ax.set_xlabel(r"mean projective distance on $\mathcal{T}_{36}$")
    ax.set_ylabel(r"hits $/ 58$ (targets reached with $\varepsilon \leq 10^{-2}$)")
    ax.set_xlim(0.004, 0.22)
    ax.set_ylim(-6, 44)
    ax.grid(True, which="both", alpha=0.25)

    # Threshold line at hits=0 to separate "never converges" rows.
    ax.axhline(0, color="0.7", lw=0.8, ls="--")

    # Legend.
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=c, marker="o", markersize=9,
                      markeredgecolor="black", markeredgewidth=0.5,
                      linestyle="none", label=_LABELS[k])
        for k, c in _COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True,
              framealpha=0.95, edgecolor="0.8")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fig_lamm_sigma36.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
