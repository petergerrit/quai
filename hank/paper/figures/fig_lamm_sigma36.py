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

# Manual label offset hints (dx, dy) per row index to avoid overlaps.
_LABEL_OFFSETS = {
    0: (0.0012, 2.2),   # Clifford+P(π/18)
    1: (0.0012, 0.8),   # hurwitz+P(π/18)
    2: (-0.0009, -1.6), # Clifford+P(π/9)
    3: (0.0015, -1.4),  # Clifford+P(2π/9)
    4: (0.0020, -0.7),  # Clifford+P(π/8)
    5: (-0.0030, 1.3),  # Clifford+T_24
    6: (0.0020, 1.6),   # Clifford+T_12
    7: (-0.0028, -1.8), # hurwitz+T_12
    8: (-0.025, 2.8),   # Clifford+T
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for i, (label, hits, mean_dist, kind) in enumerate(_ROWS):
        ax.plot(mean_dist, hits, marker="o", markersize=9,
                color=_COLORS[kind], linestyle="none",
                markeredgecolor="black", markeredgewidth=0.5,
                zorder=3)
        dx, dy = _LABEL_OFFSETS.get(i, (0.001, 1.0))
        ax.annotate(label, xy=(mean_dist, hits),
                    xytext=(mean_dist + dx, hits + dy),
                    fontsize=8, ha="left", va="center")

    ax.set_xscale("log")
    ax.set_xlabel(r"mean projective distance on $\mathcal{T}_{36}$")
    ax.set_ylabel(r"hits $/ 58$ (targets reached with $\varepsilon \leq 10^{-2}$)")
    ax.set_xlim(0.009, 0.13)
    ax.set_ylim(-4, 42)
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
