"""Ternary plot of δ on the SU(3) maximal torus, rendered with python-ternary.

Reads a ``q-panel`` cache produced by ``swiftbot.cli q-panel --panel
d3_rnd_diag`` and draws three ternary panels (one per Σ-series base
group). Each data point is a random SU(3) diagonal
``diag(e^{iθ₁}, e^{iθ₂}, e^{iθ₃})``; its position in the triangle is
the barycentric normalization of ``(θ₁, θ₂, θ₃)`` reduced to
``[0, 2π)``, colored by best-sample ``δ``. Structured-diagonal overlays
(``P(2π/9), P(π/9), P(2π/5)``) are marked as stars when their
fingerprints are present.

The ternary library handles: equilateral triangle boundary, labeled
corners, tick marks and labels along each edge, internal grid lines at
regular multiples of the scale. All data is mapped to the standard
simplex (a, b, c) with a+b+c = 1 via ``θ_i / Σθ_j`` after wrapping each
θ to ``[0, 2π)``; entries with Σθ = 0 are omitted (degenerate point).

Usage::

    python fig_d3_rnd_diag_ternary.py --db sweep_runs/d3_rnd_diag/cache.db

Requires ``python-ternary`` (pip install python-ternary).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ternary

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from swiftbot.cli import _d3_rnd_diag_panel, _d3_survey_panel    # noqa: E402
from swiftbot.stages.s3_efficiency import extension_fingerprint  # noqa: E402


_GROUP_SIZE = {"S216": 216, "S648": 648, "S1080": 1080}
# The S216 and S648 δ values are bit-identical for every random SU(3)
# diagonal at t=5, so we merge them into one panel. S1080 is distinct.
_PANEL_GROUPS = ["S216", "S1080"]
_SIGMA_LABELS = {
    "S216":  r"$\Sigma(72{\times}3)$ \& $\Sigma(216{\times}3)$",
    "S1080": r"$\Sigma(360{\times}3)$",
}

# ternary scale: three barycentric coords sum to SCALE. We use SCALE = 2
# so the triangle is labeled 0 → 2π (in units of π), matching the physical
# range of each θᵢ. Data are normalized so Σ coord = SCALE, i.e., each
# plotted coordinate is θᵢ · 2 / Σθⱼ, read as "amount of π this axis
# carries if the total θ-sum were redistributed to the unit simplex".
SCALE = 2.0


def to_barycentric(theta1: float, theta2: float, theta3: float) -> tuple[float, float, float] | None:
    """Reduce θ to ``[0, 2π)`` and divide by the (non-zero) sum.

    Returns None when Σθ = 0 (all angles are 0 or hit the torus origin);
    these are omitted from the plot rather than placed at the centroid.
    """
    two_pi = 2 * np.pi
    a = theta1 % two_pi
    b = theta2 % two_pi
    c = theta3 % two_pi
    s = a + b + c
    if s <= 1e-9:
        return None
    return (a / s) * SCALE, (b / s) * SCALE, (c / s) * SCALE


def load_best_delta(db_path: Path, fingerprint: str, group_size: int) -> float | None:
    c = sqlite3.connect(str(db_path))
    try:
        row = c.execute(
            """
            SELECT MIN(q.delta)
            FROM qt_results q
            JOIN groups g ON q.target_key = g.group_key
            WHERE q.ext_fingerprint = ? AND g.size = ?
            """,
            (fingerprint, group_size),
        ).fetchone()
    finally:
        c.close()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True,
                    help="SQLite cache from q-panel --panel d3_rnd_diag.")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "fig_d3_rnd_diag_ternary.pdf",
                    help="Output PDF path.")
    ap.add_argument("--no-latex", action="store_true",
                    help="Skip LaTeX rendering (faster; use for CI).")
    args = ap.parse_args()

    if not args.no_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.latex.preamble": r"\usepackage{amsmath}",
        })

    # Panel definitions from the CLI so fingerprints line up with the cache.
    diag_panel = _d3_rnd_diag_panel()
    survey_panel = _d3_survey_panel()

    # Structured overlays: name → (θ₁, θ₂, θ₃) triples used by the survey panel.
    survey_angles = {
        "P(2pi/9)": (0.0, 2 * np.pi / 9, (-2 * np.pi / 9) % (2 * np.pi)),
        "P(pi/9)":  (0.0, np.pi / 9,     (-np.pi / 9) % (2 * np.pi)),
        "P(2pi/5)": (0.0, 2 * np.pi / 5, (-2 * np.pi / 5) % (2 * np.pi)),
    }

    # Collect (point, δ) data per group + global δ range. Only the groups
    # we actually plot are populated here (S216 stands in for the
    # S216=S648 degeneracy).
    data: dict[str, list[tuple[tuple[float, float, float], float]]] = {
        g: [] for g in _PANEL_GROUPS
    }
    all_delta: list[float] = []
    for name, spec in diag_panel:
        fp = extension_fingerprint(spec)
        p = spec.params
        bary = to_barycentric(p["theta1"], p["theta2"], p["theta3"])
        if bary is None:
            continue
        for g in _PANEL_GROUPS:
            d = load_best_delta(args.db, fp, _GROUP_SIZE[g])
            if d is None:
                continue
            data[g].append((bary, d))
            all_delta.append(d)
    if not all_delta:
        raise SystemExit(
            f"No d3_rnd_diag rows in {args.db}. "
            "Did you run `swiftbot q-panel --panel d3_rnd_diag` first?"
        )
    vmin, vmax = min(all_delta), max(all_delta)

    # Structured overlays.
    overlay_bary: dict[str, tuple[float, float, float]] = {}
    for name, triple in survey_angles.items():
        b = to_barycentric(*triple)
        if b is not None:
            overlay_bary[name] = b

    # Two panels + colorbar. Square subplot axes so the ternary triangle
    # renders equilateral (otherwise the default aspect ratio from the
    # figsize skews it).
    fig = plt.figure(figsize=(9.0, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.04], wspace=0.18)

    cmap = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for idx, g in enumerate(_PANEL_GROUPS):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_aspect("equal")      # equilateral triangle
        _, tax = ternary.figure(ax=ax, scale=SCALE)
        tax.boundary(linewidth=1.2)
        tax.gridlines(color="0.6", multiple=0.5, linewidth=0.5)
        tax.set_title(_SIGMA_LABELS[g], pad=26)

        # Edge ticks every π/2, labeled in π units: 0, 0.5π, 1.0π, 1.5π, 2π.
        tax.ticks(axis="lbr", linewidth=0.8, multiple=0.5,
                  tick_formats=r"$%.1f\pi$", offset=0.025, fontsize=8)
        # Corner labels: a point near the θᵢ corner means θᵢ dominates.
        tax.top_corner_label(r"$\theta_1$", fontsize=11, offset=0.22)
        tax.right_corner_label(r"$\theta_2$", fontsize=11, offset=0.08)
        tax.left_corner_label(r"$\theta_3$", fontsize=11, offset=0.08)

        # Data points colored by δ.
        rows = data[g]
        if rows:
            pts = [bary for bary, _ in rows]
            colors = [cmap(norm(d)) for _, d in rows]
            tax.scatter(pts, marker="o", s=28, c=colors,
                        linewidths=0, edgecolors="none")

        # Structured overlays as red stars.
        if overlay_bary:
            tax.scatter(list(overlay_bary.values()), marker="*", s=140,
                        c=[matplotlib.colors.to_rgba("red", 1.0)] * len(overlay_bary),
                        linewidths=0.6, edgecolors="black")

        tax.clear_matplotlib_ticks()
        ax.axis("off")

    # Shared colorbar panel.
    cax = fig.add_subplot(gs[0, 2])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\delta$ (best sample, $t=5$)")

    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
