"""Ternary (SU(3) Cartan-plane) plot of δ for the d3_rnd_diag panel.

Reads a ``q-panel`` cache database produced by
``swiftbot.cli q-panel --panel d3_rnd_diag`` and renders three subplots
(one per Σ-series base group). Each subplot is a 2-D scatter in the
Cartan A₂ coordinates

    x = (θ₂ - θ₃) / 2
    y = (2θ₁ - θ₂ - θ₃) / (2√3)

of the SU(3) maximal torus; the three implicit axes (θ₁ up, θ₂ lower-
right, θ₃ lower-left) are drawn at 120° to make the S₃ Weyl symmetry
visually obvious. Each point's color is δ on that base group; the color
scale is shared across subplots so cross-group comparisons are direct.

Structured-diagonal panel points (``P(2π/9)``, ``P(π/9)``, ``P(2π/5)``
from ``d3_survey``) are overlaid as stars if their fingerprints are
present in the same cache, to place the rational-angle extensions inside
the broader landscape.

Usage::

    # After running the panel:
    #   swiftbot q-panel --panel d3_rnd_diag --t 5 --run-id d3_rnd_diag \
    #       --db sweep_runs/d3_rnd_diag/cache.db
    python fig_d3_rnd_diag_ternary.py \
        --db sweep_runs/d3_rnd_diag/cache.db

Produces ``fig_d3_rnd_diag_ternary.pdf`` in the same directory as this
script.
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
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from swiftbot.cli import _d3_rnd_diag_panel, _d3_survey_panel   # noqa: E402
from swiftbot.stages.s3_efficiency import extension_fingerprint  # noqa: E402


# ---------------------------------------------------------------------------
# Cartan-plane coordinates
# ---------------------------------------------------------------------------

def cartan_xy(theta1: float, theta2: float, theta3: float) -> tuple[float, float]:
    """Map SU(3) eigenphases to the A₂ root-lattice 2D plane."""
    x = (theta2 - theta3) / 2.0
    y = (2.0 * theta1 - theta2 - theta3) / (2.0 * np.sqrt(3.0))
    return x, y


def draw_triaxes(ax: plt.Axes, R: float) -> None:
    """Draw θ₁ (up), θ₂ (lower-right), θ₃ (lower-left) axes with labels."""
    origin = np.zeros(2)
    # θ₁ axis direction = (0, 1); θ₂ = (√3/2, -1/2); θ₃ = (-√3/2, -1/2).
    dirs = {
        r"$\theta_1$": np.array([0.0, 1.0]),
        r"$\theta_2$": np.array([np.sqrt(3) / 2, -0.5]),
        r"$\theta_3$": np.array([-np.sqrt(3) / 2, -0.5]),
    }
    segs = [[origin, origin + R * d] for d in dirs.values()]
    ax.add_collection(LineCollection(segs, colors="0.5", linewidths=0.8, alpha=0.6))
    for lbl, d in dirs.items():
        pos = (R + R * 0.08) * d
        ax.text(pos[0], pos[1], lbl, ha="center", va="center",
                fontsize=10, color="0.3")


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

_GROUP_SIZE = {"S216": 216, "S648": 648, "S1080": 1080}


def load_best_delta(db_path: Path, fingerprint: str, group_size: int) -> float | None:
    """Return the minimum δ across all samples with this (fingerprint, group)
    pair. Fingerprints are the Stage-3 hash of the ExtensionSpec; for
    ``mat`` specs they include the matrix and θ values, so each diagonal
    has a distinct fingerprint.
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True,
                    help="SQLite cache produced by q-panel --panel d3_rnd_diag.")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "fig_d3_rnd_diag_ternary.pdf",
                    help="Output PDF path.")
    ap.add_argument("--no-latex", action="store_true",
                    help="Skip the LaTeX rendering pass (useful for CI).")
    args = ap.parse_args()

    if not args.no_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.latex.preamble": r"\usepackage{amsmath}",
        })

    # Pull the panel + structured-angle overlays from the CLI definitions so
    # fingerprints line up bit-identically with what q-panel wrote.
    diag_panel = _d3_rnd_diag_panel()
    survey_panel = _d3_survey_panel()
    survey_angles_by_name = {
        "P(2pi/9)": (0.0, 2 * np.pi / 9, -2 * np.pi / 9),
        "P(pi/9)":  (0.0, np.pi / 9,     -np.pi / 9),
        "P(2pi/5)": (0.0, 2 * np.pi / 5, -2 * np.pi / 5),
    }

    # Group coordinate
    groups = ["S216", "S648", "S1080"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)

    # Collect all δ values first so we can share the color scale.
    all_delta: list[float] = []
    cache_rows: dict[str, list[tuple[float, float, float]]] = {g: [] for g in groups}
    # cache_rows[g] entries are (x, y, delta) in Cartan coords.
    for name, spec in diag_panel:
        fp = extension_fingerprint(spec)
        p = spec.params
        x, y = cartan_xy(p["theta1"], p["theta2"], p["theta3"])
        for g in groups:
            d = load_best_delta(args.db, fp, _GROUP_SIZE[g])
            if d is None:
                continue
            cache_rows[g].append((x, y, d))
            all_delta.append(d)

    if not all_delta:
        raise SystemExit(
            f"No d3_rnd_diag rows found in {args.db}. "
            "Did you run `swiftbot q-panel --panel d3_rnd_diag ...` first?"
        )
    vmin, vmax = min(all_delta), max(all_delta)

    # Axis radius: SU(3) torus radius is 2π; Cartan-plane radius comes out
    # at most ~2π·√3/√3 = 2π after rescaling; give a bit of padding.
    R = 2 * np.pi / np.sqrt(3.0)
    pad = 0.15 * R

    # Structured-angle overlay in each subplot.
    survey_xy: dict[str, tuple[float, float]] = {
        k: cartan_xy(*v) for k, v in survey_angles_by_name.items()
    }
    # Also fetch structured δ values per group if they're in the same db.
    survey_delta: dict[str, dict[str, float]] = {g: {} for g in groups}
    for name, spec in survey_panel:
        if name not in survey_angles_by_name:
            continue
        fp = extension_fingerprint(spec)
        for g in groups:
            d = load_best_delta(args.db, fp, _GROUP_SIZE[g])
            if d is not None:
                survey_delta[g][name] = d

    for ax, g in zip(axes, groups):
        rows = cache_rows[g]
        if not rows:
            ax.set_title(f"{g}: no data")
            continue
        x, y, d = np.asarray(rows).T
        sc = ax.scatter(x, y, c=d, cmap="viridis", vmin=vmin, vmax=vmax,
                        s=28, edgecolors="none")

        # Structured overlays (stars)
        for nm, (xs, ys) in survey_xy.items():
            ax.plot(xs, ys, marker="*", markersize=13,
                    markerfacecolor="red", markeredgecolor="black",
                    markeredgewidth=0.5, linestyle="none")

        draw_triaxes(ax, R)
        ax.set_xlim(-R - pad, R + pad)
        ax.set_ylim(-R - pad, R + pad)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        sigma_labels = {"S216": r"$\Sigma(72{\times}3)$",
                        "S648": r"$\Sigma(216{\times}3)$",
                        "S1080": r"$\Sigma(360{\times}3)$"}
        ax.set_title(sigma_labels[g])

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, aspect=24, pad=0.02)
    cbar.set_label(r"$\delta$ (best sample)")

    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
