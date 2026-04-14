"""Generate fig:apples100.pdf — δ distribution on 100 identical Haar matrices
across S216, S648, S1080, once the apples-100 run lands on lenore.

Reads sweep_runs/apples100_t5.db (sync back from lenore first). Produces:
  - top panel: per-group δ histograms (3 overlaid or stacked)
  - bottom panel: per-matrix δ scatter, x = apple_id, y = δ, series per group
"""
from __future__ import annotations
import math
import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swiftbot.kb.cache import Cache
from swiftbot.stages.s3_efficiency import extension_fingerprint
from swiftbot.supervisor import ExtensionSpec


def _rebuild_apple_fps() -> dict[int, str]:
    """Recompute the apples-100 fingerprints (seed=42 rng, mat extensions),
    matching sweep_runs/apples100_t5.py."""
    rng = np.random.default_rng(seed=42)

    def haar_su(d: int = 3) -> np.ndarray:
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
        return Q / np.linalg.det(Q) ** (1 / d)

    out: dict[int, str] = {}
    for k in range(100):
        M = haar_su(3)
        spec = ExtensionSpec(
            kind="mat",
            params={"matrix": M.tolist(), "apple_id": k, "seed": 42},
            rationale=f"apples-100 apple{k:03d}",
        )
        out[k] = extension_fingerprint(spec)
    return out


def main(db_path: str = "sweep_runs/apples100_t5.db"):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
    fps = _rebuild_apple_fps()
    fp_to_id = {v: k for k, v in fps.items()}

    # Read all matching rows per base group.
    data: dict[str, list[tuple[int, float]]] = {"S216": [], "S648": [], "S1080": []}
    with Cache(path=Path(db_path)) as cache:
        by_name = {g.name: g for g in cache.list_groups()}
        for g_name in ("S216", "S648", "S1080"):
            g = by_name.get(g_name)
            if g is None:
                print(f"skipping {g_name}: not in cache", file=sys.stderr)
                continue
            for rec in cache.list_qt(g.group_key):
                aid = fp_to_id.get(rec.ext_fingerprint)
                if aid is None:
                    continue
                data[g_name].append((aid, rec.delta))

    if not all(len(data[g]) for g in data):
        missing = [g for g in data if not data[g]]
        print(f"WARNING: no data for {missing}", file=sys.stderr)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5.0, 5.0),
                                         gridspec_kw={"height_ratios": [1.0, 1.2]})

    colors = {"S216": "C0", "S648": "C1", "S1080": "C2"}
    labels = {"S216": r"$\Sigma(72\!\times\!3)$",
              "S648": r"$\Sigma(216\!\times\!3)$",
              "S1080": r"$\Sigma(360\!\times\!3)$"}
    markers = {"S216": "o", "S648": "s", "S1080": "^"}

    # Top: overlaid step histograms + vertical mean lines + ±σ bands.
    bins = np.linspace(0.05, 0.60, 28)
    for g_name in ("S216", "S648", "S1080"):
        rows = data[g_name]
        if not rows:
            continue
        deltas = np.array([d for _, d in rows])
        mu, sigma = deltas.mean(), deltas.std()
        ax_top.hist(deltas, bins=bins, histtype="step", lw=1.6,
                    color=colors[g_name], label=labels[g_name])
        # Horizontal ±σ band at a fixed y near the top of each group's
        # histogram — makes mean and width readable without cluttering.
    # Stack the mean/σ bars above the histograms so they don't clash
    # with the step lines themselves.
    _, top_y = ax_top.get_ylim()
    band_y = [top_y * 1.05, top_y * 1.13, top_y * 1.21]
    for i, g_name in enumerate(("S216", "S648", "S1080")):
        rows = data[g_name]
        if not rows:
            continue
        deltas = np.array([d for _, d in rows])
        mu, sigma = deltas.mean(), deltas.std()
        y = band_y[i]
        ax_top.hlines(y, mu - sigma, mu + sigma, color=colors[g_name],
                      lw=3.0, alpha=0.65, zorder=5)
        ax_top.plot(mu, y, marker="|", color=colors[g_name],
                    markersize=12, mew=2.0, zorder=6)
    ax_top.set_ylim(0, top_y * 1.30)
    ax_top.set_xlabel(r"$\delta$")
    ax_top.set_ylabel("count (of 100)")
    ax_top.legend(fontsize=8, loc="upper right", frameon=False)
    ax_top.set_title(r"apples-100 panel: $\delta$ per base group at $d=3,\,t=5$")

    # Bottom: per-matrix scatter with distinct markers; no connecting
    # lines — just points. A faint horizontal band shows mean ± σ per
    # group for context.
    for g_name in ("S216", "S648", "S1080"):
        rows = sorted(data[g_name])
        if not rows:
            continue
        deltas_arr = np.array([d for _, d in rows])
        mu, sigma = deltas_arr.mean(), deltas_arr.std()
        ax_bot.axhspan(mu - sigma, mu + sigma, color=colors[g_name],
                       alpha=0.10, zorder=0)
        ax_bot.axhline(mu, color=colors[g_name], alpha=0.5, lw=0.8,
                       ls="--", zorder=1)
        xs = [k for k, _ in rows]
        ys = [d for _, d in rows]
        ax_bot.plot(xs, ys, linestyle="none", marker=markers[g_name],
                    markersize=5, color=colors[g_name],
                    markerfacecolor=colors[g_name],
                    markeredgecolor=colors[g_name],
                    alpha=0.85, label=labels[g_name], zorder=2)
    ax_bot.set_xlabel("matrix id (seed=42 Haar draw)")
    ax_bot.set_ylabel(r"$\delta$")
    ax_bot.grid(True, alpha=0.2, zorder=-1)
    ax_bot.legend(fontsize=8, loc="upper right", frameon=False)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fig_apples100.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    # Also dump per-group summary so it's easy to splice into paper text
    print("\n== per-group summary ==")
    for g_name in ("S216", "S648", "S1080"):
        rows = data[g_name]
        if not rows:
            continue
        deltas = np.array([d for _, d in rows])
        print(f"{g_name:6s}  n={len(deltas):3d}  mean={deltas.mean():.4f}  "
              f"std={deltas.std():.4f}  best={deltas.min():.4f}  worst={deltas.max():.4f}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "sweep_runs/apples100_t5.db"
    main(db)
