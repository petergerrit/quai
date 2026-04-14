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
    labels = {"S216": r"$\Sigma(72\!\times\!3)$ ($|\mathcal{C}|\!=\!216$)",
              "S648": r"$\Sigma(216\!\times\!3)$ ($|\mathcal{C}|\!=\!648$)",
              "S1080": r"$\Sigma(360\!\times\!3)$ ($|\mathcal{C}|\!=\!1080$)"}

    # Top: overlaid histograms
    bins = np.linspace(0.05, 0.60, 28)
    for g_name in ("S216", "S648", "S1080"):
        rows = data[g_name]
        if not rows:
            continue
        deltas = np.array([d for _, d in rows])
        ax_top.hist(deltas, bins=bins, histtype="step", lw=1.8,
                    color=colors[g_name],
                    label=f"{labels[g_name]}  mean={deltas.mean():.3f}, std={deltas.std():.3f}")
    ax_top.set_xlabel(r"$\delta$")
    ax_top.set_ylabel("count (of 100)")
    ax_top.legend(fontsize=7, loc="upper right", frameon=False)
    ax_top.set_title(r"apples-100 panel: $\delta$ per base group at $d=3,\,t=5$")

    # Bottom: per-matrix scatter, x = apple_id
    for g_name in ("S216", "S648", "S1080"):
        rows = sorted(data[g_name])
        if not rows:
            continue
        xs = [k for k, _ in rows]
        ys = [d for _, d in rows]
        ax_bot.plot(xs, ys, "o-", color=colors[g_name], markersize=3, lw=0.5,
                    alpha=0.75, label=labels[g_name])
    ax_bot.set_xlabel("matrix id (seed=42 Haar draw)")
    ax_bot.set_ylabel(r"$\delta$")
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(fontsize=7, loc="upper right", frameon=False)

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
