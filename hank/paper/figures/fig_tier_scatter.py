"""Data-driven Q_T-vs-|C| scatter plot for the d3_survey panel.

Reads a ``q-panel-summary`` text file (same tabular format as
``sweep_runs/d3_tier1_summary.txt``) and produces a PDF matching the style
of ``fig_qt_vs_groupsize.pdf``. Superseding the hardcoded numbers in
``fig_qt_vs_groupsize.py`` with a data-driven pipeline means a Tier-2 rerun
only needs ``q-panel-summary --run-id d3_tier2 > d3_tier2_summary.txt``
followed by ``python fig_tier_scatter.py d3_tier2_summary.txt`` to refresh.

Usage::

    python fig_tier_scatter.py                                    # Tier-1 (default)
    python fig_tier_scatter.py --summary sweep_runs/d3_tier2_summary.txt
    python fig_tier_scatter.py --summary <file> --out fig.pdf

The summary file must contain the header ``=== TIER ... (d3_survey) ===``
followed by one line per (group, extension) pair with columns
``group extension kind n best_δ mean_δ std_δ Q_T(b) Q_T(μ) Q_opt``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Summary-file parser
# ---------------------------------------------------------------------------

_GROUP_SIZE = {"S216": 216, "S648": 648, "S1080": 1080}

# Header-delimited section for the d3_survey panel. Everything up to the next
# '===' line belongs to the panel.
_SECTION_RE = re.compile(r"=== TIER [^=]*\(d3_survey\) ===")


def parse_summary(path: Path) -> dict[int, dict[str, float]]:
    """Return ``{|C|: {extension_name: Q_T_best, ..., 'Popt': ...}}``."""
    lines = path.read_text().splitlines()
    # Locate the d3_survey block.
    start = next(i for i, line in enumerate(lines) if _SECTION_RE.search(line))
    # Skip header + dashes.
    data_start = start + 3
    out: dict[int, dict[str, float]] = {}
    for ln in lines[data_start:]:
        if not ln.strip():
            continue
        if ln.startswith("==="):
            break
        # Split on whitespace; the extension column can contain no spaces in
        # the fixed panel, and the kind column is one word.
        parts = ln.split()
        if len(parts) < 10:
            continue
        group, ext, kind = parts[0], parts[1], parts[2]
        if group not in _GROUP_SIZE:
            continue
        size = _GROUP_SIZE[group]
        best_qt = float(parts[7])
        popt = float(parts[9])
        out.setdefault(size, {})[ext] = best_qt
        out[size]["Popt"] = popt
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_EXT_STYLE = [
    ("HV(1,1,0)_campbell",      dict(marker="o",  color="#1f77b4",
                                     label=r"$\mathrm{HV}(1,1,0)$ Campbell")),
    ("HV(1,2,0)_eq27",          dict(marker="s",  color="#ff7f0e",
                                     label=r"$\mathrm{HV}(1,2,0)$ Eq.~(27)")),
    ("HV(2,1,0)",               dict(marker="D",  color="#2ca02c",
                                     label=r"$\mathrm{HV}(2,1,0)$ new orbit")),
    ("P(2pi/9)",                dict(marker="^",  color="#d62728",
                                     label=r"$P(2\pi/9)$")),
    ("P(pi/9)",                 dict(marker="v",  color="#9467bd",
                                     label=r"$P(\pi/9)$")),
    ("P(2pi/5)",                dict(marker="<",  color="#8c564b",
                                     label=r"$P(2\pi/5)$")),
]
_RND_KEYS = ("rnd_batch1", "rnd_batch2")
_SIZES = [216, 648, 1080]
_TICK_LABELS = {216: r"$\Sigma(72{\times}3)$",
                648: r"$\Sigma(216{\times}3)$",
                1080: r"$\Sigma(360{\times}3)$"}


def plot(data: dict[int, dict[str, float]], out_path: Path, title: str | None) -> None:
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

    fig, (ax, ax_legend) = plt.subplots(
        1, 2, figsize=(7.0, 4.0),
        gridspec_kw={"width_ratios": [3.2, 1.0]},
    )

    # Kesten-McKay floor
    popt = [data[s]["Popt"] for s in _SIZES]
    ax.plot(_SIZES, popt, "--", color="0.3", lw=1.5,
            label=r"$Q_{\mathrm{opt}}$ (Kesten--McKay floor)")

    # Structured extensions
    for key, style in _EXT_STYLE:
        qt = [data[s].get(key) for s in _SIZES]
        if any(v is None for v in qt):
            continue  # skip extensions missing in this summary
        ax.plot(_SIZES, qt, linestyle="-", markersize=9, linewidth=1.2, **style)

    # Haar-random stars, one per batch per group
    rnd_label_shown = False
    for s in _SIZES:
        for bkey in _RND_KEYS:
            v = data[s].get(bkey)
            if v is None:
                continue
            if not rnd_label_shown:
                ax.plot(s, v, marker="*", color="k", markersize=14, zorder=5,
                        label=r"Haar-random (best, per batch)")
                rnd_label_shown = True
            else:
                ax.plot(s, v, marker="*", color="k", markersize=14, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\mathcal{C}|$")
    ax.set_ylabel(r"$Q_T$ (best-sample)")
    ax.set_xticks(_SIZES)
    ax.set_xticklabels([_TICK_LABELS[s] for s in _SIZES])
    ax.minorticks_off()
    ax.set_yticks([2, 3, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels([r"$2$", r"$3$", r"$5$", r"$10$",
                        r"$20$", r"$50$", r"$100$", r"$200$"])
    ax.grid(True, which="both", alpha=0.3)
    if title:
        ax.set_title(title)

    ax_legend.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center left", frameon=False,
                     fontsize=9, handlelength=1.8, handletextpad=0.6)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def _default_summary() -> Path:
    here = Path(__file__).resolve()
    # paper/figures/ -> paper -> hank -> sweep_runs/
    return here.parent.parent.parent / "sweep_runs" / "d3_tier1_summary.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=_default_summary(),
                    help="q-panel-summary text file (default: Tier-1).")
    ap.add_argument("--out", type=Path,
                    help="Output PDF path (default: derived from summary name).")
    ap.add_argument("--title", type=str, default=None,
                    help="Optional plot title.")
    args = ap.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"summary file not found: {args.summary}")

    data = parse_summary(args.summary)
    if not data:
        raise SystemExit(f"no d3_survey data parsed from {args.summary}")

    out = args.out
    if out is None:
        stem = args.summary.stem.replace("_summary", "") + "_qt_vs_size"
        out = Path(__file__).resolve().parent / f"fig_{stem}.pdf"

    plot(data, out, args.title)


if __name__ == "__main__":
    main()
