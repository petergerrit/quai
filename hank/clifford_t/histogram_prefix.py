#!/usr/bin/env python3
"""
Plot histograms of sampling fractions for all files matching
<prefix>_upnt<N>.npy, overlaid on the same axes with log x and y.

Usage:
    python histogram_prefix.py <prefix> [n_bins]

Example:
    python histogram_prefix.py clifford24_diag_su2_0.0000_0.0314 50
"""

import sys
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def parse_counts(filepath: str) -> np.ndarray:
    """Parse the '=== Total Results ===' table, return counts as numpy array."""
    counts = []
    in_table = False
    with open(filepath) as f:
        for line in f:
            if "=== Total Results" in line:
                in_table = True
                continue
            if not in_table:
                continue
            if line.strip().startswith("Element") or line.strip().startswith("---"):
                continue
            m = re.match(r"^\s*(\d+)\s+(\d+)", line)
            if m:
                counts.append(int(m.group(2)))
    return np.array(counts, dtype=np.int64)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python histogram_prefix.py <prefix> [n_bins]")

    prefix = sys.argv[1]
    n_bins = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    # strip trailing _upnt if user included it accidentally
    if prefix.endswith("_upnt"):
        prefix = prefix[:-5]

    # find all matching files: <prefix>_upnt<int> (no extension)
    candidates = glob.glob(f"{prefix}_upnt*")
    files = sorted(
        [f for f in candidates if re.search(r"_upnt\d+$", f)],
        key=lambda f: int(re.search(r"_upnt(\d+)$", f).group(1))
    )

    if not files:
        sys.exit(f"No files found matching '{prefix}_upnt<N>'"
                 f" — check the prefix and directory are correct")

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    # use a qualitative colormap for maximum distinction between lines
    cmap   = plt.get_cmap("tab10") if len(files) <= 10 else plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(files))]

    fig, ax = plt.subplots(figsize=(11, 6))

    n_elem = None  # will be set from first successfully loaded file

    for filepath, color in zip(files, colors):
        nt_match = re.search(r"_upnt(\d+)", filepath)
        nt_label = f"upnt{nt_match.group(1)}" if nt_match else filepath

        counts = parse_counts(filepath)
        if len(counts) == 0:
            print(f"  [WARN] no data found in {filepath}, skipping")
            continue

        total  = counts.sum()
        fracs  = counts / total
        if n_elem is None:
            n_elem = len(counts)

        # log-spaced bins spanning the full range across all non-zero fracs
        nonzero = fracs[fracs > 0]
        if len(nonzero) == 0:
            print(f"  [WARN] all counts zero in {filepath}, skipping")
            continue
        bin_edges = np.logspace(np.log10(nonzero.min()), np.log10(fracs.max()), n_bins + 1)
        hist, edges = np.histogram(fracs, bins=bin_edges)
        hist_normed = hist / len(counts)  # fraction of total elements per bin

        # plot as semitransparent bars
        ax.bar(edges[:-1], np.where(hist_normed > 0, hist_normed, np.nan),
               width=np.diff(edges), align="edge",
               color=color, edgecolor=color, linewidth=0.3,
               alpha=0.5, label=nt_label)

        uniform_f = 1.0 / len(counts)
        ax.axvline(uniform_f, color=color, linestyle="--", linewidth=1.2,
                   alpha=0.9, label=f"{nt_label} uniform = {uniform_f:.2e}")

        print(f"  {nt_label}: {len(counts):,} elements, {total:,} samples, "
              f"max/mean = {fracs.max()/fracs.mean():.2f}")

    if n_elem is None:
        sys.exit("No files contained valid data — cannot plot")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fraction of total samples", fontsize=12)
    ax.set_ylabel("Fraction of elements", fontsize=12)
    ax.set_title(f"Sampling distribution: {prefix}\n"
                 f"(darker = lower nt, lighter = higher nt)",
                 fontsize=10)
    # ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
