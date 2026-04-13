#!/usr/bin/env python3
"""
Read a data file from the group sampling output and display a
matplotlib histogram of: how many elements have a given fraction of total samples.

Usage:
    python histogram.py <datafile> [n_bins]
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt


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
        sys.exit("Usage: python histogram.py <datafile> [n_bins]")

    filepath = sys.argv[1]
    n_bins   = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"Reading: {filepath}")
    counts = parse_counts(filepath)

    if len(counts) == 0:
        sys.exit("No data found — check file contains a '=== Total Results ===' table")

    total  = counts.sum()
    fracs  = counts / total
    mean_f = fracs.mean()

    print(f"Found {len(counts):,} elements, {total:,} total samples")
    print(f"Min fraction: {fracs.min():.2e}  Max: {fracs.max():.2e}  Mean: {mean_f:.2e}")
    print(f"n_bins: {n_bins}")

    # compute histogram manually for full control
    bin_edges        = np.linspace(fracs.min(), fracs.max(), n_bins + 1)
    hist, edges      = np.histogram(fracs, bins=bin_edges)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(edges[:-1], hist, width=np.diff(edges),
           color="steelblue", edgecolor="white", linewidth=0.4, align="edge")

    ax.axvline(mean_f,      color="red",    linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_f:.2e}")
    ax.axvline(fracs.max(), color="orange", linestyle=":",  linewidth=1.5,
               label=f"Max = {fracs.max():.2e}")
    ax.axvline(fracs.min(), color="green",  linestyle=":",  linewidth=1.5,
               label=f"Min = {fracs.min():.2e}")

    ax.set_yscale("log")
    ax.set_xlabel("Fraction of total samples", fontsize=12)
    ax.set_ylabel("Number of elements", fontsize=12)
    ax.set_title(
        f"{filepath.split('/')[-1]}\n"
        f"{len(counts):,} elements  |  {total:,} total samples  |  "
        f"uniformity (max/mean) = {fracs.max()/mean_f:.2f}",
        fontsize=10
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
