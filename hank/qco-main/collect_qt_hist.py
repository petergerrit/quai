"""
collect_qt_hist.py  —  collect Q_T distributions from multi-sample norm files
                        and plot overlapping histograms.

File format expected (same as produced by main.py with sample_size > 1):
    [w1] [w2] ...           <- header line with weight labels (brackets optional)
    v1   v2   ...           <- one row per sample; delta = max(v1, v2, ...)
    ...

For each row:
    delta  = max norm across all representation weights for that sample
    Q_T    = log(|C|) / log(1 / delta)

Usage:
    python collect_qt_hist.py [options]

Options:
    --dir DIR        Directory containing the .txt output files  [default: ./]
    --t T            Value of t (used only for plot title)       [default: 100]
    --clifford N     Size of the Clifford group |C|              [default: 24]
    --bins N         Number of histogram bins                    [default: 40]
    --out FILE       Output plot file (PNG/PDF/SVG)              [default: qt_histogram.png]
    --no-show        Do not open the plot window interactively
    --pattern GLOB   Override the file glob pattern              [default: *.txt]
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--dir',      default='./',          help='Directory with output files')
    p.add_argument('--t',        default=100, type=int, help='Value of t (for plot title)')
    p.add_argument('--clifford', default=24,  type=int, help='|C| – Clifford group size')
    p.add_argument('--bins',     default=40,  type=int, help='Number of histogram bins')
    p.add_argument('--out',      default='qt_histogram.png')
    p.add_argument('--no-show',  action='store_true',   help='Skip interactive display (save only)')
    p.add_argument('--pattern',  default='*.txt',       help='Glob for input files')
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_files(directory, pattern):
    full_pattern = os.path.join(directory, pattern)
    files = sorted(glob.glob(full_pattern))
    # drop any "-gates.txt" helper files
    files = [f for f in files if not f.endswith('-gates.txt')]
    return files


def short_label(path):
    """Turn a filename into a readable legend label (strip directory + extension)."""
    base = os.path.basename(path)
    # remove common long prefixes to keep the legend tidy
    base = re.sub(r'^qco', '', base)
    # strip the .txt extension
    if base.endswith('.txt'):
        base = base[:-4]
    return base


def read_norms_file(path):
    """
    Parse a norms file and return a 2-D numpy array  (n_samples × n_weights).

    Header line format:  [w1] [w2] ...   OR   w1 w2 ...
    Data lines:          float float ...
    """
    with open(path) as fh:
        raw = [ln.strip() for ln in fh if ln.strip()]

    if len(raw) < 2:
        raise ValueError(f"File too short (< 2 non-empty lines): {path}")

    # ---- detect and skip the header ----------------------------------------
    # A header line contains '[' OR at least one non-numeric token.
    def is_header(line):
        if '[' in line:
            return True
        tokens = line.split()
        for tok in tokens:
            try:
                float(tok)
            except ValueError:
                return True        # non-numeric → header
        return False

    start_row = 0
    if is_header(raw[0]):
        start_row = 1

    # ---- parse data rows ----------------------------------------------------
    data = []
    for line in raw[start_row:]:
        try:
            row = [float(x) for x in line.split()]
            if row:
                data.append(row)
        except ValueError:
            pass   # silently skip any remaining non-data lines

    if not data:
        raise ValueError(f"No numeric data rows found in {path}")

    # Pad rows to equal length (fill with NaN) so we can make a 2-D array
    max_len = max(len(r) for r in data)
    padded  = [r + [float('nan')] * (max_len - len(r)) for r in data]
    return np.array(padded, dtype=float)


def compute_qt_array(matrix, clifford_size):
    """
    Given an (n_samples × n_weights) array, return a 1-D array of Q_T values.

    delta[i] = nanmax of row i
    Q_T[i]   = log(|C|) / log(1 / delta[i])

    Samples with delta <= 0 or delta >= 1 are returned as NaN and excluded
    from the histogram.
    """
    delta = np.nanmax(matrix, axis=1)          # shape: (n_samples,)

    log_C = math.log(clifford_size)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_inv_delta = np.where(
            (delta > 0) & (delta < 1),
            np.log(1.0 / delta),
            np.nan
        )
        qt = np.where(
            np.isfinite(log_inv_delta) & (log_inv_delta > 0),
            log_C / log_inv_delta,
            np.nan
        )

    return qt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histograms(datasets, bins, clifford_size, t, out_path, show):
    """
    datasets : list of (label, qt_array) tuples
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute shared bin edges from all data combined
    all_qt = np.concatenate([qt[np.isfinite(qt)] for _, qt in datasets])
    if all_qt.size == 0:
        print("[ERROR] No valid Q_T values found across all files.", file=sys.stderr)
        sys.exit(1)

    bin_edges = np.linspace(all_qt.min(), all_qt.max(), bins + 1)

    # Optimal Q_T lower bound
    opt = math.log(clifford_size) / math.log(
        clifford_size / (2 * math.sqrt(clifford_size - 1))
    )

    # Color cycle
    cmap   = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(datasets))]

    for (label, qt), color in zip(datasets, colors):
        valid = qt[np.isfinite(qt)]
        if valid.size == 0:
            print(f"  [WARN] No valid Q_T values for '{label}', skipping.", file=sys.stderr)
            continue
        ax.hist(
            valid,
            bins=bin_edges,
            alpha=0.45,
            color=color,
            edgecolor=color,
            linewidth=0.5,
            label=f"{label}  (n={valid.size}, μ={valid.mean():.3f})",
            density=False,
        )

    # Vertical line at optimal bound
    ax.axvline(opt, color='black', linestyle='--', linewidth=1.5,
               label=f"Q_T optimal ≈ {opt:.3f}")

    ax.set_xlabel("Q_T", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"Q_T distribution — Clifford+P gate family  |  t = {t},  |C| = {clifford_size}",
        fontsize=13,
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved histogram to: {out_path}", file=sys.stderr)

    if show:
        plt.show()   # blocks until the window is closed
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(datasets, clifford_size):
    opt = math.log(clifford_size) / math.log(
        clifford_size / (2 * math.sqrt(clifford_size - 1))
    )
    print(f"\n{'Label':<55}  {'n':>6}  {'min':>8}  {'mean':>8}  {'max':>8}", file=sys.stderr)
    print("-" * 90, file=sys.stderr)
    for label, qt in datasets:
        valid = qt[np.isfinite(qt)]
        if valid.size == 0:
            print(f"  {label:<53}  {'—':>6}  {'—':>8}  {'—':>8}  {'—':>8}", file=sys.stderr)
        else:
            print(
                f"  {label:<53}  {valid.size:>6}  "
                f"{valid.min():>8.4f}  {valid.mean():>8.4f}  {valid.max():>8.4f}",
                file=sys.stderr,
            )
    print(f"\n  Q_T optimal lower bound: {opt:.4f}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    files = find_files(args.dir, args.pattern)
    if not files:
        print(
            f"No files matching '{args.pattern}' found in '{args.dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(files)} file(s).", file=sys.stderr)

    datasets = []
    for path in files:
        label = short_label(path)
        try:
            matrix = read_norms_file(path)
            qt     = compute_qt_array(matrix, args.clifford)
            datasets.append((label, qt))
            print(f"  Loaded {matrix.shape[0]} samples from {os.path.basename(path)}", file=sys.stderr)
        except Exception as exc:
            print(f"  [WARN] Skipping {path}: {exc}", file=sys.stderr)

    if not datasets:
        print("No data could be loaded.", file=sys.stderr)
        sys.exit(1)

    print_summary(datasets, args.clifford)

    out_path = (
        os.path.join(args.dir, args.out)
        if not os.path.isabs(args.out)
        else args.out
    )
    plot_histograms(
        datasets,
        bins=args.bins,
        clifford_size=args.clifford,
        t=args.t,
        out_path=out_path,
        show=not args.no_show,
    )


if __name__ == '__main__':
    main()
