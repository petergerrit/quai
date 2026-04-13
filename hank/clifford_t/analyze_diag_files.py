"""
Analyzes groups of data files with names like:
    data_{header}_diag_{param1}_{param2}_upnt{n}

Computes gate-set efficiency metrics from the full count distribution:

  - max/mean ratio    : peak coverage relative to average (lower = more uniform)
  - coeff. of variation (CV) : std/mean of counts (lower = more uniform)
  - entropy efficiency : H(p) / H(uniform), where p is the count distribution
                         (higher = more uniform, max = 1)

For each header group produces:
  Figure 1 — all three metrics vs n  (scatter + log-space fit lines)
  Figure 2 — value of each metric at n=max (or mean across n) vs the 2nd
             numerical parameter (angle θ), to identify the best gate set

Usage:
    python analyze_diag_files.py                          # looks in ./norm_data
    python analyze_diag_files.py --folder /path/to/data
    python analyze_diag_files.py --folder /path/to/data --save
"""

import os
import re
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_filename(fname):
    m = re.match(r'^(.+)_upnt(\d+)$', fname)
    return (m.group(1), int(m.group(2))) if m else None


def get_header(base_key):
    m = re.match(r'^(.+?)_diag', base_key)
    return m.group(1) if m else base_key


def get_last_float_param(base_key):
    """Returns the last x.y float after '_diag_' — this is the angle θ."""
    after_diag = re.sub(r'^.+?_diag_', '', base_key)
    floats = re.findall(r'\d+\.\d+', after_diag)
    return float(floats[-1]) if floats else None


def short_label(base_key):
    return re.sub(r'^.+_diag_', '', base_key)


# ---------------------------------------------------------------------------
# File parsing — extract full count distribution
# ---------------------------------------------------------------------------

def extract_counts(filepath):
    """
    Parses the count table from a data file.
    Returns a numpy array of integer counts, one per element.
    Expects rows like:   '   0      108 0.00108000 ...'
    between the header dashes and the 'Total' line.
    """
    counts = []
    in_table = False
    with open(filepath, 'r') as f:
        for line in f:
            if re.match(r'\s*-{10,}', line):
                in_table = not in_table
                continue
            if in_table:
                m = re.match(r'\s*(\d+)\s+(\d+)\s+[\d.]+', line)
                if m:
                    counts.append(int(m.group(2)))
    return np.array(counts, dtype=float) if counts else None


# ---------------------------------------------------------------------------
# Uniformity metrics
# ---------------------------------------------------------------------------

def compute_metrics(counts):
    """
    Given a 1-D array of counts, returns a dict of uniformity metrics.

    max_count_fraction  : max(p)    — max of the probability distribution
                          lower = better; penalises both non-uniformity and small N
    std_p               : std(p)    — std of the probability distribution
                          lower = better; scale-free, ideal value = 0 for any N
    entropy_eff         : H(p) = -sum(p log p)  (Shannon entropy in nats)
                          higher = more uniform AND larger set
    """
    if counts is None or len(counts) == 0:
        return None
    mean = counts.mean()
    if mean == 0:
        return None
    p = counts / counts.sum()

    N = len(counts)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.sum(p * np.where(p > 0, np.log(p), 0))

    return {
        'max_mean_ratio': p.max(),    # max count fraction = max(p)
        'cv':             p.std(),    # std of probability distribution
        'entropy_eff':    entropy,    # H(p) in nats
    }


# ---------------------------------------------------------------------------
# Folder scan
# ---------------------------------------------------------------------------

def analyze_folder(folder):
    """
    Returns:
        data[header][base_key] = {n: metrics_dict, ...}
    """
    data = defaultdict(lambda: defaultdict(dict))

    for fname in sorted(os.listdir(folder)):
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        base_key, n = parsed
        header  = get_header(base_key)
        counts  = extract_counts(os.path.join(folder, fname))
        metrics = compute_metrics(counts)
        if metrics is not None:
            data[header][base_key][n] = metrics
        else:
            print(f"  [WARN] Could not extract counts from: {fname}")

    return data


# ---------------------------------------------------------------------------
# Log-space linear fit helper
# ---------------------------------------------------------------------------

def log_fit(ns, vals):
    """
    Fits log10(vals) ~ slope*n + intercept.
    Returns (slope, intercept, r_value) or None on failure.
    For entropy_eff values close to 1 we fit in linear space instead,
    since log of near-1 values is noisy — detected automatically.
    """
    arr = np.array(vals, dtype=float)
    ns  = np.array(ns,   dtype=float)
    if len(ns) < 2:
        return None

    # Decide whether log or linear space is more appropriate
    if np.all(arr > 0) and (arr.min() / arr.max()) < 0.5:
        # good dynamic range — use log space
        y = np.log10(arr)
    else:
        # compressed range (e.g. entropy 0.95–1.0) — use linear space
        y = arr

    slope, intercept, r_value, _, _ = stats.linregress(ns, y)
    return slope, intercept, r_value, y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRICS = [
    ('max_mean_ratio', 'Max count fraction  max(p)',  'lower = more uniform & larger'),
    ('cv',             'Std of p  std(p)',             'lower = more uniform & larger'),
    ('entropy_eff',    'Shannon entropy  H(p)  [nats]', 'higher = more uniform & larger'),
]


def plot_header(header, series_dict, output_dir=None, show_metrics=False):
    """
    Figure 1 — three metrics vs n, one panel each.
    Figure 2 — each metric's value at largest n vs angle θ (2nd param).
    """
    n_series = len(series_dict)
    colors   = cm.tab20(np.linspace(0, 1, max(n_series, 1)))
    all_ns   = sorted({n for nd in series_dict.values() for n in nd})
    safe_hdr = re.sub(r'[^\w\-]', '_', header)

    # ------------------------------------------------------------------ #
    # Figure 1 — metrics vs n  (shown only with --show-metrics)          #
    # ------------------------------------------------------------------ #
    if show_metrics:
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
        fig1.suptitle(f"{header}  —  uniformity metrics vs n",
                      fontsize=13, fontweight='bold')

        for ax, (mkey, mtitle, mnote) in zip(axes1, METRICS):
            for (base_key, n_dict), color in zip(sorted(series_dict.items()), colors):
                ns   = sorted(n_dict.keys())
                vals = [n_dict[n][mkey] for n in ns]
                lbl  = short_label(base_key)

                ax.scatter(ns, vals, marker='o', s=25, color=color, zorder=3)

                fit = log_fit(ns, vals)
                if fit is not None:
                    slope, intercept, r_value, y_fit = fit
                    n_fine = np.linspace(min(ns), max(ns), 200)
                    if np.all(np.array(vals) > 0) and (min(vals) / max(vals)) < 0.5:
                        fit_line = 10 ** (slope * n_fine + intercept)
                    else:
                        fit_line = slope * n_fine + intercept
                    ax.plot(n_fine, fit_line, color=color, linewidth=1.1,
                            label=lbl, zorder=2)
                else:
                    ax.plot([], [], color=color, label=lbl)

            if mkey in ('max_mean_ratio', 'cv'):
                ax.set_yscale('log')
                ax.set_ylabel(f'{mtitle}  (log scale)', fontsize=10)
            else:
                ax.set_ylabel(mtitle, fontsize=10)

            ax.set_xlabel('n  (upnt index)', fontsize=10)
            ax.set_title(f'{mtitle}\n({mnote})', fontsize=10)
            ax.set_xticks(all_ns)
            ax.grid(True, linestyle='--', alpha=0.4, which='both')

            if n_series <= 20:
                ax.legend(fontsize=6, loc='best', title='θ params',
                          title_fontsize=7)
            else:
                ax.legend(fontsize=5, loc='best', ncol=2,
                          title='θ params', title_fontsize=6)

        fig1.tight_layout()
        if output_dir:
            path = os.path.join(output_dir, f"plot_{safe_hdr}_metrics_vs_n.png")
            fig1.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")

    # ------------------------------------------------------------------ #
    # Figure 2 — metrics vs angle θ, all n shown with distinct markers   #
    # ------------------------------------------------------------------ #
    max_n = max(all_ns)

    # Build flat list of (theta, n, metrics_dict) for every file
    point_data = []
    for base_key, n_dict in sorted(series_dict.items()):
        theta = get_last_float_param(base_key)
        if theta is None:
            continue
        for n, m in n_dict.items():
            point_data.append((theta, n, m))

    # Transform θ → π/θ for the x-axis
    if not point_data:
        print(f"  [INFO] No θ data for {header} — skipping angle plot.")
        plt.show()
        return

    point_data_plot = [(np.pi / t, n, m) for t, n, m in point_data]

    # Assign a distinct marker shape to each n value
    marker_list = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "P"]
    n_to_marker = {n: marker_list[i % len(marker_list)]
                   for i, n in enumerate(all_ns)}

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(
        f"{header}  —  uniformity metrics vs i=π/θ  (all n values)\n"
        f"(best gate sets: low ratio/CV, high entropy efficiency)",
        fontsize=12, fontweight="bold")

    for ax, (mkey, mtitle, mnote) in zip(axes2, METRICS):
        # Global value range for this metric — consistent colour scale across n
        all_vals = np.array([m[mkey] for _, _, m in point_data_plot])
        vmin_m   = all_vals.min()
        vrange   = (all_vals.max() - vmin_m) + 1e-12

        # One scatter call per n so each gets its own marker + legend entry
        for n in all_ns:
            subset   = [(t, m) for t, nn, m in point_data_plot if nn == n]
            if not subset:
                continue
            thetas_n = np.array([t   for t, _ in subset])
            vals_n   = np.array([m[mkey] for _, m in subset])
            order_n  = np.argsort(thetas_n)
            thetas_n = thetas_n[order_n]
            vals_n   = vals_n[order_n]

            if mkey == "entropy_eff":
                norm_score = (vals_n - vmin_m) / vrange
            else:
                norm_score = 1 - (vals_n - vmin_m) / vrange

            ax.scatter(thetas_n, vals_n,
                       c=norm_score, cmap="RdYlGn",
                       marker=n_to_marker[n],
                       s=55, zorder=3, vmin=0, vmax=1,
                       edgecolors="k", linewidths=0.4,
                       label=f"n={n}")
            ax.plot(thetas_n, vals_n,
                    color="grey", linewidth=0.6,
                    linestyle="--", alpha=0.3, zorder=2)

        # Best π/θ = minimises/maximises the mean metric across all n
        x_set = sorted({t for t, _, _ in point_data_plot})
        mean_by_x = [
            np.mean([m[mkey] for t, _, m in point_data_plot if t == x])
            for x in x_set
        ]
        best_x = x_set[
            np.argmax(mean_by_x) if mkey == "entropy_eff"
            else np.argmin(mean_by_x)
        ]
        ax.axvline(best_x, color="green", linewidth=1.3,
                   linestyle=":", alpha=0.8,
                   label=f"best i={best_x:.1f} (mean over n)")

        ax.set_xlabel("i  (= π/θ)", fontsize=10)
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
        ax.set_title(f"{mtitle}\n({mnote})", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, linestyle="--", alpha=0.4, which="both")

    fig2.tight_layout()
    if output_dir:
        path = os.path.join(output_dir, f"plot_{safe_hdr}_metrics_vs_theta.png")
        fig2.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='norm_data',
                        help='Folder containing the data files (default: ./norm_data)')
    parser.add_argument('--save', action='store_true',
                        help='Save plots as PNG files into the data folder')
    parser.add_argument('--show-metrics', action='store_true',
                        help='Also show per-n metric vs n plots (Figure 1)')
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"ERROR: folder '{folder}' not found.")
        return

    print(f"Scanning folder: {folder}")
    data = analyze_folder(folder)

    if not data:
        print("No matching files found. Check that files end with _upnt<n>.")
        return

    output_dir = folder if args.save else None

    print(f"\nFound {len(data)} header group(s):")
    for header, series in sorted(data.items()):
        all_ns   = sorted({n for nd in series.values() for n in nd})
        param2s  = sorted({get_last_float_param(k) for k in series
                           if get_last_float_param(k) is not None})
        p2_str   = f"{param2s[0]:.4f}–{param2s[-1]:.4f}" if param2s else "?"
        print(f"  {header:40s}  |  {len(series):3d} series  "
              f"|  n={all_ns}  |  θ range: {p2_str}")

    print("\nGenerating plots...")
    for header, series in sorted(data.items()):
        plot_header(header, series, output_dir=output_dir,
                    show_metrics=args.show_metrics)


if __name__ == '__main__':
    main()
