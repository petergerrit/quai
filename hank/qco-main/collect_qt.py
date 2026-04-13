"""
collect_qt.py  —  collect Q_T results from all fixed-extension runs.

Parses output files produced by main.py with the new naming convention:
    qcoG<group>N<N>T<t>exttype<type>extval<val>f1s0v0.0.0.txt

Computes:
    delta  = max norm across all representation weights
    Q_T    = log(|C|) / log(1 / delta)

and writes one row per file to a results file:
    group  exttype  extval  t  delta  Q_T

Usage:
    python collect_qt.py [options]

Options:
    --dir DIR       Directory containing the output .txt files  [default: ./]
    --t T           Filter to a specific t value (default: collect all t)
    --out FILE      Output file path                [default: qt_results.txt]
    --clifford N    Override group size for Q_T computation
"""

import argparse
import glob
import math
import os
import re
import sys
import numpy as np
from itertools import groupby

GROUP_SIZES = {
    'clifford': 24,
    'clifford_group': 24,
    'BI': 120,
    'BO': 48,
    'BT': 24,
    'hurwitz': 12,
}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir',      default='./')
    p.add_argument('--t',        default=None, type=int)
    p.add_argument('--out',      default='qt_results.txt')
    p.add_argument('--clifford', default=None, type=int)
    p.add_argument('--group',    default=None,
                   help='Only collect results for this group tag (e.g. BO, BT, BI, clifford)')
    return p.parse_args()

def find_norms_files(directory, t=None):
    pattern = os.path.join(directory, f'qco*T{t}*s0*.txt' if t else 'qco*s0*.txt')
    return [f for f in glob.glob(pattern) if not f.endswith('-gates.txt')]

def parse_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    meta = {}
    m = re.search(r'G([A-Za-z][A-Za-z0-9_]*?)N', name)
    meta['group'] = m.group(1) if m else 'unknown'
    m = re.search(r'T(\d+)', name)
    if m is None:
        return None
    meta['t'] = int(m.group(1))
    m = re.search(r'exttype([A-Za-z]+?)(?=extval|f\d|$)', name)
    meta['exttype'] = m.group(1) if m else 'rnd'
    m = re.search(r'extval([^f]+?)(?=f\d|$)', name)
    meta['extval'] = m.group(1).rstrip('_') if m else ''
    meta['extval_float'] = None
    if meta['exttype'] == 'angle' and meta['extval']:
        try:
            meta['extval_float'] = float(meta['extval'])
        except ValueError:
            pass
    return meta

def read_norms_file(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2:
        raise ValueError(f"Too few lines in {path}")
    all_deltas = []
    for line in lines[1:]:
        vals = [float(x) for x in line.split()]
        if vals:
            all_deltas.append(max(vals))
    return max(all_deltas)

def infer_group_size(group_tag, override=None):
    if override is not None:
        return override
    for key, size in GROUP_SIZES.items():
        if key.lower() == group_tag.lower():
            return size
    m = re.search(r'(\d+)', group_tag)
    return int(m.group(1)) if m else None

def compute_qt(delta, group_size):
    if delta <= 0: return float('inf')
    if delta >= 1: return float('nan')
    return math.log(group_size) / math.log(1.0 / delta)

def main():
    args = parse_args()
    files = find_norms_files(args.dir, args.t)
    if not files:
        print(f"No matching files found in '{args.dir}'.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} result files.", file=sys.stderr)

    rows = []
    skipped = 0
    if args.group:
        files = [f for f in files if parse_filename(f) and
                 parse_filename(f)['group'].lower() == args.group.lower()]
        print(f"Filtered to group '{args.group}': {len(files)} files.", file=sys.stderr)
    for path in files:
        meta = parse_filename(path)
        if meta is None:
            print(f"  [WARN] Could not parse: {path}", file=sys.stderr)
            skipped += 1
            continue
        try:
            delta = read_norms_file(path)
        except Exception as e:
            print(f"  [WARN] Could not read {path}: {e}", file=sys.stderr)
            skipped += 1
            continue
        group_size = infer_group_size(meta['group'], args.clifford)
        if group_size is None:
            print(f"  [WARN] Unknown group size for '{meta['group']}'. Use --clifford N.", file=sys.stderr)
            skipped += 1
            continue
        qt = compute_qt(delta, group_size)
        rows.append({
            'group': meta['group'], 'group_size': group_size,
            'exttype': meta['exttype'], 'extval': meta['extval'],
            'extval_f': meta['extval_float'], 't': meta['t'],
            'delta': delta, 'qt': qt,
            'file': os.path.basename(path),
        })

    if skipped:
        print(f"Skipped {skipped} files.", file=sys.stderr)

    rows.sort(key=lambda r: (r['group'], r['exttype'],
                              r['extval_f'] if r['extval_f'] is not None else float('inf'),
                              r['extval'], r['t']))

    # Embed t value in filename if --t was specified
    base, ext = os.path.splitext(args.out)
    suffix = ''
    if args.group:
        suffix += f'_{args.group}'
    if args.t is not None:
        suffix += f'_t{args.t}'
    out_file = f'{base}{suffix}{ext}' if suffix else args.out
    out_path = out_file if os.path.isabs(out_file) else os.path.join(args.dir, out_file)
    with open(out_path, 'w') as f:
        f.write("# group  group_size  exttype  extval  t  delta  Q_T  file\n")
        for r in rows:
            f.write(f"{r['group']}  {r['group_size']}  {r['exttype']}  "
                    f"{r['extval']}  {r['t']}  "
                    f"{r['delta']:.18f}  {r['qt']:.10f}  {r['file']}\n")

    print(f"Wrote {len(rows)} rows to: {out_path}", file=sys.stderr)

    if rows:
        print(f"\nSummary:", file=sys.stderr)
        for group, group_rows in groupby(rows, key=lambda r: r['group']):
            group_rows = list(group_rows)
            gs = group_rows[0]['group_size']
            opt_delta = 2 * math.sqrt(gs - 1) / gs
            q_opt = math.log(gs) / math.log(1 / opt_delta)
            qt_vals = [r['qt'] for r in group_rows if math.isfinite(r['qt'])]
            if not qt_vals:
                continue
            best  = min(group_rows, key=lambda r: r['qt'])
            worst = max(group_rows, key=lambda r: r['qt'])
            print(f"\n  Group {group} (|C|={gs}, Q_opt={q_opt:.4f}):", file=sys.stderr)
            print(f"    Best  Q_T={best['qt']:.4f}  exttype={best['exttype']} extval={best['extval']} t={best['t']}", file=sys.stderr)
            print(f"    Worst Q_T={worst['qt']:.4f}  exttype={worst['exttype']} extval={worst['extval']} t={worst['t']}", file=sys.stderr)
            print(f"    Mean  Q_T={np.mean(qt_vals):.4f}  ({len(qt_vals)} results)", file=sys.stderr)

if __name__ == '__main__':
    main()
