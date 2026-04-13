#!/usr/bin/env python3
"""
parse_voronoi_data.py  <datafile> [<datafile> ...]

For each file, prints one line:
  nt=<int>  n_rows=<int>  min=<f>  avg=<f>±<f>  max=<f>  expected=<f>

'nt' is the integer following the last occurrence of 'nt' in the filename.
Exits non-zero if any file has no completed results table.
"""

import sys, re, os
import numpy as np

# Table data row: element  count  ratio  mean_tr  std_tr
ROW_RE   = re.compile(r'^\s+\d+\s+\d+\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s*$')
# "Expected per cell (uniform): 277.8 0.000278"
EXP_RE   = re.compile(r'Expected per cell \(uniform\):\s+[\d.e+-]+\s+([\d.e+-]+)')
# Signals start of the results block
START_RE = re.compile(r'^===\s*Total Results')


def parse_file(path):
    nt_match = re.search(r'nt(\d+)', os.path.basename(path))
    if not nt_match:
        raise ValueError(f"No 'nt<int>' found in filename: {path}")
    nt = int(nt_match.group(1))

    ratios, expected, in_table = [], None, False

    with open(path) as fh:
        for line in fh:
            if START_RE.match(line):
                in_table = True
                continue
            if not in_table:
                continue
            m = ROW_RE.match(line)
            if m:
                ratios.append(float(m.group(1)))
                continue
            m = EXP_RE.search(line)
            if m:
                expected = float(m.group(1))

    if not ratios:
        raise ValueError(f"No completed results table in: {path}")

    r = np.array(ratios)
    if expected is None:
        expected = 1.0 / len(r)

    return nt, len(r), r.min(), r.mean(), r.std(), r.max(), expected


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <datafile> [<datafile> ...]", file=sys.stderr)
        sys.exit(1)

    ok = True
    for path in sys.argv[1:]:
        try:
            nt, n_rows, rmin, ravg, rstd, rmax, exp = parse_file(path)
            print(f"nt = {nt}  n_rows = {n_rows}  "
                  f"min = {rmin:.8f}  avg = {ravg:.8f} ± {rstd:.8f}  "
                  f"max = {rmax:.8f}  expected = {exp:.8f}")
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
