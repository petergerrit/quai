#!/usr/bin/env python3
"""
npy_to_qco.py  —  convert a .npy group file (as produced by genBI/genBO/genBT)
to the text format expected by qco's main.py -gates_path argument.

File format produced:
    Line 0:      dimension n  (e.g. 2)
    Lines 1..:   one row of the matrix per line, space-separated complex numbers
                 written as  a+bj  or  a-bj  (no parentheses), n rows per gate.

Usage:
    python npy_to_qco.py input.npy [output.txt]

    If output.txt is omitted, writes to <input_stem>.txt in the current directory.
    e.g.  python npy_to_qco.py BI.npy          ->  BI.txt
          python npy_to_qco.py BO.npy gates.txt ->  gates.txt
"""

import sys
import os
import numpy as np


def fmt_complex(z):
    """Format a complex number that Python's complex() can parse back.
    Produces  a+bj  or  a-bj  — no parentheses, explicit sign on imaginary part."""
    r, i = float(z.real), float(z.imag)
    sign = '+' if i >= 0 else '-'
    return f"{r:.17g}{sign}{abs(i):.17g}j"


def convert(npy_path, out_path):
    gates = np.load(npy_path)

    if gates.ndim != 3 or gates.shape[1] != gates.shape[2]:
        raise ValueError(
            f"Expected shape (n_gates, d, d), got {gates.shape}"
        )

    n_gates, d, _ = gates.shape
    print(f"Loaded {n_gates} gates of dimension {d}x{d} from '{npy_path}'")

    # Sanity checks
    unitary_errors = [np.max(np.abs(g @ g.conj().T - np.eye(d))) for g in gates]
    print(f"  Max unitarity error: {max(unitary_errors):.2e}")
    det_errors = [abs(abs(np.linalg.det(g)) - 1.0) for g in gates]
    print(f"  Max |det|-1 error:   {max(det_errors):.2e}")

    with open(out_path, 'w') as f:
        f.write(f"{d}\n")
        for gate in gates:
            for row in gate:
                f.write(" ".join(fmt_complex(x) for x in row) + "\n")

    print(f"  Written to: '{out_path}'")

    # Verify round-trip
    loaded_back = []
    with open(out_path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    for i in range(1, len(lines)):
        line = lines[i].split()
        if (i - 1) % n == 0:
            loaded_back.append([])
        loaded_back[-1].append([complex(x) for x in line])
    loaded_back = np.array(loaded_back)

    roundtrip_errors = np.max(np.abs(loaded_back - gates))
    print(f"  Round-trip error:    {roundtrip_errors:.2e}")
    if roundtrip_errors > 1e-10:
        print("  WARNING: round-trip error is large — check the file.")
    else:
        print("  Round-trip OK.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    npy_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        stem = os.path.splitext(os.path.basename(npy_path))[0]
        out_path = stem + '.txt'

    convert(npy_path, out_path)
