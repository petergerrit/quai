#!/usr/bin/env python3
"""Generalized finite-group closure tool.

Given a set of generator matrices, compute the closure under multiplication and
save the resulting element list to .npy and/or to the QCO .txt format expected
by qco-main{,_opt}/main.py -gates_path.

Replaces the per-group genBI.py / genS108.py / gens60x4.py etc. with a single
parameterized entry point, so scaling up to new qudit dimensions only requires
supplying generators (as an .npy array or a small Python snippet).

Usage as a CLI:
    python genGROUP.py \\
        --generators GENS.npy \\
        --name MYGROUP \\
        [--out-dir DIR] \\
        [--max-size N] \\
        [--decimals 5] \\
        [--format npy|txt|both]

    The generators .npy must have shape (n_generators, d, d) and complex dtype.

Usage as a library:
    from genGROUP import close_group, save_npy, save_qco_txt
    elements = close_group([g1, g2, ...])
    save_npy(elements, "MYGROUP.npy")
    save_qco_txt(elements, "MYGROUP.txt")

Tolerance notes:
    Deduplication uses a rounded-entries fingerprint. Default 5 decimal places
    matches the existing gens*.py scripts; tighten with --decimals if generators
    use irrational values that collide after rounding. Loosen if numerical
    roundoff spreads true-identical elements across multiple hash buckets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def matrix_key(M: np.ndarray, decimals: int = 5) -> tuple:
    """Hashable O(d²) fingerprint: rounded real and imaginary entries."""
    arr = np.asarray(M)
    return (
        tuple(np.round(arr.real, decimals).flatten().tolist())
        + tuple(np.round(arr.imag, decimals).flatten().tolist())
    )


def projective_key(M: np.ndarray, decimals: int = 5, zero_tol: float = 1e-8) -> tuple:
    """Hashable fingerprint for the projective class [M] ∈ PU(d) = U(d)/U(1).

    Two matrices M, M' satisfy [M] = [M'] iff M' = e^{iφ}·M for some φ. We
    canonicalize by rotating M so its first non-zero entry is real positive —
    this picks a unique representative per class.
    """
    arr = np.asarray(M)
    for z in arr.flat:
        if abs(z) > zero_tol:
            canon = arr * (abs(z) / z)  # phase-remove on first non-zero
            return matrix_key(canon, decimals)
    return matrix_key(arr, decimals)  # all-zero (shouldn't happen for unitaries)


def close_group(
    generators: Sequence[np.ndarray],
    *,
    max_size: int | None = None,
    decimals: int = 5,
    projective: bool = False,
    verbose: bool = True,
) -> list[np.ndarray]:
    """Compute the closure of `generators` under matrix multiplication.

    Finite-group correctness: since every element of a finite group G has
    finite order, g⁻¹ = g^(ord(g)-1) is reachable as a product of copies of g,
    so a closure under ordinary multiplication converges to all of G without
    needing the user to supply inverses explicitly.

    Args:
        generators: list of d×d matrices (numpy arrays; matrices are coerced
            to complex ndarray).
        max_size: if set, abort with a RuntimeError if the closure exceeds
            4× this number. Guard against bad generators / bad tolerance.
        decimals: rounding for the dedup hash.
        projective: if True, identify matrices that differ by a global U(1)
            phase — i.e. close in PU(d) instead of U(d). This is the
            convention the Słowik-Dulian-Sawicki paper uses for e.g. Hurwitz
            (size 12) and 1-qubit Clifford (size 24); without it the closures
            are 2× larger (the SU(d) double covers).
        verbose: print per-iteration size on stderr.

    Returns:
        list of complex ndarray d×d matrices, one per distinct (projective)
        group element.
    """
    if not generators:
        raise ValueError("need at least one generator")
    d = np.asarray(generators[0]).shape[0]
    for g in generators:
        arr = np.asarray(g)
        if arr.shape != (d, d):
            raise ValueError(f"all generators must be {d}×{d}; got {arr.shape}")

    key_fn = projective_key if projective else matrix_key

    seen: dict[tuple, np.ndarray] = {}
    for g in generators:
        M = np.asarray(g, dtype=complex)
        seen[key_fn(M, decimals)] = M
    # Include identity so small groups that don't explicitly add it still close.
    eye = np.eye(d, dtype=complex)
    seen.setdefault(key_fn(eye, decimals), eye)

    frontier = list(seen.values())
    iteration = 0
    while frontier:
        iteration += 1
        current_snapshot = list(seen.values())
        new_by_key: dict[tuple, np.ndarray] = {}
        for A in current_snapshot:
            for B in frontier:
                for M in (A @ B, B @ A):
                    k = key_fn(M, decimals)
                    if k not in seen and k not in new_by_key:
                        new_by_key[k] = M
        if not new_by_key:
            break
        seen.update(new_by_key)
        frontier = list(new_by_key.values())
        if verbose:
            print(
                f"  iter {iteration}: |G| = {len(seen)} (+{len(new_by_key)} new)",
                file=sys.stderr,
            )
        if max_size is not None and len(seen) > 4 * max_size:
            raise RuntimeError(
                f"closure exceeded 4× max_size ({len(seen)} > {4 * max_size}). "
                "Likely wrong generators or wrong --decimals."
            )
    return list(seen.values())


def _fmt_complex(z: complex) -> str:
    """Match qco-main/npy_to_qco.fmt_complex: 'a+bj' or 'a-bj', 17 sig digits."""
    r, i = float(z.real), float(z.imag)
    sign = "+" if i >= 0 else "-"
    return f"{r:.17g}{sign}{abs(i):.17g}j"


def save_npy(matrices: Iterable[np.ndarray], path: str | Path) -> None:
    arr = np.asarray(list(matrices), dtype=complex)
    np.save(path, arr)


def save_qco_txt(matrices: Iterable[np.ndarray], path: str | Path) -> None:
    """Write in the QCO -gates_path format: first line d, then d rows/gate."""
    matrices = list(matrices)
    if not matrices:
        raise ValueError("no matrices to save")
    d = matrices[0].shape[0]
    with Path(path).open("w") as f:
        f.write(f"{d}\n")
        for M in matrices:
            for row in M:
                f.write(" ".join(_fmt_complex(z) for z in row) + "\n")


def verify_unitary(matrices: Iterable[np.ndarray], tol: float = 1e-8) -> tuple[float, float]:
    """Return (max |M M† - I|, max ||det|-1|) across the element list."""
    matrices = list(matrices)
    if not matrices:
        return 0.0, 0.0
    d = matrices[0].shape[0]
    eye = np.eye(d, dtype=complex)
    u_err = max(float(np.max(np.abs(M @ M.conj().T - eye))) for M in matrices)
    d_err = max(abs(abs(np.linalg.det(M)) - 1.0) for M in matrices)
    return u_err, d_err


def _load_generators(path: Path) -> list[np.ndarray]:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise SystemExit(
            f"{path}: generators must have shape (n_gen, d, d); got {arr.shape}"
        )
    return [np.asarray(arr[i], dtype=complex) for i in range(arr.shape[0])]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--generators", required=True, type=Path,
                   help="path to .npy with shape (n_gen, d, d)")
    p.add_argument("--name", required=True,
                   help="group name (used in output filenames)")
    p.add_argument("--out-dir", default=Path.cwd(), type=Path)
    p.add_argument("--max-size", type=int,
                   help="guard — abort if closure exceeds 4× this value")
    p.add_argument("--decimals", type=int, default=5,
                   help="rounding for dedup hash (default 5)")
    p.add_argument("--format", choices=("npy", "txt", "both"), default="both")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    gens = _load_generators(args.generators)
    d = gens[0].shape[0]
    if not args.quiet:
        print(f"Closing '{args.name}' from {len(gens)} generators in U({d}) ...",
              file=sys.stderr)

    elements = close_group(
        gens,
        max_size=args.max_size,
        decimals=args.decimals,
        verbose=not args.quiet,
    )
    u_err, det_err = verify_unitary(elements)
    print(f"|G| = {len(elements)}  (d={d})")
    print(f"  max unitarity error: {u_err:.2e}")
    print(f"  max |det|-1 error:   {det_err:.2e}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.format in ("npy", "both"):
        path = args.out_dir / f"{args.name}.npy"
        save_npy(elements, path)
        print(f"wrote {path}")
    if args.format in ("txt", "both"):
        path = args.out_dir / f"{args.name}.txt"
        save_qco_txt(elements, path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
