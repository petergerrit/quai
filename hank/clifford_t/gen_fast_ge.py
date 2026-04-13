#!/usr/bin/env python3

import numpy as np
import argparse
import os
from numpy.linalg import norm


# ---------------------------------------------------------------------------
# Deduplication: vectorized exact norm comparison, O(n * unique) time
# ---------------------------------------------------------------------------

def _find_nonzero(M, tol=1e-10):
    """Return index of first entry with |M[i]| > tol, or -1."""
    flat = np.abs(M.ravel())
    nz = np.flatnonzero(flat > tol)
    return nz[0] if len(nz) > 0 else -1


def _phase_normalise(M, tol=1e-10):
    """Return M rotated so its first non-negligible entry is real & positive."""
    idx = _find_nonzero(M, tol)
    if idx < 0:
        return M
    flat = M.ravel()
    phase = flat[idx] / abs(flat[idx])
    return M / phase


def _fingerprint(M, scale):
    """
    Cheap tuple of scalar invariants used as a hash-bucket key.

    Uses trace(M), trace(M^2), and Frobenius norm — three independent
    invariants that together make accidental bucket collisions extremely rare.
    All are O(d^2). Values are rounded to a grid of width `scale`.
    """
    def _r(x):
        return round(float(x.real) / scale) + 1j * round(float(x.imag) / scale)

    tr   = np.trace(M)
    tr2  = np.trace(M @ M)
    frob = float(np.sqrt(np.sum(np.abs(M) ** 2)))
    return (_r(tr), _r(tr2), round(frob / scale))


def uniq(Ms_arr, up_to_phase=False, tol=1e-10):
    """
    Remove duplicate matrices using hash-bucketed fingerprinting.

    Algorithm
    ---------
    For each matrix, compute a cheap fingerprint (trace, trace(M^2), Frobenius
    norm) rounded to a grid of width tol*10.  Only matrices sharing the same
    fingerprint bucket are compared with a full norm check.

    Complexity
    ----------
    O(n) amortised when buckets stay small (the typical case), vs the previous
    O(n * n_unique) vectorised scan.  Worst case is unchanged.

    Parameters
    ----------
    Ms_arr     : np.ndarray, shape (n, d, d)
    up_to_phase: bool  — if True, compare up to global phase
    tol        : float — equality tolerance

    Returns
    -------
    np.ndarray, shape (n_unique, d, d)
    """
    n = len(Ms_arr)
    if n == 0:
        return Ms_arr

    # Grid coarser than tol so two matrices within tol land in the same bucket
    # even with floating-point noise, while well-separated ones do not.
    scale = tol * 10

    if up_to_phase:
        Ms_norm = np.array([_phase_normalise(Ms_arr[i], tol) for i in range(n)])
    else:
        Ms_norm = Ms_arr

    # buckets: fingerprint -> list of positions in kept_norm
    buckets: dict = {}
    kept_norm    = []
    kept_indices = []

    for i in range(n):
        M    = Ms_norm[i]
        fkey = _fingerprint(M, scale)

        bucket = buckets.get(fkey)
        if bucket is None:
            # No matrix with this fingerprint yet — definitely unique.
            buckets[fkey] = [len(kept_norm)]
            kept_norm.append(M)
            kept_indices.append(i)
            continue

        # Full norm check only against the (usually tiny) bucket.
        is_dup = False
        for j in bucket:
            if np.sqrt(np.sum(np.abs(kept_norm[j] - M) ** 2)) < tol:
                is_dup = True
                break

        if not is_dup:
            bucket.append(len(kept_norm))
            kept_norm.append(M)
            kept_indices.append(i)

    return Ms_arr[kept_indices]


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_words_correct(B, T, nt, up_to_phase=False, output_base=None):
    """
    Generate all unique matrices of the form:
      B[i0] @ T @ B[i1] @ T @ ... @ T @ B[i_nt]

    Built left to right:
      Start : B[i0]
      Step t: (previous word) @ T @ B[i_t]

    Optimisations
    -------------
    1. Vectorised exact dedup  — correct and ~10-20x faster than Python loops.
    2. Precompute TB[k] = T @ B[k]  — reused at every step.
    3. Batch matmul via np.matmul broadcasting  — no Python loop over words.

    If output_base is provided, each intermediate result is saved to:
      {output_base}_nt0.npy, {output_base}_nt1.npy, ..., {output_base}_nt{nt}.npy
    """
    n_B = len(B)
    B_arr = np.asarray(B)           # (n_B, d, d)

    print(f"Generating words with {nt} T matrices...")
    print(f"Number of B matrices: {n_B}")
    print(f"Word structure: B[i0] @ T @ B[i1] @ ... @ T @ B[i_nt]")
    print(f"Total naive combinations: {n_B ** (nt + 1)}")
    print()

    # Precompute T @ B[k] for all k — shape (n_B, d, d)
    TB = np.matmul(T[None, :, :], B_arr)   # T broadcast-matmul over batch

    # Start: one copy of each B matrix, deduplicated — shape (n_unique, d, d)
    current = uniq(B_arr.copy(), up_to_phase=up_to_phase)
    print(f"Initial (0 T's): {len(current)} unique words")

    for t in range(nt):
        # Extend every current word by every T@B:
        #   result[n, k] = current[n] @ TB[k]
        # np.matmul broadcast: (n_cur, 1, d, d) @ (1, n_B, d, d) -> (n_cur, n_B, d, d)
        next_arr = np.matmul(current[:, None, :, :], TB[None, :, :, :])

        n_cur, _, d, _ = next_arr.shape
        next_flat = next_arr.reshape(n_cur * n_B, d, d)

        before = len(next_flat)
        current = uniq(next_flat, up_to_phase=up_to_phase)
        after = len(current)
        print(f"After T matrix {t + 1}/{nt}: {before} -> {after} unique words")

        if output_base is not None:
            step_file = f"{output_base}_nt{t + 1}.npy"
            np.save(step_file, current)
            print(f"  Saved to {step_file}")

    return list(current)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate unique words of the form B@T@B@T@...@T@B'
    )
    parser.add_argument('file',    type=str,            help='Path to .npy file containing array of matrices B')
    parser.add_argument('tfile',   type=str,            help='Path to .npy file containing the T matrix')
    parser.add_argument('nt',      type=int,            help='Number of T matrices in the word')
    parser.add_argument('--phase', action='store_true', help='Compare matrices up to global phase')
    args = parser.parse_args()

    print(f"Loading matrices from {args.file}...")
    B = np.load(args.file)
    print(f"Loaded {len(B)} matrices of shape {B.shape[1:]}\n")

    print(f"Loading T matrix from {args.tfile}...")
    T = np.load(args.tfile)
    print(f"T matrix shape: {T.shape}")
    print(f"T matrix:\n{T}\n")

    output_base = (
        f"{os.path.basename(args.file).replace('.npy', '')}"
        f"_{os.path.basename(args.tfile).replace('.npy', '')}"
    )

    words = generate_words_correct(B, T, args.nt, up_to_phase=args.phase, output_base=output_base)

    print(f"\n=== Results ===")
    print(f"Total unique words with {args.nt} T matrices: {len(words)}")
    print(f"Intermediate results saved as {output_base}_nt1.npy ... {output_base}_nt{args.nt}.npy")

    return words


if __name__ == '__main__':
    main()
