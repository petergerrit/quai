"""
For each matrix in file B, find the closest matrix in file A by Frobenius distance,
and report per-matrix distances along with best, average, and worst.

Usage
-----
    python closest_matrix.py <file_A.npy> <file_B.npy> [--phase]

Options
-------
    --phase   Compare matrices up to global phase (minimise over U(1) factor)
"""

import sys
import argparse
import numpy as np


def frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius distance between two matrices."""
    return np.linalg.norm(A - B)


def frobenius_distance_up_to_phase(A: np.ndarray, B: np.ndarray) -> float:
    """
    Frobenius distance minimised over global phase:
        min_{theta} || A - e^{i*theta} * B ||_F
    The optimal phase is theta = -arg( Tr(A @ B^dagger) ),
    giving distance = sqrt( ||A||^2 + ||B||^2 - 2|Tr(A @ B^dagger)| ).
    """
    overlap = np.trace(A @ B.conj().T)
    dist_sq = (np.linalg.norm(A)**2 + np.linalg.norm(B)**2
               - 2 * abs(overlap))
    return np.sqrt(max(dist_sq, 0.0))


def find_closest(A: np.ndarray, B: np.ndarray, up_to_phase: bool = False):
    """
    For each matrix B[j], find the index and distance of the closest matrix in A.

    Parameters
    ----------
    A : ndarray, shape (n_A, d, d)
    B : ndarray, shape (n_B, d, d)

    Returns
    -------
    best_indices  : int array, shape (n_B,)
    best_distances: float array, shape (n_B,)
    """
    dist_fn = frobenius_distance_up_to_phase if up_to_phase else frobenius_distance

    best_indices   = np.zeros(len(B), dtype=int)
    best_distances = np.full(len(B), np.inf)

    for j, Bj in enumerate(B):
        for i, Ai in enumerate(A):
            d = dist_fn(Ai, Bj)
            if d < best_distances[j]:
                best_distances[j] = d
                best_indices[j]   = i

    return best_indices, best_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For each matrix in file B, find the closest matrix in file A."
    )
    parser.add_argument("file_A", type=str, help="Path to .npy file containing array of matrices A")
    parser.add_argument("file_B", type=str, help="Path to .npy file containing array of matrices B")
    parser.add_argument("--phase", action="store_true", help="Compare up to global phase")
    args = parser.parse_args()

    print(f"Loading A from {args.file_A}...")
    A = np.load(args.file_A)
    print(f"  {len(A)} matrices of shape {A.shape[1:]}")

    print(f"Loading B from {args.file_B}...")
    B = np.load(args.file_B)
    print(f"  {len(B)} matrices of shape {B.shape[1:]}\n")

    if A.shape[1:] != B.shape[1:]:
        print(f"Error: matrix shapes don't match ({A.shape[1:]} vs {B.shape[1:]})")
        sys.exit(1)

    mode = "up to global phase" if args.phase else "exact"
    print(f"Computing closest matches ({mode})...\n")

    indices, distances = find_closest(A, B, up_to_phase=args.phase)

    # Per-matrix results
    col_w = max(len(str(len(B))), 5)
    print(f"{'B idx':>{col_w}}  {'A idx':>{col_w}}  {'Distance':>12}")
    print("-" * (col_w * 2 + 16))
    for j, (idx, dist) in enumerate(zip(indices, distances)):
        print(f"{j:>{col_w}}  {idx:>{col_w}}  {dist:>12.8f}")

    # Summary statistics
    print()
    print(f"Best (min) distance : {distances.min():.8f}  (B[{distances.argmin()}] -> A[{indices[distances.argmin()]}])")
    print(f"Average distance    : {distances.mean():.8f}")
    print(f"Worst (max) distance: {distances.max():.8f}  (B[{distances.argmax()}] -> A[{indices[distances.argmax()]}])")
