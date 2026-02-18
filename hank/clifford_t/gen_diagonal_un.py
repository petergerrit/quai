"""
Generate a random diagonal SU(N) matrix and save it to a .npy file.

A diagonal SU(N) matrix has the form diag(e^{i*theta_1}, ..., e^{i*theta_N})
where the angles satisfy the constraint sum(theta_k) = 0 (mod 2*pi) to ensure det = 1.

Usage:
    python gen_diagonal_sun.py <N>
"""

import sys
import numpy as np


def random_diagonal_sun(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a random diagonal SU(N) matrix.

    The first angle is fixed to 0 (so the first diagonal entry is always 1),
    and the remaining (N-1) angles are drawn uniformly from [0, 2*pi).
    The matrix is in U(N) (not necessarily SU(N)).

    Returns:
        matrix : complex128 ndarray of shape (N, N)
        angles : float64 ndarray of shape (N,)
    """
    free_angles = np.random.uniform(0, 2 * np.pi, size=n - 1)
    angles = np.concatenate([[0.0], free_angles])

    diagonal = np.exp(1j * angles)
    matrix = np.diag(diagonal)
    return matrix, angles


def angles_to_filename(angles: np.ndarray) -> str:
    """Build a filename embedding the angles, rounded to 4 decimal places."""
    angle_str = "_".join(f"{a:.4f}" for a in angles)
    return f"diag_su{len(angles)}_{angle_str}.npy"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <N> [theta_1 ... theta_N]")
        print(f"  Angles are given as multiples of pi (e.g. 0.5 means pi/2).")
        print(f"  If angles are omitted, they are chosen randomly.")
        sys.exit(1)

    try:
        N = int(sys.argv[1])
    except ValueError:
        print(f"Error: N must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)

    if N < 1:
        print("Error: N must be at least 1.")
        sys.exit(1)

    n_extra = len(sys.argv) - 2
    if n_extra == 0:
        # No angles provided — sample randomly
        matrix, angles = random_diagonal_sun(N)
        angle_prefactors = angles / np.pi
        print(f"Generated random diagonal U({N}) matrix")
    else:
        # Angles provided as multiples of pi — validate count then parse
        if n_extra != N:
            print(f"Error: expected {N} angle prefactors for U({N}), got {n_extra}.")
            sys.exit(1)
        try:
            angle_prefactors = np.array([float(a) for a in sys.argv[2:]])
        except ValueError as e:
            print(f"Error parsing angles: {e}")
            sys.exit(1)
        if not np.isclose(angle_prefactors[0], 0.0):
            print(f"Error: first angle prefactor must be 0, got {angle_prefactors[0]}.")
            sys.exit(1)
        angles = angle_prefactors * np.pi
        diagonal = np.exp(1j * angles)
        matrix = np.diag(diagonal)
        print(f"Generated diagonal U({N}) matrix from provided angles")

    # Quick verification
    det = np.linalg.det(matrix)
    is_unitary = np.allclose(matrix @ matrix.conj().T, np.eye(N))
    print(f"  Angle prefactors (units of pi): {np.round(angle_prefactors, 4)}")
    print(f"  Angles (rad):                   {np.round(angles, 4)}")
    print(f"  Diagonal entries: {np.round(np.diag(matrix), 6)}")
    print(f"  det = {det:.6f}")
    print(f"  Unitary: {is_unitary}")

    filename = angles_to_filename(angles)
    np.save(filename, matrix)
    print(f"  Saved to: {filename}")
