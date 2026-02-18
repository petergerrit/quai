"""
Produce the SU(N) generalization of the T gate and save to 'extension_T_{n}.npy'.

Convention
----------
The SU(N) T gate is built from the N-th roots of unity:

    T_N = diag(1, omega, omega^2, ..., omega^{N-1})

where omega = e^{2*pi*i/N} is the primitive N-th root of unity.

The first diagonal entry is always exactly 1. The determinant is
omega^{N(N-1)/2}, which equals 1 (SU(N)) when N ≡ 0 or 1 (mod 4),
and -1 (U(N)) otherwise.

Special cases:
  N=2 : diag(1, e^{i*pi}) = diag(1, -1)   (note: standard qubit T uses e^{i*pi/4}
                                             which is the 8th root of unity, not 2nd)
  N=3 : diag(1, e^{2*pi*i/3}, e^{4*pi*i/3})  (standard qutrit T gate)

Usage
-----
    python sun_T_gate.py <N>
"""

import sys
import numpy as np


def sun_T_gate(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the SU(N) T gate using N-th roots of unity.

    Parameters
    ----------
    n : int
        Dimension (n >= 2).

    Returns
    -------
    matrix : complex128 ndarray, shape (n, n)
    angles : float64 ndarray, shape (n,)
    """
    k = np.arange(n, dtype=float)
    angles = 2 * np.pi * k / n             # diag(1, omega, omega^2, ...) exactly
    matrix = np.diag(np.exp(1j * angles))
    return matrix, angles


def verify(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    det = np.linalg.det(matrix)
    is_unitary = np.allclose(matrix @ matrix.conj().T, np.eye(n))
    is_diagonal = np.allclose(matrix, np.diag(np.diag(matrix)))
    return {"det": det, "unitary": is_unitary, "diagonal": is_diagonal}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <N>")
        sys.exit(1)

    try:
        N = int(sys.argv[1])
    except ValueError:
        print(f"Error: N must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)

    if N < 2:
        print("Error: N must be at least 2.")
        sys.exit(1)

    matrix, angles = sun_T_gate(N)
    info = verify(matrix)

    print(f"SU({N}) T gate")
    print(f"  Angles (rad): {np.round(angles, 6)}")
    print(f"  Diagonal entries: {np.round(np.diag(matrix), 6)}")
    print(f"  det = {info['det']:.8f}  (should be ~1+0j)")
    print(f"  Unitary : {info['unitary']}")
    print(f"  Diagonal: {info['diagonal']}")

    # Spot-checks against known standard forms
    if N == 2:
        expected = np.diag([1.0+0j, -1.0+0j])
        print(f"  Matches diag(1, -1): {np.allclose(matrix, expected)}")
    if N == 3:
        omega = np.exp(2j * np.pi / 3)
        expected = np.diag([1.0+0j, omega, omega**2])
        print(f"  Matches standard qutrit T gate: {np.allclose(matrix, expected)}")

    filename = f"extension_T_{N}.npy"
    np.save(filename, matrix)
    print(f"  Saved to: {filename}")
