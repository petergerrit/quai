"""
Produce the qudit pi/8 gate (U_upsilon) as defined in Howard & Vala (2012),
"Qudit versions of the qubit pi/8 gate", PRA 86, 022316.

Convention
----------
For prime p > 3 (Eq. 22):
    U_upsilon = diag(omega^{v_0}, ..., omega^{v_{p-1}})
    where omega = e^{2*pi*i/p} and
    v_k = (1/12) * k * { gamma' + k * [6*z' + (2k-3)*gamma'] } + k*epsilon  (mod p)
    with v_0 = 0 (boundary condition).

For p = 3 (Eq. 25):
    U_upsilon = diag(zeta^{v_0}, zeta^{v_1}, zeta^{v_2})
    where zeta = e^{2*pi*i/9} (9th root of unity) and
    v = (0, 6z' + 2*gamma' + 3*epsilon, 6z' + gamma' + 6*epsilon) mod 9

For p = 2:
    The standard qubit pi/8 gate: diag(e^{-i*pi/8}, e^{i*pi/8}), Eq. (1).

Parameters z', gamma', epsilon are integers in Z_p (default: z'=1, gamma'=1, epsilon=0).
These defaults reproduce the M(p) gate of Campbell et al., Eqs. (64-66) of the paper.

Usage
-----
    python sun_T_gate.py <p> [z'] [gamma'] [epsilon]
"""

import sys
import numpy as np
from sympy import isprime, mod_inverse


def sun_T_gate_p3(z: int, gamma: int, eps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Qutrit (p=3) U_upsilon gate via Eq. (25).
    Phases are 9th roots of unity: zeta = e^{2*pi*i/9}.
    """
    zeta = np.exp(2j * np.pi / 9)
    v = np.array([
        0,
        (6*z + 2*gamma + 3*eps) % 9,
        (6*z +   gamma + 6*eps) % 9,
    ])
    matrix = np.diag(zeta ** v)
    return matrix, v.astype(float)


def sun_T_gate_general(p: int, z: int, gamma: int, eps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    U_upsilon for prime p > 3 via Eq. (22).
    Phases are p-th roots of unity: omega = e^{2*pi*i/p}.
    All arithmetic modulo p.
    """
    inv12 = int(mod_inverse(12, p))
    omega = np.exp(2j * np.pi / p)

    v = np.zeros(p, dtype=int)
    for k in range(p):
        inner = (gamma + k * ((6*z + (2*k - 3)*gamma) % p)) % p
        v[k] = (inv12 * k * inner + k * eps) % p

    matrix = np.diag(omega ** v)
    return matrix, v.astype(float)


def sun_T_gate(p: int, z: int = 1, gamma: int = 1, eps: int = 0):
    """Dispatch to the correct construction based on p."""
    if not isprime(p):
        raise ValueError(f"p={p} is not prime. This construction requires prime p.")
    if p == 2:
        matrix = np.diag([1.0+0j, np.exp(1j * np.pi / 4)])
        angles = np.array([0, np.pi/4])
        return matrix, angles
    elif p == 3:
        return sun_T_gate_p3(z % 3, gamma % 3, eps % 3)
    else:
        return sun_T_gate_general(p, z % p, gamma % p, eps % p)


def verify(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    det = np.linalg.det(matrix)
    is_unitary = np.allclose(matrix @ matrix.conj().T, np.eye(n))
    is_diagonal = np.allclose(matrix, np.diag(np.diag(matrix)))
    return {"det": det, "unitary": is_unitary, "diagonal": is_diagonal}


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print(f"Usage: python {sys.argv[0]} <p> [z'] [gamma'] [epsilon]")
        sys.exit(1)

    try:
        P   = int(sys.argv[1])
        Z   = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        G   = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        EPS = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    matrix, v = sun_T_gate(P, Z, G, EPS)
    info = verify(matrix)

    print(f"U_upsilon gate for p={P}, z'={Z}, gamma'={G}, epsilon={EPS}")
    print(f"  Reference: Howard & Vala, PRA 86, 022316 (2012)")
    print(f"  Exponents v_k : {v}")
    print(f"  Diagonal entries: {np.round(np.diag(matrix), 6)}")
    print(f"  det  = {info['det']:.8f}")
    print(f"  Unitary : {info['unitary']}")
    print(f"  Diagonal: {info['diagonal']}")

    # Spot-checks against paper examples
    if P == 3 and Z == 1 and G == 2 and EPS == 0:
        zeta = np.exp(2j * np.pi / 9)
        expected = np.diag([1.0+0j, zeta**1, zeta**8])
        print(f"  Matches paper Eq. (27) [z'=1, gamma'=2, eps=0]: {np.allclose(matrix, expected)}")
    if P == 5 and Z == 1 and G == 4 and EPS == 0:
        omega5 = np.exp(2j * np.pi / 5)
        expected = np.diag([omega5**0, omega5**3, omega5**4, omega5**2, omega5**1])
        print(f"  Matches paper Eq. (24) [z'=1, gamma'=4, eps=0]: {np.allclose(matrix, expected)}")
    
    filename = f"T_{P}_k{Z}{G}{EPS}.npy"
    np.save(filename, matrix)
    print(f"  Saved to: {filename}")
