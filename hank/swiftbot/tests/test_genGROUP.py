"""Validate clifford_t/genGROUP.py against known finite groups.

Strategy: reconstruct the generators used by the per-group gen*.py scripts in
clifford_t/, run them through our generalized `close_group`, and assert the
resulting element count matches both the known group order and the element
count in the corresponding qco-main_opt/<group>.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]  # .../hank/
CLIFFORD_T = REPO / "clifford_t"
QCO_OPT = REPO / "qco-main_opt"

# Put clifford_t on sys.path so genGROUP imports cleanly from tests.
sys.path.insert(0, str(CLIFFORD_T))

from genGROUP import close_group, save_npy, save_qco_txt, verify_unitary  # noqa: E402


def _count_matrices_in_qco_txt(path: Path) -> tuple[int, int]:
    """Return (d, n_matrices) by parsing the 'first line = d, then d rows/gate'
    format. Lightweight — used only to cross-check expected group order."""
    with path.open() as f:
        d = int(f.readline().strip())
        n_data_lines = sum(1 for line in f if line.strip())
    if n_data_lines % d != 0:
        raise ValueError(f"{path}: data line count {n_data_lines} not divisible by d={d}")
    return d, n_data_lines // d


# SU(2) binary icosahedral group — 120 elements.
def _bi_generators() -> list[np.ndarray]:
    e = np.eye(2, dtype=complex)
    i = 1j * np.array([[0, 1], [1, 0]], dtype=complex)
    j = 1j * np.array([[0, -1j], [1j, 0]], dtype=complex)
    k = 1j * np.array([[1, 0], [0, -1]], dtype=complex)
    phi = (1 + np.sqrt(5)) / 2
    return [e, i, j, 0.5 * (e + i + j + k), (i + j / phi + phi * k) / 2]


# SU(3) Σ(36×3) — 108 elements.  From clifford_t/genS108.py.
def _s108_generators() -> list[np.ndarray]:
    w = np.exp(2.0 * np.pi * 1j / 3.0)
    a = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
    v = (1.0 / (np.sqrt(3.0) * 1j)) * np.array(
        [[1, 1, 1], [1, w, w * w], [1, w * w, w]], dtype=complex
    )
    z = np.array([[1, 0, 0], [0, w, 0], [0, 0, w * w]], dtype=complex)
    return [a, v, z, np.linalg.inv(a), np.linalg.inv(v), np.linalg.inv(z)]


# SU(3) Σ(216×3) — 648 elements.  From clifford_t/genS648.py.
def _s648_generators() -> list[np.ndarray]:
    w = np.exp(2.0 * np.pi * 1j / 3.0)
    eps = np.exp(4.0 * np.pi * 1j / 9.0)
    a = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
    v = (1.0 / (np.sqrt(3.0) * 1j)) * np.array(
        [[1, 1, 1], [1, w, w * w], [1, w * w, w]], dtype=complex
    )
    z = np.array([[1, 0, 0], [0, w, 0], [0, 0, w * w]], dtype=complex)
    w_mat = np.array([[eps, 0, 0], [0, eps, 0], [0, 0, eps * w]], dtype=complex)
    inv = np.linalg.inv
    return [
        a, v, z, inv(a), inv(v), inv(z),
        v @ v @ v, a @ z @ a @ z, a @ inv(v) @ z, a @ v @ inv(z),
        w_mat,
    ]


CASES = [
    # (name, expected_size, reference_txt, generators_fn)
    ("BI", 120, None, _bi_generators),
    ("S108", 108, QCO_OPT / "S108.txt", _s108_generators),
    ("S648", 648, QCO_OPT / "S648.txt", _s648_generators),
]


@pytest.mark.parametrize("name,expected,ref_txt,gens_fn", CASES)
def test_close_group_sizes(name, expected, ref_txt, gens_fn):
    gens = gens_fn()
    elements = close_group(gens, max_size=expected, verbose=False)
    assert len(elements) == expected, (
        f"{name}: closure yielded {len(elements)}; expected {expected}"
    )
    u_err, d_err = verify_unitary(elements)
    assert u_err < 1e-8, f"{name}: unitarity error {u_err:.2e}"
    assert d_err < 1e-8, f"{name}: |det|-1 error {d_err:.2e}"

    if ref_txt is not None and ref_txt.exists():
        d_ref, n_ref = _count_matrices_in_qco_txt(ref_txt)
        assert d_ref == elements[0].shape[0], (
            f"{name}: dimension mismatch {d_ref} vs {elements[0].shape[0]}"
        )
        assert n_ref == len(elements), (
            f"{name}: reference {ref_txt.name} has {n_ref} elements, "
            f"closure gave {len(elements)}"
        )


def test_save_roundtrip(tmp_path):
    """Write .npy and .txt for BI, reload the .txt, confirm matrices match."""
    gens = _bi_generators()
    elements = close_group(gens, max_size=120, verbose=False)
    npy_path = tmp_path / "BI.npy"
    txt_path = tmp_path / "BI.txt"
    save_npy(elements, npy_path)
    save_qco_txt(elements, txt_path)

    # .npy round-trip
    reloaded_npy = np.load(npy_path)
    assert reloaded_npy.shape == (120, 2, 2)

    # .txt round-trip — parse as qco does.
    with txt_path.open() as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    d = int(lines[0])
    assert d == 2
    reloaded_txt = []
    rows = lines[1:]
    for i in range(0, len(rows), d):
        mat = np.array(
            [[complex(x) for x in rows[i + r].split()] for r in range(d)],
            dtype=complex,
        )
        reloaded_txt.append(mat)
    assert len(reloaded_txt) == 120
    # Each original element must appear exactly once in the reloaded set
    # (order may differ due to dict iteration). Use the closure machinery
    # to dedupe across the union and confirm no growth.
    combined = close_group(list(elements) + reloaded_txt, max_size=120, verbose=False)
    assert len(combined) == 120, (
        f"round-trip produced {len(combined) - 120} novel matrices; format error?"
    )
