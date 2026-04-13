"""Tests for swiftbot.stages.s3_efficiency.

We cover three layers:
  1. materialize_extension — pure function; all branches unit-tested.
  2. _build_argv / _run_spec_for — deterministic argv construction from specs.
  3. evaluate_extension — one end-to-end test that actually invokes
     qco-main_opt as a subprocess on a trivially tiny case (t=5, 1 sample).
     Marked as an integration test; skip if MAIN_PY missing.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from swiftbot.kb.cache import Cache
from swiftbot.stages import s3_efficiency as s3
from swiftbot.supervisor import ExtensionSpec


D = 2  # most tests are d=2


# ---------------------------------------------------------------------------
# Pure function: materialize_extension
# ---------------------------------------------------------------------------

def _is_sud(M: np.ndarray, atol: float = 1e-10) -> bool:
    d = M.shape[0]
    unit = np.allclose(M @ M.conj().T, np.eye(d), atol=atol)
    det_one = abs(np.linalg.det(M) - 1.0) < atol
    return unit and det_one


def test_materialize_rnd_returns_none() -> None:
    spec = ExtensionSpec(kind="rnd", params={}, rationale="baseline")
    assert s3.materialize_extension(spec, d=2) is None


def test_materialize_angle_is_sud() -> None:
    spec = ExtensionSpec(kind="angle", params={"theta": math.pi / 4}, rationale="t")
    M = s3.materialize_extension(spec, d=2)
    assert M.shape == (2, 2)
    assert _is_sud(M)


@pytest.mark.parametrize("phases", [(0.3, -0.5), (0.1, 0.2, -0.3)])
def test_materialize_angles_is_sud(phases: tuple) -> None:
    spec = ExtensionSpec(kind="angles", params={"phases": list(phases)}, rationale="")
    d = len(phases) if sum(phases) == 0 else len(phases) + 1
    M = s3.materialize_extension(spec, d=d)
    assert M.shape == (d, d)
    assert _is_sud(M)


def test_materialize_mat_normalizes() -> None:
    raw = np.array([[0, 1], [1, 0]], dtype=complex)  # det = -1
    spec = ExtensionSpec(kind="mat", params={"matrix": raw.tolist()}, rationale="")
    M = s3.materialize_extension(spec, d=2)
    assert _is_sud(M)


def test_materialize_mat_rejects_wrong_shape() -> None:
    bad = np.eye(3, dtype=complex)
    spec = ExtensionSpec(kind="mat", params={"matrix": bad.tolist()}, rationale="")
    with pytest.raises(ValueError, match="shape"):
        s3.materialize_extension(spec, d=2)


def test_materialize_howard_vala_qubit_campbell_default() -> None:
    """HV defaults reduce to standard qubit T = diag(1, e^{iπ/4}) up to phase."""
    spec = ExtensionSpec(kind="howard_vala", params={"z": 1, "gamma": 1, "eps": 0}, rationale="")
    M = s3.materialize_extension(spec, d=2)
    assert M.shape == (2, 2)
    assert _is_sud(M)
    # After SU(2) normalisation, the T gate is (phase-equivalent to)
    # diag(e^{-iπ/8}, e^{iπ/8}). Check by comparing to this projectively.
    T = np.diag([np.exp(-1j * np.pi / 8), np.exp(1j * np.pi / 8)])
    # Allow overall sign: T or -T.
    assert np.allclose(M, T, atol=1e-9) or np.allclose(M, -T, atol=1e-9)


def test_materialize_howard_vala_qutrit() -> None:
    """HV for p=3 uses 9th roots of unity (Howard-Vala Eq. 25)."""
    spec = ExtensionSpec(kind="howard_vala",
                         params={"z": 1, "gamma": 1, "eps": 0}, rationale="")
    M = s3.materialize_extension(spec, d=3)
    assert M.shape == (3, 3)
    assert _is_sud(M)


def test_materialize_howard_vala_matches_paper_eq27() -> None:
    """Paper Eq. (27): for p=3, z'=1, γ'=2, ε=0 the exponents are (0, 1, 8),
    so the raw HV matrix is diag(1, ζ, ζ^8) with ζ = e^{2πi/9}. After SU(3)
    normalisation (dividing by det^{1/3}) this should agree up to sign."""
    spec = ExtensionSpec(kind="howard_vala",
                         params={"z": 1, "gamma": 2, "eps": 0}, rationale="")
    M = s3.materialize_extension(spec, d=3)
    zeta = np.exp(2j * np.pi / 9)
    raw = np.diag([1.0 + 0j, zeta, zeta ** 8])
    det = np.linalg.det(raw)
    expected = raw / det ** (1.0 / 3)
    assert np.allclose(M, expected, atol=1e-10)


def test_materialize_howard_vala_accepts_primed_param_names() -> None:
    """The paper uses primed symbols z', γ', ε; LLMs often emit them as
    z_prime / gamma_prime / epsilon. Both naming styles should produce
    identical matrices."""
    short = ExtensionSpec(kind="howard_vala",
                          params={"z": 1, "gamma": 2, "eps": 0}, rationale="")
    primed = ExtensionSpec(kind="howard_vala",
                           params={"z_prime": 1, "gamma_prime": 2, "epsilon": 0},
                           rationale="")
    assert np.allclose(
        s3.materialize_extension(short, d=3),
        s3.materialize_extension(primed, d=3),
    )


def test_materialize_howard_vala_rejects_non_prime_d() -> None:
    spec = ExtensionSpec(kind="howard_vala", params={"z": 1}, rationale="")
    with pytest.raises(ValueError, match="prime"):
        s3.materialize_extension(spec, d=4)


# ---------------------------------------------------------------------------
# Deterministic argv construction
# ---------------------------------------------------------------------------

def test_build_argv_rnd(tmp_path: Path) -> None:
    gates = tmp_path / "clifford.txt"
    gates.write_text("2\n1 0\n0 1\n")
    spec = s3.QCORunSpec(
        d=2, t=5, sample_size=1, gates_path=gates, ext_kind="rnd",
    )
    argv = s3._build_argv(spec, python="/x/python")
    assert "-fixed_gate_angle" not in argv
    assert "-fixed_gate_matrix" not in argv
    assert "-n_of_generators" in argv
    assert str(gates.resolve()) in argv


def test_build_argv_angle(tmp_path: Path) -> None:
    gates = tmp_path / "BI.txt"
    gates.write_text("2\n1 0\n0 1\n")
    spec = s3.QCORunSpec(
        d=2, t=50, sample_size=1, gates_path=gates,
        ext_kind="angle", ext_value="0.2",
    )
    argv = s3._build_argv(spec, python="/x/python")
    assert "-fixed_gate_angle" in argv
    assert argv[argv.index("-fixed_gate_angle") + 1] == "0.2"


def test_build_argv_mat_requires_path(tmp_path: Path) -> None:
    gates = tmp_path / "c.txt"
    gates.write_text("2\n1 0\n0 1\n")
    spec = s3.QCORunSpec(
        d=2, t=5, sample_size=1, gates_path=gates, ext_kind="mat",
    )
    with pytest.raises(ValueError, match="ext_matrix_path"):
        s3._build_argv(spec, python="/x/python")


def test_build_argv_symmetric_flag(tmp_path: Path) -> None:
    gates = tmp_path / "c.txt"
    gates.write_text("2\n1 0\n0 1\n")
    spec = s3.QCORunSpec(
        d=2, t=5, sample_size=1, gates_path=gates, ext_kind="rnd", symmetric=True,
    )
    argv = s3._build_argv(spec, python="/x/python")
    assert "-symmetric" in argv


def test_run_spec_angle_converts_radians_to_pi_units(tmp_path: Path) -> None:
    ext = ExtensionSpec(kind="angle", params={"theta": math.pi / 4}, rationale="")
    spec = s3._run_spec_for(
        ext, "clifford", d=2, t=5, sample_size=1,
        work_dir=tmp_path, symmetric=False, n_of_generators=1,
    )
    # The value passed to qco is (theta / π) as a string; for π/4 that's 0.25.
    assert spec.ext_kind == "angle"
    assert float(spec.ext_value) == pytest.approx(0.25, abs=1e-12)


# ---------------------------------------------------------------------------
# Integration: actually invoke qco-main_opt once
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not s3.MAIN_PY.exists(), reason="qco-main_opt/main.py missing")
def test_evaluate_extension_rnd_tiny_case(tmp_path: Path) -> None:
    """End-to-end: Clifford + Haar-random extension, t=5, single sample.

    This exercises the full subprocess path. Runtime target: < 10s on laptop.
    """
    spec = ExtensionSpec(kind="rnd", params={}, rationale="baseline")
    with Cache(tmp_path / "cache.db", run_id="s3-it") as cache:
        records = s3.evaluate_extension(
            spec, "clifford",
            t=5, sample_size=1, cache=cache,
            work_dir=tmp_path / "qco_work",
            timeout_s=60,
        )
    assert len(records) == 1
    rec = records[0]
    assert rec.t == 5
    assert rec.sample_id == 0
    assert 0 < rec.delta < 1                 # sensible spectral norm
    assert rec.qt is not None and math.isfinite(rec.qt)
    assert rec.q_opt == pytest.approx(
        math.log(24) / math.log(24 / (2 * math.sqrt(23)))
    )
