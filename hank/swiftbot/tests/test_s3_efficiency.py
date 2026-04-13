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
import os
import subprocess
import sys
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


@pytest.mark.skipif(not s3.MAIN_PY.exists(), reason="qco-main_opt/main.py missing")
def test_conj_cache_matches_direct_path(tmp_path: Path) -> None:
    """Conjugation cache (default ON) must produce identical δ to the direct path.

    Uses a fixed-angle extension so the run is deterministic — no per-sample
    RNG, so OFF and ON sample the *same* Π(g·rg·g†) up to floating-point.
    Tolerance is generous (1e-10) to leave headroom for accumulated round-off
    in the conjugation chain across all 24 Clifford elements.
    """
    work_off = tmp_path / "off"
    work_on = tmp_path / "on"
    work_off.mkdir(); work_on.mkdir()

    def run(work_dir: Path, cache_on: bool) -> list[list[float]]:
        argv = [
            sys.executable, str(s3.MAIN_PY),
            "-d", "2", "-t", "5", "-sample_size", "2",
            "-gates_path", str(s3.QCO_DIR / "clifford.txt"),
            "-n_of_generators", "1",
            "-fixed_gate_angle", "0.25",  # P(π/4)
            "-base_rep_cache", "1" if cache_on else "0",
        ]
        proc = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        norms = [p for p in work_dir.glob("qcoG*.txt") if not p.stem.endswith("-gates")]
        assert len(norms) == 1
        with norms[0].open() as f:
            lines = f.readlines()
        # line 0 is header; lines 1.. are per-sample δ rows
        return [[float(x) for x in ln.split()] for ln in lines[1:]]

    off = run(work_off, cache_on=False)
    on = run(work_on, cache_on=True)
    assert len(off) == len(on) == 2, (off, on)
    for s, (r_off, r_on) in enumerate(zip(off, on)):
        max_diff = max(abs(a - b) for a, b in zip(r_off, r_on))
        assert max_diff < 1e-10, (
            f"OFF/ON deltas diverge at sample {s}: max diff {max_diff:.3e}\n"
            f"OFF={r_off}\nON={r_on}"
        )


@pytest.mark.skipif(not s3.MAIN_PY.exists(), reason="qco-main_opt/main.py missing")
def test_pi_disk_cache_hit_matches_miss(tmp_path: Path) -> None:
    """Two subprocess invocations sharing a Π(g) cache dir must match a
    no-cache baseline to machine precision. Covers the {miss, hit} path and
    catches serialization round-trip bugs."""
    cache_dir = tmp_path / "pi_cache"
    baseline_work = tmp_path / "baseline"
    first_work = tmp_path / "first"
    second_work = tmp_path / "second"
    for d in (baseline_work, first_work, second_work):
        d.mkdir()

    def run(work_dir: Path, *, use_cache: bool) -> list[list[float]]:
        env = dict(os.environ)
        if use_cache:
            env["QCO_PI_CACHE_DIR"] = str(cache_dir)
        else:
            env.pop("QCO_PI_CACHE_DIR", None)
        argv = [
            sys.executable, str(s3.MAIN_PY),
            "-d", "2", "-t", "5", "-sample_size", "2",
            "-gates_path", str(s3.QCO_DIR / "clifford.txt"),
            "-n_of_generators", "1",
            "-fixed_gate_angle", "0.25",
        ]
        proc = subprocess.run(
            argv, cwd=work_dir, capture_output=True, text=True, timeout=60, env=env,
        )
        assert proc.returncode == 0, proc.stderr
        norms = [p for p in work_dir.glob("qcoG*.txt") if not p.stem.endswith("-gates")]
        assert len(norms) == 1
        with norms[0].open() as f:
            lines = f.readlines()
        return [[float(x) for x in ln.split()] for ln in lines[1:]]

    baseline = run(baseline_work, use_cache=False)
    miss = run(first_work, use_cache=True)    # populates the cache
    hit = run(second_work, use_cache=True)    # loads from cache

    cached_files = list(cache_dir.glob("v*_w*.pkl"))
    assert cached_files, f"no Pi(g) cache files written under {cache_dir}"

    for label, rows in (("miss", miss), ("hit", hit)):
        for s, (r_base, r_cached) in enumerate(zip(baseline, rows)):
            max_diff = max(abs(a - b) for a, b in zip(r_base, r_cached))
            assert max_diff < 1e-10, (
                f"{label} diverges at sample {s}: max diff {max_diff:.3e}"
            )


@pytest.mark.skipif(not s3.MAIN_PY.exists(), reason="qco-main_opt/main.py missing")
def test_in_process_matches_subprocess(tmp_path: Path) -> None:
    """evaluate_extension(in_process=True) must agree with the subprocess path
    to machine precision on a deterministic (fixed-angle) case."""
    spec = ExtensionSpec(kind="angle", params={"theta": math.pi / 4}, rationale="")
    with Cache(tmp_path / "cache_sp.db", run_id="sp") as cache_sp:
        sp_records = s3.evaluate_extension(
            spec, "clifford", t=5, sample_size=2, cache=cache_sp,
            work_dir=tmp_path / "sp_work", timeout_s=60,
            in_process=False,
        )
    with Cache(tmp_path / "cache_ip.db", run_id="ip") as cache_ip:
        ip_records = s3.evaluate_extension(
            spec, "clifford", t=5, sample_size=2, cache=cache_ip,
            work_dir=tmp_path / "ip_work",
            in_process=True,
        )
    assert len(sp_records) == len(ip_records) == 2
    for r_sp, r_ip in zip(sp_records, ip_records):
        assert r_sp.delta == pytest.approx(r_ip.delta, abs=1e-10), (
            f"subprocess δ={r_sp.delta} vs in-process δ={r_ip.delta}"
        )
        assert r_sp.qt == pytest.approx(r_ip.qt, abs=1e-6)
