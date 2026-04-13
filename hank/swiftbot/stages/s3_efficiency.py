"""Stage 3: materialize an ExtensionSpec, invoke qco-main_opt, record Q_T.

Flow:
    1. Materialize the ExtensionSpec into either a concrete d×d SU(d) matrix
       (for angle/angles/mat/howard_vala) or None (for rnd, which the qco
       subprocess generates itself).
    2. Build a `QCORunSpec` carrying the exact argv + context.
    3. Invoke qco-main_opt/main.py as a subprocess in a temporary work dir
       using the current Python interpreter. mpi4py is optional in
       scripts_optimized.py, so the single-process path works fine.
    4. Parse the resulting qcoG*.txt back into per-sample QTRecords and
       persist them to the SQLite cache with provenance.

Conventions inherited from qco-main_opt:
    * -fixed_gate_angle X    — X is a phase in *units of π* (not radians).
    * -fixed_gate_angles a,b — d-1 phases, also in units of π.
    * -fixed_gate_matrix P   — path to an .npy or .txt unitary.
    * no flag               — Haar-random ensemble (exttype=rnd).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from swiftbot.kb.cache import Cache
from swiftbot.state import QTRecord
from swiftbot.supervisor import ExtensionSpec
from swiftbot.tools import groups as gmod
from swiftbot.tools import qco as qcomod


def extension_fingerprint(ext_spec: ExtensionSpec) -> str:
    """Stable short hash of an ExtensionSpec (kind + params).

    Two evaluations of the same spec have the same fingerprint; two
    different specs (even of the same kind) have different fingerprints.
    Used to uniquify qt_results rows so `evaluate_extension(rnd, C)` and
    `evaluate_extension(angle, C)` don't overwrite one another.
    """
    payload = json.dumps(
        {"kind": ext_spec.kind, "params": ext_spec.params},
        sort_keys=True, default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]

REPO = Path(__file__).resolve().parents[2]           # .../hank/
QCO_DIR = REPO / "qco-main_opt"
MAIN_PY = QCO_DIR / "main.py"
CLIFFORD_T_DIR = REPO / "clifford_t"


# ---------------------------------------------------------------------------
# Materialization: ExtensionSpec → concrete SU(d) matrix (or None for rnd)
# ---------------------------------------------------------------------------

def _normalize_sud(M: np.ndarray) -> np.ndarray:
    d = M.shape[0]
    det = np.linalg.det(M)
    return M / det ** (1.0 / d)


def materialize_extension(spec: ExtensionSpec, d: int) -> np.ndarray | None:
    """Produce a concrete d×d SU(d) matrix for the given spec.

    Returns None when the spec describes an ensemble (rnd) that qco itself
    samples; the caller should then pass no -fixed_gate_* flag.

    Parameter conventions:
        kind="angle":     params["theta"]  — radians (converted to π-units for qco).
        kind="angles":    params["phases"] — list of radians; if len = d-1, the
                          last phase is inferred so the det is 1.
        kind="mat":       params["matrix"] — d×d array-like; normalised to SU(d).
        kind="howard_vala": params z/gamma/epsilon — deferred (not implemented yet).
        kind="rnd":       nothing; returns None.
    """
    if spec.kind == "rnd":
        return None
    if spec.kind == "angle":
        theta = float(spec.params["theta"])
        diag = np.ones(d, dtype=complex)
        diag[1] = np.exp(1j * theta)
        return _normalize_sud(np.diag(diag))
    if spec.kind == "angles":
        phases = np.asarray(spec.params["phases"], dtype=float)
        if phases.size == d - 1:
            phases = np.concatenate([phases, [-phases.sum()]])
        if phases.size != d:
            raise ValueError(
                f"angles: expected d={d} or d-1={d-1} phases, got {phases.size}"
            )
        return np.diag(np.exp(1j * phases))
    if spec.kind == "mat":
        M = np.asarray(spec.params["matrix"], dtype=complex)
        if M.shape != (d, d):
            raise ValueError(f"mat: shape {M.shape} ≠ ({d},{d})")
        return _normalize_sud(M)
    if spec.kind == "howard_vala":
        # Accept several naming conventions the LLM might emit.
        # Paper uses primed symbols (z', γ', ε); LLMs often translate those
        # to z_prime / gamma_prime / epsilon. Also accept the short form.
        p = spec.params
        if "z_prime" in p or "gamma_prime" in p or "epsilon" in p:
            z = int(p.get("z_prime", p.get("z", 1)))
            gamma = int(p.get("gamma_prime", p.get("gamma", 1)))
            eps = int(p.get("epsilon", p.get("eps", 0)))
        else:
            z = int(p.get("z", 1))
            gamma = int(p.get("gamma", 1))
            eps = int(p.get("eps", 0))
        return materialize_howard_vala(d, z=z, gamma=gamma, eps=eps)
    raise ValueError(f"unknown extension kind {spec.kind!r}")


def materialize_howard_vala(
    d: int, *, z: int = 1, gamma: int = 1, eps: int = 0,
) -> np.ndarray:
    """Howard-Vala qudit π/8-like T gate (arXiv:1206.1598), normalised to SU(d).

    Wraps `clifford_t/gen_un_T_gate.sun_T_gate`. Only defined for prime d;
    raises ValueError otherwise. Defaults (z=1, gamma=1, eps=0) reproduce
    Campbell's M(p) magic gate.
    """
    if str(CLIFFORD_T_DIR) not in sys.path:
        sys.path.insert(0, str(CLIFFORD_T_DIR))
    from gen_un_T_gate import sun_T_gate         # local import — sympy side-effect
    matrix, _ = sun_T_gate(d, z=z, gamma=gamma, eps=eps)
    return _normalize_sud(np.asarray(matrix, dtype=complex))


# ---------------------------------------------------------------------------
# QCO subprocess invocation
# ---------------------------------------------------------------------------

ExtKind = Literal["rnd", "angle", "angles", "mat"]


@dataclass(frozen=True)
class QCORunSpec:
    """Fully specifies a single qco-main_opt/main.py sampling run."""
    d: int
    t: int
    sample_size: int
    gates_path: Path                                # base group .txt
    ext_kind: ExtKind
    ext_value: str | None = None                    # for angle / angles
    ext_matrix_path: Path | None = None             # for mat
    symmetric: bool = False
    n_of_generators: int = 1                        # required by qco


@dataclass
class QCORunResult:
    run_spec: QCORunSpec
    output_path: Path                               # qcoG*.txt produced
    stdout: str = ""
    stderr: str = ""


def _build_argv(spec: QCORunSpec, python: str) -> list[str]:
    argv: list[str] = [
        python, str(MAIN_PY),
        "-d", str(spec.d),
        "-t", str(spec.t),
        "-sample_size", str(spec.sample_size),
        "-gates_path", str(spec.gates_path.resolve()),
        "-n_of_generators", str(spec.n_of_generators),
    ]
    if spec.ext_kind == "angle":
        if spec.ext_value is None:
            raise ValueError("angle ext_kind requires ext_value (π-units)")
        argv += ["-fixed_gate_angle", spec.ext_value]
    elif spec.ext_kind == "angles":
        if spec.ext_value is None:
            raise ValueError("angles ext_kind requires ext_value (comma-separated, π-units)")
        argv += ["-fixed_gate_angles", spec.ext_value]
    elif spec.ext_kind == "mat":
        if spec.ext_matrix_path is None:
            raise ValueError("mat ext_kind requires ext_matrix_path")
        argv += ["-fixed_gate_matrix", str(spec.ext_matrix_path.resolve())]
    # rnd: no extra flag
    if spec.symmetric:
        argv.append("-symmetric")
    return argv


def run_qco(
    spec: QCORunSpec,
    *,
    work_dir: Path | None = None,
    python: str | None = None,
    timeout_s: float | None = None,
    verbose: bool = False,
) -> QCORunResult:
    """Run qco-main_opt/main.py as a subprocess and return the output path.

    work_dir: scratch directory for the subprocess's CWD (default: tempdir).
              The qco binary writes qcoG*.txt files into CWD.
    python:   interpreter for the subprocess (default: sys.executable).
    timeout_s: wall-clock limit (None = no limit).
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="swiftbot_qco_"))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
    argv = _build_argv(spec, python or sys.executable)
    if verbose:
        print(" ".join(argv), flush=True)
    proc = subprocess.run(
        argv,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"qco-main_opt exited with {proc.returncode}\n"
            f"argv: {argv}\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    norm_files = [
        p for p in work_dir.glob("qcoG*.txt")
        if not p.stem.endswith("-gates")
    ]
    if len(norm_files) != 1:
        raise RuntimeError(
            f"expected exactly 1 norms file, got {len(norm_files)} in "
            f"{work_dir}: {norm_files}"
        )
    return QCORunResult(
        run_spec=spec,
        output_path=norm_files[0],
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


# ---------------------------------------------------------------------------
# High-level: spec + base group → cached QTRecords
# ---------------------------------------------------------------------------

def _gates_path_for(base_group_name: str, work_dir: Path) -> Path:
    """Locate the canonical .txt for `base_group_name`, or synthesise one in
    `work_dir` from the cached matrices (for custom-registered groups)."""
    canonical = QCO_DIR / f"{base_group_name}.txt"
    if canonical.exists():
        return canonical
    sys.path.insert(0, str(REPO / "clifford_t"))
    from genGROUP import save_qco_txt            # local import to avoid cycles
    dest = work_dir / f"{base_group_name}.txt"
    save_qco_txt(list(gmod.get_group(base_group_name)), dest)
    return dest


def _run_spec_for(
    ext_spec: ExtensionSpec,
    base_group_name: str,
    *,
    d: int,
    t: int,
    sample_size: int,
    work_dir: Path,
    symmetric: bool,
    n_of_generators: int,
) -> QCORunSpec:
    gates_path = _gates_path_for(base_group_name, work_dir)
    if ext_spec.kind == "rnd":
        return QCORunSpec(
            d=d, t=t, sample_size=sample_size, gates_path=gates_path,
            ext_kind="rnd", symmetric=symmetric, n_of_generators=n_of_generators,
        )
    if ext_spec.kind == "angle":
        theta_rad = float(ext_spec.params["theta"])
        theta_pi = theta_rad / math.pi
        return QCORunSpec(
            d=d, t=t, sample_size=sample_size, gates_path=gates_path,
            ext_kind="angle", ext_value=f"{theta_pi:.10g}",
            symmetric=symmetric, n_of_generators=n_of_generators,
        )
    if ext_spec.kind == "angles":
        phases = np.asarray(ext_spec.params["phases"], dtype=float)
        if phases.size == d - 1:
            phases = np.concatenate([phases, [-phases.sum()]])
        vals = ",".join(f"{(p / math.pi):.10g}" for p in phases[:d - 1])
        return QCORunSpec(
            d=d, t=t, sample_size=sample_size, gates_path=gates_path,
            ext_kind="angles", ext_value=vals,
            symmetric=symmetric, n_of_generators=n_of_generators,
        )
    if ext_spec.kind in ("mat", "howard_vala"):
        M = materialize_extension(ext_spec, d)
        mat_path = work_dir / "ext.npy"
        np.save(mat_path, M)
        return QCORunSpec(
            d=d, t=t, sample_size=sample_size, gates_path=gates_path,
            ext_kind="mat", ext_matrix_path=mat_path,
            symmetric=symmetric, n_of_generators=n_of_generators,
        )
    raise ValueError(f"unknown extension kind {ext_spec.kind!r}")


def evaluate_extension(
    ext_spec: ExtensionSpec,
    base_group_name: str,
    *,
    t: int,
    sample_size: int = 1,
    cache: Cache,
    symmetric: bool = False,
    n_of_generators: int = 1,
    work_dir: Path | None = None,
    python: str | None = None,
    timeout_s: float | None = None,
    verbose: bool = False,
) -> list[QTRecord]:
    """Evaluate one (base_group, extension_spec) pair end-to-end.

    Caches: the base group (as registry:<name>) and one QTRecord per sample
    row, keyed by the base group's group_key. Extension provenance lives in
    the source_file (path of the qcoG*.txt produced by qco).
    """
    if base_group_name not in gmod.REGISTRY:
        raise KeyError(f"{base_group_name!r} is not a registered group")
    group_spec = gmod.REGISTRY[base_group_name]
    base_matrices = np.asarray(gmod.get_group(base_group_name))
    base_key = cache.put_group(
        base_matrices,
        name=base_group_name,
        source=f"registry:{base_group_name}",
        projective=group_spec.projective,
    )

    owned_work = work_dir is None
    if owned_work:
        work_dir = Path(tempfile.mkdtemp(prefix="swiftbot_qco_"))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    run_spec = _run_spec_for(
        ext_spec, base_group_name,
        d=group_spec.d, t=t, sample_size=sample_size,
        work_dir=work_dir, symmetric=symmetric,
        n_of_generators=n_of_generators,
    )
    result = run_qco(
        run_spec, work_dir=work_dir, python=python,
        timeout_s=timeout_s, verbose=verbose,
    )

    _, rows = qcomod.read_norm_rows(result.output_path)
    q_opt = qcomod.q_opt(group_spec.expected_size)
    fingerprint = extension_fingerprint(ext_spec)
    records: list[QTRecord] = []
    for i, row in enumerate(rows):
        delta = max(row)
        qt = qcomod.compute_qt(delta, group_spec.expected_size)
        qt_store: float | None = qt if math.isfinite(qt) else None
        rec = QTRecord(
            target_key=base_key,
            t=t,
            sample_id=i,
            ext_fingerprint=fingerprint,
            delta=delta,
            qt=qt_store,
            q_opt=q_opt,
            source_file=str(result.output_path),
        )
        cache.put_qt(rec)
        records.append(rec)
    return records
