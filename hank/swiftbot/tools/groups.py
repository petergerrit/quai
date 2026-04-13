"""Named finite subgroups of U(d), the Stage-1 tool for SWIFTbot.

Each entry in REGISTRY has:
    * name, d, expected_size
    * either a txt_file (loaded from qco-main_opt/, the canonical .txt format
      written by qco-main/npy_to_qco.py) or an inline_key pointing to a raw
      generator-list builder.

Groups sourced from .txt are loaded exactly (byte-for-byte identical to what
qco-main consumed during the Słowik-Dulian-Sawicki paper's runs); groups
without a canonical .txt are closed on demand via clifford_t/genGROUP.

`register_custom()` lets the LLM supervisor (or a user) add an ad-hoc group
at runtime without modifying this file.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[2]  # .../hank/
CLIFFORD_T = REPO / "clifford_t"
QCO_TXT_DIR = REPO / "qco-main_opt"  # canonical .txt location

# Expose clifford_t/ so we can import the closure engine.
if str(CLIFFORD_T) not in sys.path:
    sys.path.insert(0, str(CLIFFORD_T))

from genGROUP import close_group  # noqa: E402


class GroupSpec(BaseModel):
    """Static metadata for a registered finite subgroup of U(d)."""

    name: str
    d: int = Field(..., gt=0)
    expected_size: int = Field(..., gt=0)
    source: Literal["txt", "inline"]
    txt_file: str | None = None       # filename under QCO_TXT_DIR
    inline_key: str | None = None     # key into _INLINE_GEN_FNS
    projective: bool = False          # if True, close in PU(d) = U(d)/U(1)
    notes: str = ""                   # free-text provenance

    def model_post_init(self, _):
        if self.source == "txt" and self.txt_file is None:
            raise ValueError(f"{self.name}: source=txt requires txt_file")
        if self.source == "inline" and self.inline_key is None:
            raise ValueError(f"{self.name}: source=inline requires inline_key")


_INLINE_GEN_FNS: dict[str, Callable[[], list[np.ndarray]]] = {}


def _inline_generators(key: str):
    def deco(fn: Callable[[], list[np.ndarray]]):
        _INLINE_GEN_FNS[key] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Inline generator builders (for groups without a canonical .txt)
# ---------------------------------------------------------------------------

@_inline_generators("hurwitz")
def _hurwitz_gens() -> list[np.ndarray]:
    """12-element Hurwitz group in PU(2) — projectively closed A4.

    WARNING: this key is a NAMING COLLISION hotspot. Three different objects
    get called "Hurwitz" in the quantum-gate literature:

    1. Our "hurwitz" (registered here): projective A4 ⊂ PU(2), |C| = 12.
       The SU(2) lift is the binary tetrahedral 2T = 24 elements, same abstract
       group order as the Pauli-Clifford (also 24) but distinct embedding.
       Used as the base group in Parzanchevski-Sarnak super-golden constructions.

    2. "Hurwitz golden gates" in Parzanchevski-Sarnak / Sarnak 2015: a specific
       generating set whose closure is this same group 1 (or a covering thereof).
       NOT the same as the 120-element group below.

    3. The binary icosahedral group 2I = |C| = 120, ubiquitously mislabelled
       "Hurwitz" in QCO-style papers because Hurwitz's quaternion theory
       underlies its golden-gate status. Registered in SWIFTbot under the
       key "BI", NOT "hurwitz". The Kubischta et al. 2I transversal-gate
       code has transversal group == our "BI", not our "hurwitz".

    If you see the name "Hurwitz" in a paper and it reports |C| = 120, it
    means our BI. If it reports |C| = 12 or 24, it means this group.

    Generators below are raw (not unit-normalised); get_group() normalises
    to PU(2) with projective=True per the registry spec.
    """
    return [
        np.array([[1j, 0], [0, -1j]], dtype=complex),
        np.array([[1, 1], [1j, -1j]], dtype=complex),
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _spec_txt(name: str, d: int, size: int, fn: str, notes: str = "") -> GroupSpec:
    return GroupSpec(
        name=name, d=d, expected_size=size, source="txt", txt_file=fn, notes=notes
    )


def _spec_inline(
    name: str, d: int, size: int, key: str, *, projective: bool = False, notes: str = ""
) -> GroupSpec:
    return GroupSpec(
        name=name,
        d=d,
        expected_size=size,
        source="inline",
        inline_key=key,
        projective=projective,
        notes=notes,
    )


REGISTRY: dict[str, GroupSpec] = {
    # --- d = 2 (SU(2)) ---
    "BI":        _spec_txt("BI",        2, 120, "BI.txt",       notes="binary icosahedral (2I); double cover of A5"),
    "BO":        _spec_txt("BO",        2,  48, "BO.txt",       notes="binary octahedral (2O)"),
    "BT":        _spec_txt("BT",        2,  24, "BT.txt",       notes="binary tetrahedral (2T)"),
    "clifford":  _spec_txt("clifford",  2,  24, "clifford.txt", notes="1-qubit Clifford group"),
    "hurwitz":   _spec_inline(
        "hurwitz", 2, 12, "hurwitz",
        projective=True,
        notes=(
            "1-qubit Hurwitz = A4 ⊂ PU(2). Closed projectively (paper convention, "
            "|C|=12). The SU(2) lift is 2A4 = 24 matrices — distinct embedding "
            "but isomorphic to BT as abstract groups."
        ),
    ),
    # --- d = 3 (SU(3)) — the Σ-series of crystallographic subgroups ---
    "S60":       _spec_txt("S60",       3,   60, "S60.txt"),
    "S108":      _spec_txt("S108",      3,  108, "S108.txt",    notes="Σ(36×3)"),
    "S216":      _spec_txt("S216",      3,  216, "S216.txt",    notes="Σ(72×3)"),
    "S648":      _spec_txt("S648",      3,  648, "S648.txt",    notes="Σ(216×3)"),
    "S1080":     _spec_txt("S1080",     3, 1080, "S1080.txt",   notes="Σ(360×3)"),
    # --- d = 4 (SU(4)) — lowercase-s prefix in source .txt files ---
    "s60":       _spec_txt("s60",       4,   60, "s60.txt"),
    "s60x4":     _spec_txt("s60x4",     4,  240, "s60x4.txt"),
    "s120x41":   _spec_txt("s120x41",   4,  480, "s120x41.txt"),
    "s120x42":   _spec_txt("s120x42",   4,  480, "s120x42.txt"),
    "s720x4":    _spec_txt("s720x4",    4, 2880, "s720x4.txt"),
    "s7f":       _spec_txt("s7f",       4, 5040, "s7f.txt",     notes="order 7! = 5040"),
}


_CACHE: dict[str, np.ndarray] = {}


def list_groups(d: int | None = None) -> list[GroupSpec]:
    """All registered group specs, optionally filtered by qudit dimension d."""
    specs = list(REGISTRY.values())
    if d is not None:
        specs = [s for s in specs if s.d == d]
    return sorted(specs, key=lambda s: (s.d, s.expected_size, s.name))


def get_group(name: str) -> np.ndarray:
    """Return the group element array of shape (|C|, d, d), complex dtype.

    Load from the canonical .txt if available; otherwise close the inline
    generator list. Results are cached in-process.
    """
    if name in _CACHE:
        return _CACHE[name]
    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(
            f"unknown group '{name}'. Registered: {sorted(REGISTRY)}"
        )
    if spec.source == "txt":
        arr = _load_qco_txt(QCO_TXT_DIR / spec.txt_file, spec.d)
    else:
        gens = _INLINE_GEN_FNS[spec.inline_key]()
        gens_su = [_normalize_to_sud(g) for g in gens]
        els = close_group(
            gens_su,
            max_size=spec.expected_size,
            projective=spec.projective,
            verbose=False,
        )
        arr = np.asarray(els, dtype=complex)
    _check_shape(arr, spec)
    _CACHE[name] = arr
    return arr


def register_custom(
    name: str,
    generators: list[np.ndarray],
    *,
    expected_size: int | None = None,
    projective: bool = False,
    notes: str = "",
) -> np.ndarray:
    """Close `generators` on demand and register the result under `name`.

    Useful for ad-hoc groups proposed by the LLM supervisor without editing
    this file. The resulting elements are cached; subsequent get_group(name)
    returns the same array.
    """
    if not generators:
        raise ValueError("need at least one generator")
    gens_su = [_normalize_to_sud(g) for g in generators]
    d = gens_su[0].shape[0]
    els = close_group(
        gens_su, max_size=expected_size, projective=projective, verbose=False
    )
    arr = np.asarray(els, dtype=complex)
    spec = GroupSpec(
        name=name,
        d=d,
        expected_size=len(els),
        source="inline",
        inline_key=f"__custom__{name}",
        projective=projective,
        notes=notes or "registered via register_custom",
    )
    REGISTRY[name] = spec
    _CACHE[name] = arr
    return arr


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Generating sets (for BFS word-tree enumeration)
# ---------------------------------------------------------------------------

_H_qubit = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S_qubit = np.array([[1, 0], [0, 1j]], dtype=complex)


def _hurwitz_minimal_gens() -> list[np.ndarray]:
    g1 = np.array([[1j, 0], [0, -1j]], dtype=complex)   # already SU(2)
    g2 = np.array([[1, 1], [1j, -1j]], dtype=complex)
    # Normalise g2 to SU(2).
    det = np.linalg.det(g2)
    g2_su = g2 / det ** 0.5
    return [g1, g2_su, g2_su.conj().T]


_GENERATING_SETS: dict[str, list[np.ndarray]] = {
    "clifford": [_H_qubit, _S_qubit, _S_qubit.conj().T],
    "hurwitz":  _hurwitz_minimal_gens(),
}


def get_generating_set(name: str) -> list[np.ndarray]:
    """Minimal generating set for BFS word-tree enumeration.

    Distinct from `get_group(name)` which returns the full closure. The
    generating set is what the word-tree BFS multiplies by at each step —
    branching factor is len(generating_set) + 2 (for T, T†)."""
    if name not in _GENERATING_SETS:
        raise KeyError(
            f"no canonical generating set for {name!r}. "
            f"Registered: {sorted(_GENERATING_SETS)}"
        )
    return [m.copy() for m in _GENERATING_SETS[name]]


def list_generating_bases() -> list[str]:
    return sorted(_GENERATING_SETS)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _normalize_to_sud(M: np.ndarray) -> np.ndarray:
    """Divide out det to land in SU(d). Uses the principal d-th root of det.

    For unitary M the result has |det|=1 and arg(det) mapped into [-π/d, π/d]
    — which is what SU(d) requires. Agnostic to d.
    """
    M = np.asarray(M, dtype=complex)
    det = np.linalg.det(M)
    d = M.shape[0]
    return M / det ** (1.0 / d)


def _load_qco_txt(path: Path, d: int) -> np.ndarray:
    """Parse the 'first line = d, then d rows per matrix' QCO gate-list format.

    Produces an array of shape (n_gates, d, d), complex dtype.
    """
    with path.open() as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: empty file")
    if int(lines[0]) != d:
        raise ValueError(f"{path}: header dim {lines[0]} != expected {d}")
    data_lines = lines[1:]
    if len(data_lines) % d != 0:
        raise ValueError(
            f"{path}: {len(data_lines)} data lines not divisible by d={d}"
        )
    mats = []
    for i in range(0, len(data_lines), d):
        mat = np.array(
            [[complex(x) for x in data_lines[i + r].split()] for r in range(d)],
            dtype=complex,
        )
        mats.append(mat)
    return np.asarray(mats, dtype=complex)


def _check_shape(arr: np.ndarray, spec: GroupSpec) -> None:
    want = (spec.expected_size, spec.d, spec.d)
    if arr.shape != want:
        raise RuntimeError(
            f"{spec.name}: loaded shape {arr.shape} != expected {want}"
        )
