"""Parse qco-main output files and compute the δ-approximate t-design norm and Q_T.

Mirrors the logic of qco-main_opt/collect_qt.py so our pipeline can consume the
same artifacts the Słowik-Dulian-Sawicki numerics produce.

File naming (from main.py + dataStructures.py):
    qcoG<group>N<N>T<t>exttype<type>[extval<val>]f<f>[s<s>]v<v>[-gates].txt
  where N = floor(log10(sample_size)), and the data file (no -gates suffix)
  contains: first line = weight labels, subsequent lines = per-sample rows of
  max eigenvalues per weight.

Definitions (Słowik-Dulian-Sawicki, arXiv:2505.00683):
    δ(ν_S, t)  = ||T_{ν_S,t} - T_{μ,t}||_∞    (max singular value of averaged t-moment op)
    Q_T(S, ε) = log|C| / log(1/δ)              (upper bound on T-QCO)
    Q_opt    = log|C| / log(|C| / (2√(|C|-1))) (Kesten-McKay lower bound, >= 2 asymptotically)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


GROUP_SIZES: dict[str, int] = {
    "clifford": 24,
    "clifford_group": 24,
    "BI": 120,  # binary icosahedral (2I)
    "BO": 48,   # binary octahedral (2O)
    "BT": 24,   # binary tetrahedral (2T)
    "hurwitz": 12,
}


@dataclass(frozen=True)
class FileMetadata:
    group: str
    t: int
    exttype: str            # rnd | angle | angles | mat
    extval: str             # raw string; float-parse via extval_float
    extval_float: float | None
    is_gates: bool


_FNAME_RE_GROUP = re.compile(r"G([A-Za-z][A-Za-z0-9_]*?)N")
_FNAME_RE_T = re.compile(r"T(\d+)")
_FNAME_RE_EXTTYPE = re.compile(r"exttype([A-Za-z]+?)(?=extval|f\d|$)")
_FNAME_RE_EXTVAL = re.compile(r"extval([^f]+?)(?=f\d|$)")


def parse_filename(path: str | Path) -> FileMetadata | None:
    """Extract (group, t, exttype, extval, is_gates) from a qcoG*.txt filename.

    Returns None if the filename does not match the expected pattern — callers
    should treat that as a skip.
    """
    name = Path(path).stem
    t_m = _FNAME_RE_T.search(name)
    if t_m is None:
        return None
    group_m = _FNAME_RE_GROUP.search(name)
    exttype_m = _FNAME_RE_EXTTYPE.search(name)
    extval_m = _FNAME_RE_EXTVAL.search(name)
    extval = extval_m.group(1).rstrip("_") if extval_m else ""
    extval_float: float | None = None
    exttype = exttype_m.group(1) if exttype_m else "rnd"
    if exttype == "angle" and extval:
        try:
            extval_float = float(extval)
        except ValueError:
            pass
    return FileMetadata(
        group=group_m.group(1) if group_m else "unknown",
        t=int(t_m.group(1)),
        exttype=exttype,
        extval=extval,
        extval_float=extval_float,
        is_gates=Path(path).stem.endswith("-gates"),
    )


def read_norm_rows(path: str | Path) -> tuple[list[str], list[list[float]]]:
    """Return (weight_labels, per_sample_rows) from a qco norms file.

    The file format: line 1 = space-separated weight labels like '[2] [4] ...';
    line i+1 = max eigenvalues per weight for sample i. This is the per-sample
    data; for a single collapsed δ use `read_norms_file` instead.
    """
    p = Path(path)
    with p.open() as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Too few lines in {p}")
    labels = lines[0].split()
    rows: list[list[float]] = []
    for ln in lines[1:]:
        vals = [float(x) for x in ln.split()]
        if vals:
            rows.append(vals)
    if not rows:
        raise ValueError(f"No data rows in {p}")
    return labels, rows


def read_norms_file(path: str | Path) -> float:
    """Return δ = max over all samples and all weights from a qco norms file.

    Matches qco-main_opt/collect_qt.py:read_norms_file exactly (single δ,
    maximised across the whole data block). For per-sample data use
    `read_norm_rows`.
    """
    _, rows = read_norm_rows(path)
    return max(max(r) for r in rows)


def infer_group_size(group_tag: str, override: int | None = None) -> int | None:
    """Resolve |C| for a group tag. Explicit override beats the table, which
    beats a trailing-digits fallback (e.g., 'S216' → 216)."""
    if override is not None:
        return override
    for key, size in GROUP_SIZES.items():
        if key.lower() == group_tag.lower():
            return size
    m = re.search(r"(\d+)", group_tag)
    return int(m.group(1)) if m else None


def compute_qt(delta: float, group_size: int) -> float:
    """Upper bound on T-QCO: log|C| / log(1/δ). +inf if δ ≤ 0; NaN if δ ≥ 1."""
    if delta <= 0:
        return math.inf
    if delta >= 1:
        return math.nan
    return math.log(group_size) / math.log(1.0 / delta)


def q_opt(group_size: int) -> float:
    """Kesten-McKay optimal value: log|C| / log(|C| / (2√(|C|-1)))."""
    if group_size < 2:
        return math.nan
    opt_delta = 2.0 * math.sqrt(group_size - 1) / group_size
    return math.log(group_size) / math.log(1.0 / opt_delta)
