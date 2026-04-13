"""Validate the SWIFTbot qco parser against qco-main_opt's aggregated qt_results files.

For each row of qt_results_*_t50.txt, open the referenced norms file and confirm
that SWIFTbot's (δ, Q_T) matches to machine precision.

This validates the entire parse + compute chain with zero new compute.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from swiftbot.tools import qco

REPO = Path(__file__).resolve().parents[2]  # .../hank/
QCO_OPT = REPO / "qco-main_opt"

# (aggregated-results file, data subdir) — both must exist for the test to run.
# The combined qt_results_t50.txt aggregates BI + BO + BT, so we give it all
# three data dirs. Each row's file is looked up across the list in order.
FIXTURES = [
    ("qt_results_BI_t50.txt",   ["data_BI"]),
    ("qt_results_t50.txt",      ["data_BI", "data_BO", "data_BT"]),
]

# Reasonable numerical tolerance for reproducing aggregated Q_T from raw norms.
# qt_results rounds δ to 18 decimals and Q_T to 10; we compute in double
# precision from the raw 2-line norms file, so agreement should be near ulp.
DELTA_ATOL = 1e-15
QT_ATOL = 1e-9


def _parse_qt_results(path: Path) -> list[dict]:
    """Read a qt_results file: tab/space-separated columns starting with a '#' header line."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8:
                continue
            group, size, exttype, extval, t, delta, q_t, fname = fields[:8]
            rows.append(
                {
                    "group": group,
                    "size": int(size),
                    "exttype": exttype,
                    "extval": extval,
                    "t": int(t),
                    "delta": float(delta),
                    "qt": float(q_t),
                    "file": fname,
                }
            )
    return rows


@pytest.mark.parametrize("results_name,data_names", FIXTURES)
def test_reproduce_qt_results(results_name: str, data_names: list[str]) -> None:
    results_file = QCO_OPT / results_name
    data_dirs = [QCO_OPT / n for n in data_names]
    if not results_file.exists():
        pytest.skip(f"missing fixture: {results_file}")
    data_dirs = [d for d in data_dirs if d.exists()]
    if not data_dirs:
        pytest.skip(f"none of the expected data dirs exist: {data_names}")

    rows = _parse_qt_results(results_file)
    assert rows, f"no rows parsed from {results_file}"

    mismatches: list[str] = []
    n_checked = 0
    for r in rows:
        norms_path = next((d / r["file"] for d in data_dirs if (d / r["file"]).exists()), None)
        if norms_path is None:
            continue  # row's file not present in any of the provided data dirs
        n_checked += 1

        meta = qco.parse_filename(norms_path)
        assert meta is not None, f"failed to parse filename: {norms_path}"
        assert meta.group == r["group"], f"group mismatch for {r['file']}"
        assert meta.t == r["t"], f"t mismatch for {r['file']}"
        assert meta.exttype == r["exttype"], f"exttype mismatch for {r['file']}"

        delta = qco.read_norms_file(norms_path)
        qt = qco.compute_qt(delta, r["size"])

        if abs(delta - r["delta"]) > DELTA_ATOL:
            mismatches.append(
                f"{r['file']}: δ parsed {delta!r} vs aggregated {r['delta']!r}"
            )
        if not math.isfinite(qt) or abs(qt - r["qt"]) > QT_ATOL:
            mismatches.append(
                f"{r['file']}: Q_T computed {qt!r} vs aggregated {r['qt']!r}"
            )

    assert n_checked > 0, f"no norms files found under {data_dirs}"
    assert not mismatches, "\n".join(
        [f"{len(mismatches)} rows disagree with aggregated results; first 20:", *mismatches[:20]]
    )
    # Emit a line the reporter can surface so we know how much was actually validated.
    print(f"\nvalidated {n_checked}/{len(rows)} rows from {results_name} across {data_names}")


def test_q_opt_known_values() -> None:
    # Sanity checks for Kesten-McKay. For |C|=120 (BI), 2√119/120 ≈ 0.0910...
    assert qco.q_opt(120) == pytest.approx(
        math.log(120) / math.log(120 / (2 * math.sqrt(119)))
    )
    # Asymptotic floor: Q_opt → 2 as |C| → ∞.
    assert qco.q_opt(10_000) < 2.5


def test_compute_qt_edge_cases() -> None:
    assert math.isinf(qco.compute_qt(0.0, 120))
    assert math.isnan(qco.compute_qt(1.0, 120))
    assert math.isnan(qco.compute_qt(1.5, 120))
    # δ well below 1 gives finite Q_T matching the formula.
    assert qco.compute_qt(0.5, 24) == pytest.approx(math.log(24) / math.log(2.0))


def test_parse_filename_known_patterns() -> None:
    f1 = qco.parse_filename("qcoGBIN0T50exttypeangleextval0.20253f1s0v0.0.0.txt")
    assert f1 is not None
    assert f1.group == "BI"
    assert f1.t == 50
    assert f1.exttype == "angle"
    assert f1.extval_float == pytest.approx(0.20253)
    assert not f1.is_gates

    f2 = qco.parse_filename("qcoGcliffordN2T50exttyperndf1s0v0.0.0.txt")
    assert f2 is not None
    assert f2.group == "clifford"
    assert f2.t == 50
    assert f2.exttype == "rnd"
    assert f2.extval == ""

    f3 = qco.parse_filename("qcoGBIN0T50exttypeangleextval0.2f1v0.0.0-gates.txt")
    assert f3 is not None and f3.is_gates
