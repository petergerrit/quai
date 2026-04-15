"""Classify ⟨C, T⟩ as finite, universal-likely, or infinite-but-not-dense.

Stage-2 extension — the ordinary Sawicki test in `tools.sawicki` is run on
the base group C only; it can't detect when an extension T leaves the base
group finite (e.g. Clifford + S-gate, which stays in Clifford) or when the
closure exceeds a known classification bound (e.g. Clifford + T, which is
dense in SU(2)).

This module fills that gap using:

1. **Classification of finite subgroups of SU(d)** — there's a finite list
   of possible finite orders. If `close_group(⟨C, T⟩, max_size=bound)`
   terminates under the bound, the extension is finite, not universal.

2. **Character-formula Sawicki on the closed group** — when finite, we can
   check irreducibility of the Ad representation directly.

3. **Base-group Sawicki as a fallback** — when closure exceeds the bound,
   we report `universal_likely` if the base is Ad-irreducible (Bourgain-
   Gamburd / Benoist-de-Saxcé spectral-gap density is guaranteed for
   algebraic generators in this case), else `infinite_but_not_dense`.

Classification bounds:

    CLASSIFICATION_BOUND[d] = max order of a finite subgroup of SU(d)
        d=2 → 120 (= |2I|, the binary icosahedral group)
        d=3 → 1080 (= |Σ(360×3)|, the largest exceptional Σ-subgroup)
        d=4 → 5040 (= 7!, from the s7f exotic subgroup)

For d≥3 the classification includes infinite Σ/Δ-families; the bound here
covers the exceptional groups plus our registered series. If you need to
consider larger registered subgroups, extend the dict.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from swiftbot.state import ExtensionRegime, ExtensionVerdict
from swiftbot.tools.sawicki import commutant_dimension, commutant_dimension_of_generators

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "clifford_t") not in sys.path:
    sys.path.insert(0, str(REPO / "clifford_t"))
from genGROUP import close_group     # noqa: E402


# ---------------------------------------------------------------------------
# Classification data
# ---------------------------------------------------------------------------

CLASSIFICATION_BOUND: dict[int, int] = {
    2:  120,      # |2I|
    3: 1080,      # |Σ(360×3)|
    4: 5040,      # |s7f| = 7!
}

# Named non-abelian exceptional groups by (d, order).
# Ambiguous sizes (e.g. order 24 could be 2T or S4) get disambiguating prose.
CLASSIFICATION_KNOWN: dict[int, dict[int, str]] = {
    2: {
        24: "order 24 (ambiguous: 2T = binary tetrahedral, or S4 = 1-qubit projective Clifford)",
        48: "order 48 (2O = binary octahedral / SU(2) lift of 1-qubit Clifford)",
        120: "order 120 (2I = binary icosahedral)",
    },
    3: {
        21:   "order 21 (likely Σ(21))",
        27:   "order 27 (Δ(27))",
        54:   "order 54 (Δ(54))",
        60:   "order 60 (possibly A5-analogue / S60 subgroup of SU(3))",
        108:  "order 108 (Σ(36×3))",
        216:  "order 216 (Σ(72×3))",
        648:  "order 648 (Σ(216×3))",
        1080: "order 1080 (Σ(360×3))",
    },
    4: {
        60:    "order 60 (s60)",
        240:   "order 240 (s60×4)",
        480:   "order 480 (s120×4 variant)",
        2880:  "order 2880 (s720×4)",
        5040:  "order 5040 (s7f = 7!)",
    },
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def try_close_up_to(
    matrices: Sequence[np.ndarray],
    cap: int,
    *,
    projective: bool = False,
) -> list[np.ndarray] | None:
    """Close `matrices` under multiplication. Return the closed list if its
    size is ≤ cap, else None (signals "exceeded the classification bound,
    likely infinite").

    Uses genGROUP.close_group under the hood. close_group's `max_size`
    guard raises at 4·max_size, so we pass max_size = ⌈cap/4⌉ and then
    also bail out if the returned list is larger than cap.
    """
    if not matrices:
        raise ValueError("matrices is empty")
    mats = [np.asarray(m, dtype=complex) for m in matrices]
    try:
        elements = close_group(
            mats,
            max_size=max(cap // 4, 1),
            projective=projective,
            verbose=False,
        )
    except RuntimeError:
        return None
    return elements if len(elements) <= cap else None


def identify_finite_subgroup(closed: list[np.ndarray]) -> str:
    """Heuristic name for a finite closed matrix group. Matches by order
    against the CLASSIFICATION_KNOWN dictionary; falls back to generic
    "finite subgroup of SU(d), order N"."""
    if not closed:
        return "empty"
    n = len(closed)
    d = closed[0].shape[0]
    if n == 1:
        return "trivial (identity only)"
    named = CLASSIFICATION_KNOWN.get(d, {}).get(n)
    if named is not None:
        return named
    if _is_abelian(closed):
        return f"abelian group of order {n} (likely cyclic Z_{n} in SU({d}))"
    return f"finite subgroup of SU({d}), order {n} (not in SWIFTbot's classification catalog)"


def _is_abelian(mats: Sequence[np.ndarray], *, n_pairs: int = 10, atol: float = 1e-8) -> bool:
    """Cheap abelianness check via `n_pairs` random pairs."""
    n = len(mats)
    if n <= 1:
        return True
    rng = random.Random(0)
    for _ in range(n_pairs):
        i, j = rng.sample(range(n), 2)
        if not np.allclose(mats[i] @ mats[j], mats[j] @ mats[i], atol=atol):
            return False
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extension_verdict(
    base_closure: Sequence[np.ndarray],
    extension: np.ndarray,
    d: int,
    *,
    cap: int | None = None,
    projective: bool = False,
) -> ExtensionVerdict:
    """Classify ⟨base_closure, extension⟩ as finite / universal_likely /
    infinite_but_not_dense / unknown.

    Args:
        base_closure: full matrix list of the base group C (closed under
            multiplication). Must be the actual group, not just generators
            — the character formula in `commutant_dimension` requires this.
        extension: a single d×d SU(d) matrix.
        d: qudit dimension.
        cap: classification-bound override. Default: CLASSIFICATION_BOUND[d].
        projective: close modulo global phase (relevant when base_closure
            is stored as projective representatives and the extension
            produces SU(d)-level phase differences).

    Returns:
        ExtensionVerdict with regime + closure_size + identified_group + ...
    """
    if d not in CLASSIFICATION_BOUND and cap is None:
        raise ValueError(
            f"d={d} has no classification bound in SWIFTbot. "
            f"Pass `cap` explicitly if you want to proceed."
        )
    if cap is None:
        cap = CLASSIFICATION_BOUND[d]

    seeds: list[np.ndarray] = list(base_closure) + [
        np.asarray(extension, dtype=complex),
        np.asarray(extension, dtype=complex).conj().T,
    ]
    closed = try_close_up_to(seeds, cap, projective=projective)

    if closed is not None:
        try:
            cdim = commutant_dimension(closed)
        except RuntimeError as exc:
            return ExtensionVerdict(
                regime="unknown",
                d=d,
                closure_size=len(closed),
                identified_group=identify_finite_subgroup(closed),
                notes=f"Finite closure but commutant_dim failed: {exc}",
            )
        return ExtensionVerdict(
            regime="finite",
            d=d,
            closure_size=len(closed),
            identified_group=identify_finite_subgroup(closed),
            commutant_dim=cdim,
            irreducible=(cdim == 1),
            notes=(
                f"Closure terminated at order {len(closed)} under "
                f"cap={cap}; extension leaves the group finite, hence "
                f"NOT universal in SU({d})."
            ),
        )

    # Closure exceeded the classification bound → infinite subgroup.
    try:
        base_cdim = commutant_dimension(list(base_closure))
    except RuntimeError as exc:
        return ExtensionVerdict(
            regime="unknown",
            d=d,
            closure_size=None,
            notes=(
                f"Closure exceeded cap={cap} (likely infinite), and base "
                f"group commutant_dim could not be computed: {exc}. "
                f"Base may not be a closed finite group."
            ),
        )
    if base_cdim == 1:
        return ExtensionVerdict(
            regime="universal_likely",
            d=d,
            closure_size=None,
            commutant_dim=base_cdim,
            irreducible=True,
            notes=(
                f"Closure of ⟨C, T⟩ exceeds cap={cap} (largest known finite "
                f"subgroup of SU({d})). Base C has irreducible Ad, so by the "
                f"classification of finite subgroups of SU({d}) and Benoist-"
                f"de-Saxcé/Bourgain-Gamburd spectral-gap theorems, ⟨C, T⟩ is "
                f"dense in SU({d}) for algebraic generators — i.e. universal."
            ),
        )

    # Ad_C is reducible. But Ad_{⟨C,T⟩} may still be irreducible if T mixes
    # C's invariant subspaces — the Sawicki criterion is about the generated
    # group S = C ∪ {T, T†}, not C alone. Check the commutant on the full
    # generating set via the non-closed-set algorithm.
    T = np.asarray(extension, dtype=complex)
    try:
        full_cdim = commutant_dimension_of_generators(
            list(base_closure) + [T, T.conj().T]
        )
    except RuntimeError as exc:
        return ExtensionVerdict(
            regime="unknown",
            d=d,
            closure_size=None,
            commutant_dim=base_cdim,
            notes=(
                f"Ad_C reducible (commutant_dim={base_cdim}); closure of "
                f"⟨C, T⟩ exceeds cap={cap}; commutant-on-generators failed: "
                f"{exc}."
            ),
        )
    if full_cdim == 1:
        return ExtensionVerdict(
            regime="universal_likely",
            d=d,
            closure_size=None,
            commutant_dim=full_cdim,
            irreducible=True,
            notes=(
                f"Base Ad_C is reducible (commutant_dim={base_cdim}), but the "
                f"extension T lifts the invariant-subspace structure: "
                f"Comm(Ad_⟨C,T⟩) = 1 on the generating set. Closure of ⟨C, T⟩ "
                f"exceeds cap={cap}, and by Benoist-de-Saxcé / Bourgain-Gamburd "
                f"⟨C, T⟩ is dense in SU({d}) — i.e. universal. (Example: a "
                f"generic Haar SU(d) extension of a reducible-Ad base.)"
            ),
        )
    return ExtensionVerdict(
        regime="infinite_but_not_dense",
        d=d,
        closure_size=None,
        commutant_dim=full_cdim,
        irreducible=False,
        notes=(
            f"Closure of ⟨C, T⟩ exceeds cap={cap}, so NOT finite. Both Ad_C "
            f"(commutant_dim={base_cdim}) and Ad_⟨C,T⟩ (commutant_dim="
            f"{full_cdim}) are reducible, so ⟨C, T⟩ preserves a non-trivial "
            f"Ad-invariant subspace and cannot be dense in all of SU({d})."
        ),
    )
