"""Curated catalog of quantum error-correcting codes + transversal-gate groups.

Scope (per the agent design decision: tractable codes only):
    * Qubit stabilizer codes with known transversal-gate classification
      (Anderson-Jochym-O'Connor, arXiv:1409.8320; Grier-Schaeffer).
    * Kubischta's exotic non-stabilizer codes with transversal 2I / 2O.
    * Qutrit triorthogonal codes from the 2025 Quantum paper for qutrit MSD.
    * One or two ternary Golay entries.

For base groups not covered here we return an empty list with a pointer to
errorcorrectionzoo.org — the caller (stage 4 in the supervisor) annotates
those with `research_needed=True`.

Cross-validation: the `transversal_groups` strings MUST be a subset of
swiftbot.tools.groups.REGISTRY keys; tested in tests/test_codes.py.
"""
from __future__ import annotations

from typing import Iterable

from swiftbot.state import CodeRecord

# ---------------------------------------------------------------------------
# Curated catalog
# ---------------------------------------------------------------------------

# Each entry cites a concrete paper. We stay conservative: only include
# transversal-group assignments that the literature states explicitly.
_CODES: list[CodeRecord] = [
    # --- Qubit stabilizer codes ---
    CodeRecord(
        name="Steane [[7,1,3]]",
        n=7, k=1, distance=3, qudit_dim=2,
        transversal_groups=["clifford", "BO"],
        stabilizer=True,
        reference=(
            "Steane, quant-ph/9601029; "
            "Denys-Leverrier, PRL 133, 240603 (2024) [arXiv:2306.11621] for 2O transversality"
        ),
        notes=(
            "All single-qubit Cliffords transversal; first CSS triple-error-detecting code. "
            "The SU(2) lift of the single-qubit Clifford group is 2O=BO (48 elements); "
            "the projective group 'clifford' and its lift 'BO' both tag this entry."
        ),
    ),
    CodeRecord(
        name="5-qubit perfect [[5,1,3]]",
        n=5, k=1, distance=3, qudit_dim=2,
        transversal_groups=["BT"],
        stabilizer=True,
        reference=(
            "Laflamme-Miquel-Paz-Zurek, PRL 77, 198 (1996); "
            "Denys-Leverrier, PRL 133, 240603 (2024) [arXiv:2306.11621] for 2T transversality"
        ),
        notes=(
            "Logical single-qubit Clifford transversal group is cyclic (Gottesman). "
            "However, at the multi-qubit codeblock level the binary tetrahedral "
            "2T=BT acts transversally via the Denys-Leverrier covariant encoding."
        ),
    ),
    CodeRecord(
        name="Reed-Muller [[15,1,3]]",
        n=15, k=1, distance=3, qudit_dim=2,
        transversal_groups=["clifford"],
        stabilizer=True,
        reference="Knill-Laflamme-Zurek, Science 279 (1998); Bravyi-Kitaev 2005",
        notes=(
            "Transversal T-gate; union with the transversal-Clifford subgroup of "
            "other RM codes gives Clifford+T. Hosts Bravyi-Kitaev 15-to-1 distillation."
        ),
    ),
    CodeRecord(
        name="Bravyi-Haah [[49,1,5]] triorthogonal",
        n=49, k=1, distance=5, qudit_dim=2,
        transversal_groups=["clifford"],
        stabilizer=True,
        reference="Bravyi-Haah, PRA 86, 052329 (2012)",
        notes="Triorthogonal; transversal T and CCZ via Reed-Muller punctures.",
    ),
    # --- Exotic / non-stabilizer qubit codes ---
    CodeRecord(
        name="Kubischta 2I-icosahedral family ((7,2,3))",
        n=7, k=2, distance=3, qudit_dim=2,
        transversal_groups=["BI"],
        stabilizer=False,
        reference="Kubischta & Teixeira, PRL 131, 240601 (2023) [arXiv:2305.07023]",
        notes="Nonadditive quantum code with transversal binary icosahedral group 2I = 120.",
    ),
    CodeRecord(
        name="Kubischta 2O family",
        n=None, k=None, distance=None, qudit_dim=2,
        transversal_groups=["BO"],
        stabilizer=False,
        reference="Kubischta & Teixeira, arXiv:2305.07023 (and related work)",
        notes=(
            "Family admitting transversal binary octahedral 2O = 48 action. "
            "Parameters (n,k,d) family-dependent; see paper for specific instances."
        ),
        research_needed=True,
    ),
    # --- Qutrit stabilizer / triorthogonal ---
    CodeRecord(
        name="Qutrit triorthogonal [[20,7,2]]_3",
        n=20, k=7, distance=2, qudit_dim=3,
        transversal_groups=[],
        stabilizer=True,
        reference="Low-overhead qutrit MSD, Quantum 9, 1768 (2025)",
        notes=(
            "Family [[9m-k,k,2]]_3; this member has yield γ ≈ 1.51. "
            "Hosts qutrit T (Howard-Vala / Campbell) magic-state distillation. "
            "No full transversal-group classification for qutrit stabilizer codes yet."
        ),
    ),
    CodeRecord(
        name="Ternary Golay [[11,1,5]]_3",
        n=11, k=1, distance=5, qudit_dim=3,
        transversal_groups=[],
        stabilizer=True,
        reference="Magic state distillation with the ternary Golay code, RSPA (2020)",
        notes="Distills the 'strange' qutrit magic state; no fully classified transversal group.",
    ),
    CodeRecord(
        name="2T-qutrit bosonic code (two-mode)",
        n=None, k=1, distance=None, qudit_dim=3,
        transversal_groups=["BT"],
        stabilizer=False,
        reference="Denys & Leverrier, Quantum 7, 1032 (2023) [arXiv:2210.16188]",
        notes=(
            "Bosonic qutrit encoded in two modes, spanned by 24 coherent states "
            "indexed by the binary tetrahedral group 2T=BT. The group 2T acts "
            "transversally on the logical qutrit via its 3-dimensional irrep; "
            "no direct classification with the qutrit Pauli-Clifford group. "
            "Useful as a C_data candidate for BT-extensions at d=3."
        ),
        research_needed=False,
    ),
    # --- Twisted unitary t-group codes (Kubischta-Teixeira 2024) —
    #     EXPLICIT qutrit codes with transversal Σ(360×3) action ---
    CodeRecord(
        name="Kubischta-Teixeira [[7,1,2]]_3 with transversal Σ(360×3)",
        n=7, k=1, distance=2, qudit_dim=3,
        transversal_groups=["S1080"],  # Σ(360×3) in our registry
        stabilizer=False,
        reference="Kubischta & Teixeira, PRL 133, 030602 (2024) [arXiv:2402.01638]",
        notes=(
            "Smallest instance of the χ_3 family: 7 physical qutrits encode "
            "1 logical qutrit at distance 2, detecting any single error, with "
            "the full 1080-element exceptional subgroup Σ(360×3) acting "
            "transversally. First published explicit qutrit code hosting a "
            "Σ-series subgroup transversally. Family extends to n ≡ 1 (mod 3), "
            "n ≥ 7, all at distance 2."
        ),
    ),
    CodeRecord(
        name="Kubischta-Teixeira [[5,1,2]]_3 with transversal Σ(360×3)",
        n=5, k=1, distance=2, qudit_dim=3,
        transversal_groups=["S1080"],
        stabilizer=False,
        reference="Kubischta & Teixeira, PRL 133, 030602 (2024) [arXiv:2402.01638]",
        notes=(
            "Smallest instance of the χ_4 family: 5 physical qutrits encode "
            "1 logical qutrit at distance 2, Σ(360×3) transversal. Family "
            "extends to n ≡ 2 (mod 3), n ≥ 5. Together with the [[7,1,2]]_3 "
            "χ_3 family, these are the first published explicit transversal "
            "pairings for a d=3 exceptional Σ-series subgroup."
        ),
    ),
    CodeRecord(
        name="Twisted unitary t-group codes (framework for other subgroups)",
        n=None, k=None, distance=None, qudit_dim=2,
        transversal_groups=[],
        stabilizer=False,
        reference="Kubischta & Teixeira, PRL 133, 030602 (2024) [arXiv:2402.01638]",
        notes=(
            "General framework: twisted unitary t-groups correspond to "
            "nonadditive quantum codes with distance d=t+1 and many transversal "
            "gates. Explicit instances cataloged separately: 2I (Kubischta 2I "
            "entry above) and Σ(360×3) (the two entries directly above). "
            "Generalizing to Σ(72×3), Σ(216×3), or other SU(3) / SU(4) "
            "subgroups is open — natural SWIFTbot follow-up."
        ),
        research_needed=True,
    ),
    # --- Permutation-invariant (PI) qubit codes: an AJOC loophole ---
    CodeRecord(
        name="Pollatsek-Ruskai [[7,1,3]]_PI",
        n=7, k=1, distance=3, qudit_dim=2,
        transversal_groups=[],
        stabilizer=False,
        reference=(
            "Pollatsek & Ruskai (2004); re-analysed in "
            "Aydin et al. and Kubischta-Teixeira; "
            "deployed for code-switching in Ouyang, Jing & Brennen, arXiv:2411.13142"
        ),
        notes=(
            "7-qubit permutation-invariant distance-3 qubit code with a transversal "
            "logical T gate via Z(3π/4)^{⊗7} (i.e. T† = Z(3π/4) on each physical). "
            "Non-stabilizer, so the Anderson-Jochym-O'Connor (AJOC) no-go for "
            "transversal non-Clifford gates on qubit stabilizer codes does not "
            "apply. Clifford gates are NOT transversal — fault-tolerant universality "
            "requires code-switching (Ouyang-Jing-Brennen) or state injection to a "
            "Clifford-compatible code. Member of the Aydin (b,g)-PI family at g=3, b=2."
        ),
    ),
    CodeRecord(
        name="Kubischta-Teixeira (2b+3)-qubit PI family",
        n=None, k=1, distance=3, qudit_dim=2,
        transversal_groups=[],
        stabilizer=False,
        reference=(
            "Kubischta-Teixeira PI family (ref [47] of arXiv:2411.13142); "
            "deployed by Ouyang, Jing & Brennen, arXiv:2411.13142"
        ),
        notes=(
            "(b,g=3,k)-PI family on 2b+3 physical qubits. Each member admits "
            "transversal Z(πg/b) = Z(3π/b) — i.e. tunable rational-angle logical "
            "Z-rotations (b=4 → 11-qubit code with transversal Z(3π/4), etc.). "
            "Non-stabilizer qubit codes; AJOC exclusion of rational-angle "
            "transversal P(θ) applies only to stabilizer codes, so this family "
            "provides a second fault-tolerant route (alongside programmable MSD "
            "on Clifford+T) to arbitrary rational-angle qubit rotations. "
            "Clifford gates are not transversal — Ouyang-Jing-Brennen code-switch "
            "to a Steane-like stabilizer code for the Clifford layer."
        ),
    ),
    # --- Zhang-Wu-Huang-Zeng (2025) nonadditive codes with transversal
    #     non-Clifford gates discovered via Stiefel-manifold optimization ---
    CodeRecord(
        name="Zhang et al. ((6,1,3)) with transversal 2T",
        n=6, k=1, distance=3, qudit_dim=2,
        transversal_groups=["BT"],
        stabilizer=False,
        reference=(
            "Zhang, Wu, Huang, Zeng, arXiv:2504.20847 (2025). "
            "Appendix A, single-point λ*=1 construction."
        ),
        notes=(
            "Smallest known distance-3 qubit code supporting transversal "
            "binary tetrahedral action. Derived via Stiefel-manifold search. "
            "Related ((6,1,3)) variants support C_{2k} and binary dihedral BD_4 "
            "along a continuous parameter λ*; we catalog the 2T=BT specialization "
            "whose transversal group matches our d=2 registry."
        ),
    ),
    CodeRecord(
        name="Zhang et al. ((7,1,3)) with transversal 2I",
        n=7, k=1, distance=3, qudit_dim=2,
        transversal_groups=["BI"],
        stabilizer=False,
        reference=(
            "Zhang, Wu, Huang, Zeng, arXiv:2504.20847 (2025)."
        ),
        notes=(
            "New ((7,1,3))_{2I} construction at λ*=√(3/4), distinct from "
            "Kubischta-Teixeira's λ*=0 original. Both give the same transversal "
            "group (binary icosahedral, |BI|=120). Catalogued separately "
            "because the parameter λ* affects optimality bounds and distillation "
            "threshold estimates."
        ),
    ),
    CodeRecord(
        name="Zhang et al. ((7,1,3)) with transversal 2T",
        n=7, k=1, distance=3, qudit_dim=2,
        transversal_groups=["BT"],
        stabilizer=False,
        reference=(
            "Zhang, Wu, Huang, Zeng, arXiv:2504.20847 (2025), Appendix C."
        ),
        notes=(
            "New ((7,1,3)) code with transversal binary tetrahedral action. "
            "Smaller |transversal| than 2I but a natural candidate for BT-based "
            "gate synthesis."
        ),
    ),
    CodeRecord(
        name="Zhang et al. ((7,1,3)) with transversal 2O",
        n=7, k=1, distance=3, qudit_dim=2,
        transversal_groups=["BO"],
        stabilizer=False,
        reference=(
            "Zhang, Wu, Huang, Zeng, arXiv:2504.20847 (2025), Appendix C."
        ),
        notes=(
            "((7,1,3)) with transversal binary octahedral (=full qubit Clifford "
            "lift) action. Bridges between single-qubit Clifford and non-Clifford "
            "via the same code."
        ),
    ),
    CodeRecord(
        name="Zhang et al. ((8,1,3)) with transversal T^(1/4) via BD_64",
        n=8, k=1, distance=3, qudit_dim=2,
        transversal_groups=[],  # BD_64 not in our registry
        stabilizer=False,
        reference=(
            "Zhang, Wu, Huang, Zeng, arXiv:2504.20847 (2025), Appendix E."
        ),
        notes=(
            "Binary dihedral BD_64 transversal action supports T^(1/4) gate "
            "fault-tolerantly — deepest rational-phase rotation in this family "
            "with a direct transversal implementation. The ((7,1,3))_{BD_16} "
            "variant from the same paper provides the first ((7,1,3)) supporting "
            "transversal T; BD_32 on 7 qubits supports T√T. BD_{2n} not yet in "
            "the SWIFTbot groups registry (open work)."
        ),
    ),
    # --- Kubischta-Teixeira (2025) intrinsic codes ---
    CodeRecord(
        name="Kubischta-Teixeira [[13,2,3]] PI (intrinsic {14,2,3}_{SU(2)})",
        n=13, k=2, distance=3, qudit_dim=2,
        transversal_groups=["clifford", "BO"],  # single-qubit Clifford on each logical qubit
        stabilizer=False,
        reference=(
            "Kubischta-Teixeira, arXiv:2511.14840 (2025), Example 2."
        ),
        notes=(
            "Permutation-invariant 13-qubit code encoding 2 logical qubits at "
            "distance 3. Derived as the minimal realization of the intrinsic "
            "code {14, 2, 3}_{SU(2)} (depth-3 SU(2) symmetry in the spin-13/2 "
            "irrep). The single-qubit Clifford group acts transversally on each "
            "logical qubit. Exponential [[n,2,3]] family exists for odd n≥15. "
            "The 'Schur bootstrap' certifies distance-3 protection across all "
            "SU(2)-equivariant realizations in one shot."
        ),
    ),
    CodeRecord(
        name="Kubischta-Teixeira [[4,5,2]] multi-qutrit intrinsic {27,5,2}_{SU(3)}",
        n=4, k=5, distance=2, qudit_dim=3,
        transversal_groups=[],  # A_6 logical symmetry; not in our SU(3) registry
        stabilizer=False,
        reference=(
            "Kubischta-Teixeira, arXiv:2511.14840 (2025), Example 3."
        ),
        notes=(
            "Multi-qutrit realization of the intrinsic {27, 5, 2}_{SU(3)} code: "
            "4 physical qutrits encode 5 logical qudits at distance 2. Logical "
            "symmetry group is A_6 (alternating group on 6 elements). Companion "
            "[[6,5,2]] realization also exists. First SU(3)-symmetry intrinsic "
            "code with a multi-qutrit extrinsic instance; A_6 not yet in the "
            "SWIFTbot d=3 groups registry."
        ),
    ),
    CodeRecord(
        name="Kubischta-Teixeira [[6,5,2]] multi-qutrit intrinsic {27,5,2}_{SU(3)}",
        n=6, k=5, distance=2, qudit_dim=3,
        transversal_groups=[],
        stabilizer=False,
        reference=(
            "Kubischta-Teixeira, arXiv:2511.14840 (2025), Example 3."
        ),
        notes=(
            "Alternative realization of {27, 5, 2}_{SU(3)} on 6 qutrits instead "
            "of 4. Same logical symmetry A_6 and distance 2."
        ),
    ),
    CodeRecord(
        name="Kubischta-Teixeira [[4,2,2]] PI intrinsic {5,2,2}_{SU(2)}",
        n=4, k=2, distance=2, qudit_dim=2,
        transversal_groups=[],
        stabilizer=False,
        reference=(
            "Kubischta-Teixeira, arXiv:2511.14840 (2025), Example 1."
        ),
        notes=(
            "Permutation-invariant [[4,2,2]] qubit code realizing the intrinsic "
            "{5,2,2}_{SU(2)} code (spin-2 irrep, depth 2). Other realizations "
            "include the Chuang-Leung-Yamamoto 2-mode bosonic code and a "
            "continuous CP^4 family of [[6,2,2]] qubit codes — all equivalent "
            "under the Schur bootstrap."
        ),
    ),
    # --- Herbert-Gross-Newman (2023) qutrit codes from SU(3) irreps ---
    CodeRecord(
        name="Herbert-Gross-Newman [[15,1,*]] qutrit, error-detecting",
        n=15, k=1, distance=None, qudit_dim=3,
        transversal_groups=[],  # He(3) = Heisenberg-Weyl at d=3; not in our registry
        stabilizer=False,
        reference=(
            "Herbert, Gross, Newman, arXiv:2312.00162 (2023)."
        ),
        notes=(
            "Qutrit error-detecting code embedded in SU(3) irrep (4,0) (15-dim "
            "rep, k=1 logical qutrit). Transversal Heisenberg-Weyl He(3) action "
            "via qutrit X and Z operators; protects against 'small SU(3) "
            "displacements'. Codewords: |0̄⟩=(√3|0000⟩+|1122⟩)/3 etc. A companion "
            "error-correcting [[247,1,*]] code from irrep (37,0) is impractical "
            "but exhibits the same symmetry structure; we omit it from the main "
            "catalog for tractability. Distance not formally given — the "
            "detection claim is for SU(3)-displacement errors, not Pauli weight."
        ),
    ),
    # --- Uy-Gangloff (2024) qudit codes from SU(d) irreps ---
    CodeRecord(
        name="Uy-Gangloff (d-1)^2-qudit family (smallest case d=5)",
        n=16, k=1, distance=None, qudit_dim=5,
        transversal_groups=[],
        stabilizer=False,
        reference=(
            "Uy, Gangloff, arXiv:2410.02407 (2024)."
        ),
        notes=(
            "Infinite family of qudit codes encoding one logical qudit on "
            "(d-1)^2 physical qudits for odd d ≥ 5, via SU(d) irreducible "
            "representations acting on symmetric tensor products. Transversal "
            "Heisenberg-Weyl HW(d) action by construction. At d=5 → 16 physical "
            "qudits/logical; d=7 → 36 physical/logical; d=11 → 100 physical/"
            "logical. Distance parameters scale with the SU(d) representation "
            "dimension but are not given in closed form in the main text; "
            "computing them for specific d is a natural catalog-extension task."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_all_codes() -> list[CodeRecord]:
    """Return a shallow copy of the full curated catalog."""
    return list(_CODES)


def codes_with_transversal_group(group_name: str) -> list[CodeRecord]:
    """All curated codes that admit `group_name` transversally."""
    return [c for c in _CODES if group_name in c.transversal_groups]


def codes_for_dim(qudit_dim: int) -> list[CodeRecord]:
    """All curated codes for a given qudit dimension."""
    return [c for c in _CODES if c.qudit_dim == qudit_dim]


def codes_for_group(group_name: str) -> tuple[list[CodeRecord], str]:
    """Look up codes compatible with `group_name` (Stage 4 entry point).

    Returns a (codes, notes) pair. When nothing matches, `codes` is empty
    and `notes` contains a pointer to errorcorrectionzoo.org.
    """
    hits = codes_with_transversal_group(group_name)
    if hits:
        note = f"{len(hits)} curated code(s) match base group {group_name!r}."
        return hits, note
    note = (
        f"No curated code with transversal group {group_name!r}. "
        "Research may be needed — start at https://errorcorrectionzoo.org/c/qudit_stabilizer "
        "and https://errorcorrectionzoo.org/c/qubit_stabilizer."
    )
    return [], note


def validate_registry_names(registry: Iterable[str]) -> list[str]:
    """Return any `transversal_groups` strings that are NOT in the given
    registry — used as a schema-integrity test."""
    registered = set(registry)
    bad: list[str] = []
    for c in _CODES:
        for g in c.transversal_groups:
            if g not in registered:
                bad.append(f"{c.name} → {g!r} (not in registry)")
    return bad
