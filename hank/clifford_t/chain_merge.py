#!/usr/bin/env python3
"""
Chain-merge group extension files.

For each file in group+extension/ matching <anything>_nt<N>.npy,
find the appropriate base file in groups/ by matching the longest
base name that is a prefix of the extension filename.

Example:
    groups/clifford24.npy
    group+extension/clifford24_diag_su2_0.0000_0.5236_nt1.npy
    -> base = clifford24, suffix = diag_su2_0.0000_0.5236

Chain:
    groups/clifford24.npy + clifford24_diag_su2_..._nt1.npy -> clifford24_diag_su2_..._upnt1.npy
    clifford24_diag_su2_..._upnt1.npy + ..._nt2.npy         -> clifford24_diag_su2_..._upnt2.npy
    ...
"""

import re
import subprocess
import sys
from pathlib import Path

# ── configuration ────────────────────────────────────────────────────────────
GROUPS_DIR    = Path("groups")
EXTENSION_DIR = Path("group+extension")
MERGE_SCRIPT  = Path("merge_sets.py")
OUTPUT_DIR    = Path("full_sets")
# ─────────────────────────────────────────────────────────────────────────────


def run_merge(in1: Path, in2: Path, out: Path):
    """Invoke merge_sets.py as a subprocess: python merge_sets.py in1 in2 out"""
    cmd = [sys.executable, str(MERGE_SCRIPT), str(in1), str(in2), str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"    [ERROR] merge_sets.py stderr:\n{result.stderr.strip()}")
        raise RuntimeError(f"merge failed: {in1.name} + {in2.name}")


def get_base_names(groups_dir: Path) -> list[str]:
    """Return stems of all .npy files in groups/, longest first."""
    names = [f.stem for f in groups_dir.glob("*.npy")]
    # sort longest-first so we match the most specific base possible
    return sorted(names, key=len, reverse=True)


def discover_groups(extension_dir: Path, base_names: list[str]) -> dict:
    """
    Scan extension_dir for files matching <full_prefix>_nt<N>.npy.
    For each file, find which base_name it starts with.
    Returns {full_prefix: (base_name, sorted [nt ints])}.
    """
    pattern = re.compile(r"^(.+)_nt(\d+)\.npy$")
    groups: dict[str, tuple[str, list[int]]] = {}

    for f in extension_dir.iterdir():
        m = pattern.match(f.name)
        if not m:
            continue
        full_prefix = m.group(1)
        nt          = int(m.group(2))

        # skip files we already produced (_upntN)
        if "_upnt" in full_prefix:
            continue

        # find the matching base name
        base = next((b for b in base_names if full_prefix.startswith(b)), None)
        if base is None:
            print(f"[WARN] no base file found for prefix '{full_prefix}', skipping")
            continue

        if full_prefix not in groups:
            groups[full_prefix] = (base, [])
        groups[full_prefix][1].append(nt)

    # sort nt lists
    return {k: (base, sorted(nts)) for k, (base, nts) in groups.items()}


def chain_merge(full_prefix: str, base_name: str, nt_values: list[int]):
    """
    Chain:
        groups/<base_name>.npy  +  <full_prefix>_nt1.npy  ->  <full_prefix>_upnt1.npy
        <full_prefix>_upnt1.npy +  <full_prefix>_nt2.npy  ->  <full_prefix>_upnt2.npy
        ...
    """
    base_file = GROUPS_DIR / f"{base_name}.npy"
    current   = base_file

    for nt in nt_values:
        nt_file  = EXTENSION_DIR / f"{full_prefix}_nt{nt}.npy"
        out_file = OUTPUT_DIR    / f"{full_prefix}_upnt{nt}.npy"

        if not nt_file.exists():
            print(f"  [WARN] file not found, stopping chain: {nt_file}")
            break

        print(f"  {current.name}  +  {nt_file.name}  ->  {out_file.name}")
        run_merge(current, nt_file, out_file)
        current = out_file

    print(f"  [DONE]")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in [GROUPS_DIR, EXTENSION_DIR, MERGE_SCRIPT]:
        if not path.exists():
            sys.exit(f"ERROR: required path not found: {path}")

    base_names = get_base_names(GROUPS_DIR)
    if not base_names:
        sys.exit(f"No .npy files found in {GROUPS_DIR}")
    print(f"Base files in groups/: {base_names}\n")

    groups = discover_groups(EXTENSION_DIR, base_names)
    if not groups:
        sys.exit("No matching _ntN.npy files found in group+extension/")

    print(f"Found {len(groups)} prefix group(s)\n")

    for full_prefix, (base_name, nt_values) in sorted(groups.items()):
        print(f"[{full_prefix}]  base={base_name}.npy  nt sequence={nt_values}")
        chain_merge(full_prefix, base_name, nt_values)
        print()


if __name__ == "__main__":
    main()
