#!/usr/bin/env python3
"""
test_optimized.py - Test the optimized SU(3) representation code.

Usage:
    python test_optimized.py [--t T] [--samples N]

Examples:
    python test_optimized.py --t 10 --samples 5
    python test_optimized.py --t 15 --samples 1
"""

import argparse
import time
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t', type=int, default=10, help='t-design parameter (default: 10)')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples (default: 1)')
    parser.add_argument('--gates_path', default='S60.txt', help='Path to gates file')
    args = parser.parse_args()

    # Check feasibility
    if args.t > 25:
        print(f"WARNING: t={args.t} may be very slow or run out of memory!")
        print("Recommended: t ≤ 20 for reasonable performance.")
        resp = input("Continue anyway? [y/N] ")
        if resp.lower() != 'y':
            sys.exit(0)

    print("=" * 60)
    print(f"Testing optimized SU(3) code with t={args.t}, samples={args.samples}")
    print("=" * 60)

    # Import optimized modules
    from scripts_optimized import (
        t_design_weights, filter_mirror_weights, get_gates_from_file, get_random_SU
    )
    from representation_optimized import (
        SURepresentationOptimized as SURepresentation,
        suRepresentationOptimized as suRepresentation,
    )
    from scipy.linalg import norm

    # Load gates
    print(f"\nLoading gates from {args.gates_path}...")
    d, f_gates = get_gates_from_file(args.gates_path)
    print(f"  Dimension: {d}")
    print(f"  Group elements: {len(f_gates)}")

    # Generate weights
    print(f"\nGenerating t={args.t} design weights for SU({d})...")
    weights = t_design_weights(d, args.t)
    proj_weights = [w for w in weights if suRepresentation.is_projective(w)]
    weights = filter_mirror_weights([list(w) for w in proj_weights])
    
    dims = [suRepresentation.weight_to_dim(w) for w in weights]
    print(f"  Representations: {len(weights)}")
    print(f"  Max dimension: {max(dims)}")
    print(f"  Total dimension: {sum(dims):,}")

    # Build representations
    print("\nBuilding representations...")
    start = time.time()
    reps = []
    for i, w in enumerate(weights):
        rep = SURepresentation(w)
        reps.append(rep)
        if (i + 1) % 20 == 0 or i == len(weights) - 1:
            print(f"  Built {i + 1}/{len(weights)} representations...")
    build_time = time.time() - start
    print(f"  Total build time: {build_time:.2f}s")

    # Run samples
    print(f"\nRunning {args.samples} sample(s)...")
    all_deltas = []
    
    for sample_idx in range(args.samples):
        print(f"\n--- Sample {sample_idx + 1}/{args.samples} ---")
        
        # Generate random gate and conjugate with group
        random_gate = get_random_SU(d)
        gates = [g @ random_gate @ g.conj().T for g in f_gates]
        
        # Compute norms for each representation
        sample_start = time.time()
        sample_norms = []
        
        for rep in reps:
            repr_gates = [rep(g) for g in gates]
            T = sum(repr_gates) / len(repr_gates)
            T_arr = T.toarray() if hasattr(T, 'toarray') else np.array(T)
            delta = norm(T_arr, 2)
            sample_norms.append(delta)
        
        sample_time = time.time() - sample_start
        max_delta = max(sample_norms)
        all_deltas.append(max_delta)
        
        print(f"  Max delta: {max_delta:.6f}")
        print(f"  Sample time: {sample_time:.2f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  t = {args.t}")
    print(f"  Representations: {len(weights)}")
    print(f"  Samples: {args.samples}")
    print(f"  Build time: {build_time:.2f}s")
    if len(all_deltas) > 1:
        print(f"  Delta range: [{min(all_deltas):.6f}, {max(all_deltas):.6f}]")
        print(f"  Mean delta: {np.mean(all_deltas):.6f}")
    else:
        print(f"  Max delta: {all_deltas[0]:.6f}")
    print("\n✓ Test completed successfully!")


if __name__ == '__main__':
    main()
