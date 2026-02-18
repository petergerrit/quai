#!/usr/bin/env python3

import numpy as np
import sys
import argparse
from numpy.linalg import norm

def equal(A, B, tol=1e-10):
    """Check if two matrices are equal up to tolerance"""
    return norm(A - B) < tol

def equal_up_to_phase(A, B, tol=1e-10):
    """Check if two matrices are equal up to global phase"""
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i,j]) > tol:
                phase = B[i,j] / A[i,j]
                return norm(phase * A - B) < tol
    return True

def uniq(Ms, up_to_phase=False):
    """Remove duplicate matrices"""
    unique = []
    for M in Ms:
        is_dup = any(
            equal_up_to_phase(M, U) if up_to_phase else equal(M, U)
            for U in unique
        )
        if not is_dup:
            unique.append(M)
    return unique

def generate_words(B, T, nt, up_to_phase=False):
    """
    Generate all unique matrices of the form:
    B[i0] @ T @ B[i1] @ T @ B[i2] @ ... @ T @ B[int]
    where nt counts the number of T matrices in the word.
    
    Structure: B @ (T @ B) repeated nt times
    Total length: nt+1 B matrices and nt T matrices
    """
    n_B = len(B)
    
    print(f"Generating words with {nt} T matrices...")
    print(f"Number of B matrices: {n_B}")
    print(f"Word structure: B @ T @ B @ T @ ... @ T @ B ({nt+1} B's, {nt} T's)")
    print(f"Total naive combinations: {n_B**(nt+1)}")
    print()
    
    # Start with all B matrices
    current = [b.copy() for b in B]
    
    for t in range(nt):
        print(f"  Adding T matrix {t+1}/{nt}...")
        next_level = []
        
        # For each current word, multiply by T then by each B
        for word in current:
            TB = T @ word  # Hmm wait - structure is B@T@B@T...
            for b in B:
                next_level.append(b @ T @ word)
        
        # Remove duplicates
        print(f"    Before dedup: {len(next_level)}")
        current = uniq(next_level, up_to_phase=up_to_phase)
        print(f"    After dedup:  {len(current)}")
    
    return current

def generate_words_correct(B, T, nt, up_to_phase=False):
    """
    Generate all unique matrices of the form:
    B[i0] @ T @ B[i1] @ T @ B[i2] @ ... @ T @ B[int]
    
    Built left to right:
    Start: B[i0]
    Step 1: B[i0] @ T @ B[i1]
    Step 2: B[i0] @ T @ B[i1] @ T @ B[i2]
    ...
    """
    n_B = len(B)
    
    print(f"Generating words with {nt} T matrices...")
    print(f"Number of B matrices: {n_B}")
    print(f"Word structure: B[i0] @ T @ B[i1] @ ... @ T @ B[i_nt]")
    print(f"Total naive combinations: {n_B**(nt+1)}")
    print()
    
    # Start with all B matrices as initial words
    current = [b.copy() for b in B]
    print(f"Initial (0 T's): {len(current)} words")
    
    for t in range(nt):
        next_level = []
        
        # Extend each word by appending T @ B[i]
        for word in current:
            for b in B:
                next_level.append(word @ T @ b)
        
        # Remove duplicates
        before = len(next_level)
        current = uniq(next_level, up_to_phase=up_to_phase)
        after = len(current)
        print(f"After T matrix {t+1}/{nt}: {before} -> {after} unique words")
    
    return current

def main():
    parser = argparse.ArgumentParser(
    description='Generate unique words of the form B@T@B@T@...@T@B'
    )
    parser.add_argument('file',    type=str,            help='Path to .npy file containing array of matrices B')
    parser.add_argument('tfile',   type=str,            help='Path to .npy file containing the T matrix')
    parser.add_argument('nt',      type=int,            help='Number of T matrices in the word')
    parser.add_argument('--phase', action='store_true', help='Compare matrices up to global phase')
    args = parser.parse_args()

    # Load B matrices
    print(f"Loading matrices from {args.file}...")
    B = np.load(args.file)
    print(f"Loaded {len(B)} matrices of shape {B.shape[1:]}\n")

    # Load T matrix
    print(f"Loading T matrix from {args.tfile}...")
    T = np.load(args.tfile)
    print(f"T matrix shape: {T.shape}")
    print(f"T matrix:\n{T}\n")
    
    # Generate words
    words = generate_words_correct(B, T, args.nt, up_to_phase=args.phase)
    
    print(f"\n=== Results ===")
    print(f"Total unique words with {args.nt} T matrices: {len(words)}")
    
    # Save results
    output_file = f"{args.file.replace('.npy', '')}_{args.tfile.replace('.npy', '')}_nt{args.nt}.npy"
    words_array = np.array(words)
    np.save(output_file, words_array)
    print(f"Saved to {output_file}")
    
    return words

if __name__ == '__main__':
    main()
