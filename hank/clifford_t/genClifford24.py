#!/usr/bin/env python3

# Generates 24-element projective Clifford group

from numpy import *
from numpy.linalg import norm

def equal(A, B):
    # Find first non-zero entry to determine phase
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i,j]) > 1e-5:
                phase = B[i,j] / A[i,j]
                return norm(phase * A - B) < 1e-5
    return True

def uniq(Ms):
    l = []
    for M in Ms:
        eq = False
        for A in l:
            if equal(A,M):
                eq = True
        if not eq:
            l.append(M)
    return l

def generate(Ms):
    l = []
    for A in Ms:
        for B in Ms:
            l.append(A*B)
    return l

# Hadamard
H = (1/sqrt(2)) * matrix([[1, 1],
                         [1, -1]])
# Phase gate (S gate)
S = matrix([[1, 0],
           [0, 1j]])
# S dagger
S_dag = matrix([[1, 0],
               [0, -1j]])

X = matrix([[0, 1],
            [1, 0]])

gen = [H, S, S_dag, X]
els = uniq(generate(gen))

l = len(gen)

while len(els) > l:
    print("Elements: ", l)
    l = len(els)
    els = uniq(generate(els))

print(l)

trs = []
for i in range(l):
    trs.append(-els[i].trace()[0,0].real)
print(' '.join(str(x) for x in trs))

trs = []
for i in range(l):
    trs.append(-els[i].trace()[0,0].imag)
print(' '.join(str(x) for x in trs))

for A in els:
    for B in els:
        M = A*B
        for i in range(l):
            if equal(M, els[i]):
                print(i, end=' ')
                break
    print('')

matrices_array = array([asarray(M) for M in els])

print(f"\nSaving {len(els)} matrices of shape {matrices_array.shape} to clifford24.npy")
save('clifford24.npy', matrices_array)
print("Saved successfully!")

# Verify by reloading
loaded = load('clifford24.npy')
print(f"Verification - loaded array shape: {loaded.shape}")
print(f"First matrix:\n{loaded[0]}")
print(f"Last matrix:\n{loaded[-1]}")    
