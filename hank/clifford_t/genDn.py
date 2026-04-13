#!/usr/bin/env python3

# Generates the dihedral group D_{2n}. 

from numpy import *
from numpy.linalg import norm
import sys

order=int(sys.argv[1])/2

def equal(A, B):
	return norm(A-B) < 1e-5

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

w=exp(1j*pi/order)
e = matrix([[1,0],[0,1]])
i = matrix([[0,-1],[1,0]])
k = matrix([[w,0],[0,conj(w)]])

gen = [e, i, k]
els = uniq(generate(gen))

l = len(gen)
while len(els) > l:
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
	print('')

