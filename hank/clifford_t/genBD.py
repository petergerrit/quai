#!/usr/bin/env python3

# Generates the binary dihedral group BD_{2n}. Not all of these are
# `crystal-like' subgroups! The crystal-like ones are given by n=2.

from numpy import *
from numpy.linalg import norm
import sys

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

n = int(sys.argv[1])

e = matrix([[1,0],[0,1]])
i = 1j*matrix([[0,1],[1,0]])
j = 1j*matrix([[0,-1j],[1j,0]])
k = 1j*matrix([[1,0],[0,-1]])

gen = [e, j, cos(pi/n) * e + sin(pi/n) * i]
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

for A in els:
	for B in els:
		M = A*B
		for i in range(l):
			if equal(M, els[i]):
				print(i, end=' ')
	print('')

