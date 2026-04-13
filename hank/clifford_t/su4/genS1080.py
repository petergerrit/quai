#!/usr/bin/env python3

# Generates S(1080)

from numpy import *
from numpy.linalg import norm

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

w=exp(2.0*pi*1j/3.0)
up=0.5*(-1.0+sqrt(5.0))
um=0.5*(-1.0-sqrt(5.0))
a= matrix([[0,1,0],[0,0,1],[1,0,0]])
f= matrix([[1,0,0],[0,-1,0],[0,0,-1]])
h= 0.5*matrix([[-1,um,up],[um,up,-1],[up,-1,um]])
q= matrix([[-1,0,0],[0,0,-w],[0,-w*w,0]])

gen = [a,f,h,q]
els = uniq(generate(gen))

l = len(gen)

while len(els) > l:
    print ("Elements: ",l)
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

