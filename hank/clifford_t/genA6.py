#!/usr/bin/env python3

# Generates A6...not confirmed to work.

from numpy import *
from numpy.linalg import norm
from numpy.linalg import inv

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
			l.append(B*A)
	return l

a= matrix([[2,0,0],[0,0,1],[0,1,0]])
b= matrix([[0,1,0],[2,0,0],[3,3,1]])


#gen = [a,v,z]
#gen = [a,v,z,inv(a),inv(v),inv(z)]
gen = [a,b,inv(a),inv(b),a*a,a*b,b*a,b*b]
#gen = [a,v,z,w,a*a,a*v,a*z,a*w,v*a,v*v,v*z,v*w,z*a,z*v,z*z,z*w,w*a,w*v,w*z,w*w,a*z*a*z,v*v*v]
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

