#!/usr/bin/env python3

# Generates S(648)

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
	#		l.append(B*A)
	return l

w=exp(2.0*pi*1j/3.0)
eps=exp(4.0*pi*1j/9.0)
a= matrix([[0,1,0],[0,0,1],[1,0,0]])
v= 1.0/(sqrt(3.0)*1j)*matrix([[1,1,1],[1,w,w*w],[1,w*w,w]])
z= matrix([[1,0,0],[0,w,0],[0,0,w*w]])
w= matrix([[eps,0,0],[0,eps,0],[0,0,eps*w]])


#gen = [a,v,z,w]
#gen = [a,v,z,w,inv(a),inv(v),inv(z),inv(w)]
gen = [a,v,z,w,inv(a),inv(v),inv(z),inv(w),v*v*v,a*z*a*z,a*inv(v)*z,a*v*inv(z),w*w*w*w,a*v*z*w]
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

