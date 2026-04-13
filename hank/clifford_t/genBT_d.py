#!/usr/bin/env python3

# Generates the binary tetrahedral group BT.

from numpy import *
from numpy.linalg import norm

def equal(A, B):
	return norm(A-B) < 1e-5

def uniq(Ms,Mt):
    print(len(Ms),len(Mt))
    l = []
    m = []
    for index,M in enumerate(Ms):
        eq = False
        for A in l:
            if equal(A,M):
                eq = True
        if not eq:
                l.append(M)
                m.append(Mt[index])
    return l, m

def generate(Ms):
    l = []
    for A in Ms:
        for B in Ms:
            l.append(A*B)
    return l

w=exp(2.0*pi*1j/3.0)

#trivial x1
#r=matrix([[1]])
#s=matrix([[1]])

#1d x2
#r2=matrix([[w*w]])
#s2=matrix([[1]])

#1d x3
#r2=matrix([[w]])
#s2=matrix([[1]])

#real rep x4 -- one for simulating
r=matrix([[w,0],[-w*w,w*w]])
s=matrix([[0,w],[-w*w,0]])

#complex rep x5
#r2=matrix([[0,-1.0],[w*w,-w]])
#s2=matrix([[-w*w,w],[w,w*w]])

#complex rep x6
#r2=matrix([[w,w*w],[0,1]])
#s2=matrix([[-w,-w*w],[-w*w,w]])

#3drep x7
r2=matrix([[-1.0,-1.0,-1.0],[0,1.0,0],[1.0,0,0]])
s2=matrix([[-1.0,-1.0,-1.0],[0,0,1.0],[0.0,1.0,0.0]])

gen1 = [r, s]
gen2 = [r2, s2]
els1, els2 = uniq(generate(gen1),generate(gen2))

l = len(gen1)
print(l,len(els1),len(els2))
while len(els1) > l:
    l = len(els1)
    els1, els2 = uniq(generate(els1),generate(els2))

print(l)
trs = []

print("ReTr(Irrep1)")
for i in range(l):
	trs.append(round(els1[i].trace()[0,0].real,10))
print(' '.join(str(x) for x in trs))

trs = []
print("ImTr(Irrep1)")
for i in range(l):
        trs.append(round(els1[i].trace()[0,0].imag,10))
print(' '.join(str(x) for x in trs))

print("ReTr(Irrep2)")
for i in range(l):
        trs.append(round(els2[i].trace()[0,0].real,10))
print(' '.join(str(x) for x in trs))

trs = []
print("ImTr(Irrep2)")
for i in range(l):
        trs.append(round(els2[i].trace()[0,0].imag,10))
print(' '.join(str(x) for x in trs))


print("First Irrep Elements")
for k in range(r.shape[0]):
    for j in range(r.shape[1]):
        mels = []
        print(k,j)
        for i in range(l):
#        print(els[i])
            mels.append(round(els1[i][k,j],10))
        print(', '.join(str(x) for x in mels))

print("Second Irrep Elements")
for k in range(r2.shape[0]):
    for j in range(r2.shape[1]):
        mels = []
        print(k,j)
        for i in range(l):
#        print(els[i])
            mels.append(round(els2[i][k,j],10))
        print(', '.join(str(x) for x in mels))

#Note that if any irrep doesn't generate all 24 elements uniquely, the multiplication tables will have multiple elements
print("Irrep1 Mulp Table")
for A in els1:
    for B in els1:
        M = A*B
        for i in range(l):
            if equal(M, els1[i]):
                print(i, end=' ')
        print(' ',end='')
    print('')
print('')

print("Irrep2 Mulp Table")
for A in els2:
        for B in els2:
                M = A*B
                for i in range(l):
                        if equal(M, els2[i]):
                                print(i, end=' ')
                print(' ',end='')
        print('')

