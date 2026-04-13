#!/usr/bin/env python3

# Generates 60x4 subgroup of su4

from numpy import *
from numpy.linalg import norm

def equal(A, B):
	return norm(A-B) < 1e-5

def matrix_key(M):
    return tuple(asarray(round(M.real, 5)).flatten().tolist()) + tuple(asarray(round(M.imag, 5)).flatten().tolist())

def extend_unique(Ms):
    seen_keys = set(matrix_key(M) for M in Ms)
    result = Ms.copy()

    for A in Ms:
        for B in Ms:
            C = A * B
            key = matrix_key(C)
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(C)

    return result

w=exp(2.0*pi*1j/3.0)
b=exp(2.0*pi*1j/7.0)
p=b+b*b+b*b*b*b
q=b*b*b+b*b*b*b*b+b*b*b*b*b*b
s=b*b+b*b*b*b*b
t=b*b*b+b*b*b*b
u=b+b*b*b*b*b*b

f1= matrix([[1,0,0,0],[0,1,0,0],[0,0,w,0],[0,0,0,w*w]])
f2= 1.0/sqrt(3.0)*matrix([[1,0,0,sqrt(2.0)],[0,-1.0,sqrt(2.0),0],[0,sqrt(2.0),1.0,0],[sqrt(2.0),0,0,-1.0]])
f3= matrix([[sqrt(3.0)/2.0,0.5,0,0],[0.5,-sqrt(3.0)/2.0,0,0],[0,0,0,1],[0,0,1,0]])
f2p= 1.0/3.0*matrix([[3.0,0,0,0],[0,-1.0,2.0,2.0],[0,2.0,-1.0,2.0],[0,2.0,2.0,-1.0]])
f3p= 0.25*matrix([[-1.0,sqrt(15.0),0,0],[sqrt(15.0),1.0,0,0],[0,0,0,4.0],[0,0,4.0,0]])
f4= matrix([[0,1.0,0,0],[1.0,0,0,0],[0,0,0,-1.0],[0,0,-1.0,0]])
sm= matrix([[1.0,0,0,0],[0,b,0,0],[0,0,b*b*b*b,0],[0,0,0,b*b]])
tm= matrix([[1.0,0,0,0],[0,0,1.0,0],[0,0,0,1.0],[0,1.0,0,0]])
wm= 1.0/1j/sqrt(7.0)*matrix([[p*p,1.0,1.0,1.0],[1.0,-q,-p,-p],[1.0,-p,-q,-p],[1.0,-p,-p,-q]])
cm= matrix([[1.0,0,0,0],[0,1.0,0,0],[0,0,w,0],[0,0,0,w*w]])
dm= matrix([[w,0,0,0],[0,w,0,0],[0,0,w,0],[0,0,0,1.0]])
vm= 1.0/1j/sqrt(3.0)*matrix([[sqrt(3.0)*1j,0,0,0],[0,1.0,1.0,1.0],[0,1.0,w,w*w],[0,1.0,w*w,w]])
fm= matrix([[0,0,-1.0,0],[0,1.0,0,0],[-1.0,0,0,0],[0,0,0,-1.0]])
fp=(1.0+1j)/sqrt(2.0)*matrix([[1.0,0,0,0],[0,1.0,0,0],[0,0,0,1.0],[0,0,1.0,0]])
fpp=matrix([[0,1.0,0,0],[-1.0,0,0,0],[0,0,0,1.0],[0,0,-1.0,0]])
gen = [tm,cm,dm,vm,fm]
els = extend_unique(gen)

l = len(gen)

while True:
    l = len(els)
    print("Elements:", l)
    els = extend_unique(els)
    if len(els) == l:
        break

print(l)

trs = []
for i in range(l):
	trs.append(-els[i].trace()[0,0].real)
print(' '.join(str(x) for x in trs))

trs = []
for i in range(l):
        trs.append(-els[i].trace()[0,0].imag)
print(' '.join(str(x) for x in trs))

#for A in els:
#    for B in els:
#        M = A*B
#        for i in range(l):
#            if equal(M, els[i]):
#                print(i, end=' ')
#                break
#    print('')

matrices_array = array([asarray(M) for M in els])

print(f"\nSaving {len(els)} matrices of shape {matrices_array.shape} to towerpower.npy")
save('towerpower.npy', matrices_array)
print("Saved successfully!")

loaded = load('towerpower.npy')
print(f"Verification - loaded array shape: {loaded.shape}")
print(f"First matrix:\n{loaded[0]}")
print(f"Last matrix:\n{loaded[-1]}")
