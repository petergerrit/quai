#!/usr/bin/env python

from numpy import *
import sys

p = int(sys.argv[1])

print(p)

n = array(range(p))
ReTr = - cos(2*pi*n/p)
print(' '.join(str(x) for x in ReTr))

ImTr = - sin(2*pi*n/p)
print(' '.join(str(x) for x in ImTr))

a,b = meshgrid(n,n)
print("\n".join(' '.join(str(y) for y in x) for x in (a+b)%p))
