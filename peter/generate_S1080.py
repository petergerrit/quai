#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import import_ipynb
from functions import *


# We generate the 1080 element $S(1080)$ (or Valentiner) group using as generators
# \begin{equation}
#     g_{A} = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix},\text{\quad}
#     g_{F} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{pmatrix},\text{\quad}
#     g_{H} = \frac{1}{2} \begin{pmatrix} -1 & \mu_{m} & \mu_{p} \\ \mu_{m} & \mu_{p} & -1 \\
#             \mu_{p} & -1 & \mu_{m} \end{pmatrix},\quad
#     g_{Q} = \begin{pmatrix} -1 & 0 & 0 \\ 0 & 0 & -\omega \\ 0 & -\omega^{2} & 0 \end{pmatrix}
# \end{equation}
# where $\mu_{m} = \frac{1}{2}(-1-\sqrt{5})$, $\mu_{p} = \frac{1}{2}(-1+\sqrt{5})$ and $\omega=e^{\frac{2\pi}{3} i}$.
# We note that we impliment by hand a number of subgroups in the first iteration, e.g., $g_{A}^{4}= (g_{A}*g_{Q})^{6} = \mathbb{1}$.
# 
# The general procedure is to begin with the generators plus a subgroup added by hand, iterate through all pairs of subgroup elements and add any which are not already listed as elements of the group.
# We continue this procedure until we find all 1080 elements.

# In[ ]:


mu_m = (1/2)*(-1-np.sqrt(5))
mu_p = (1/2)*(-1+np.sqrt(5))
omega = np.exp(2*np.pi*1j/3)

gA = np.array([[0,1,0],[0,0,1],[1,0,0]])
gF = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
gH = (1/2)*np.array([[-1, mu_m, mu_p], [mu_m, mu_p, -1], [mu_p, -1, mu_m]])
gQ = np.array([[-1,0,0],[0,0,-omega],[0,-omega**2,0]])


# In[ ]:


# Create a subset of the group; round to 10**-6 and remove any duplicated elements.
S_1080_start = np.array([np.identity(3), *[np.linalg.matrix_power(gA, i) for i in range(1, 3)], gF, gH, gQ,\
                gF@gQ, gA@gH, gA@gF, gF@gH, gH@gQ, *[np.linalg.matrix_power(gA@gQ, i) for i in range(1, 7)]])
S_1080_start = np.round(S_1080_start, 6)
S_1080_start = np.unique(S_1080_start ,axis=0)
S_1080 = S_1080_start


# In[ ]:


while S_1080.shape[0] < 1080:
    # k = len(S_1080)
    for i in range(len(S_1080)):
        for j in range(len(S_1080)):
            # print(i, j, k)
            if not is_mem(S_1080[i]@S_1080[j], S_1080_start):
                S_1080_start = np.append(S_1080_start, np.array([S_1080[i]@S_1080[j]]), axis=0)

    S_1080_start = np.round(S_1080_start, 6)
    S_1080_start = np.unique(S_1080_start ,axis=0)
    S_1080 = S_1080_start


# In[ ]:


np.save('S_1080', S_1080)


# In[ ]:




