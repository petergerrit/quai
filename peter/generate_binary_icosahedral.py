#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import import_ipynb
from functions import *


# We generate the 120 element binary icosahedral group $\mathbb{BI}$ using ([source](https://web.math.princeton.edu/~le3339/autcode.pdf))
# \begin{equation}
#     M_{1} = \begin{pmatrix} -\epsilon^{3} & 0 \\ 0 & -\epsilon^{2} \end{pmatrix},\text{\quad}
#     M_{2} = \frac{1}{\sqrt{5}} \begin{pmatrix} -\epsilon + \epsilon^{4} & \epsilon^{2} - \epsilon^{3} \\
#             \epsilon^{2} - \epsilon^{3} & \epsilon - \epsilon^{4} \end{pmatrix},
# \end{equation}
# as generators, where $\epsilon = e^{\frac{2}{5} \pi i}$.
# We note that $M_{1}^{5} = M_{2}^{2} = (M_{1}*M_{2})^3 =  - \mathbb{1}$

# In[2]:


eps=np.exp(2*np.pi*1j/5)

M1 = np.array([[-eps**3, 0], [0, -eps**2]])
M2 = 5**(-1/2)*np.array([[-eps+eps**4, eps**2-eps**3], [eps**2-eps**3, eps-eps**4]])

bin_icos_start = np.array([np.identity(2),\
                          *[np.linalg.matrix_power(M1, i) for i in range(1,10)],\
                          *[np.linalg.matrix_power(M2, i) for i in range(1,4)],\
                          *[np.linalg.matrix_power(M1@M2, i) for i in range(1, 6)]])
bin_icos_start = np.round(bin_icos_start, 6)
bin_icos_start = np.unique(bin_icos_start, axis=0)
bin_icos=bin_icos_start


# In[ ]:


while bin_icos.shape[0] < 120:
    #k = len(bin_icos)
    for i in range(len(bin_icos)):
        for j in range(len(bin_icos)):
            #print(i, j, k)
            if not is_mem(bin_icos[i]@bin_icos[j], bin_icos_start):
                bin_icos_start = np.append(bin_icos_start, np.array([bin_icos[i]@bin_icos[j]]), axis=0)
    
    bin_icos_start = np.round(bin_icos_start, 6)
    bin_icos_start = np.unique(bin_icos_start, axis=0)
    bin_icos=bin_icos_start


# In[4]:


np.save('binary_icosahedral', bin_icos)


# In[ ]:




