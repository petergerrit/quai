#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools


# # Tests

# ## average value of function

# We compute the average value of $f(x)=x^{2}$ on the interval 0 to 4.  This is given by $\int_{0}^{n} dx\, \frac{x^{2}}{n}=\frac{n^{2}}{3}$.

# In[2]:


n=5


# In[3]:


# f(x)=x**2
def x_squared(x):
    return x**2


# In[4]:


# sample 10**5 points uniformly in the range 0 to 4 and append to my vals_array
vals_array = np.empty(0)
for _ in range(10**5):
    num = n*np.random.rand(1)
    vals_array = np.append(vals_array, x_squared(num))


# In[5]:


# average value of vals_array is approximately 16/3 = 5.333; uncertainty is standard deviation
# print central_val - 1 std, central_val, central_val + 1 std
mean_val = vals_array.mean()
std_val = vals_array.std()/np.sqrt(len(vals_array))
print(mean_val-std_val, mean_val, mean_val+std_val)


# In[6]:


# exact answer
(n**2)/3


# ## $\mathbb{Z}_{n}$ approximation of $U(1)$

# We compute the average value of the group element $e^{i\phi}$ in the interval where it is nearer the identity element than to any other element of $\mathbb{Z}_{n}$.  This is given by $\int_{-\pi/n}^{\pi/n}d\phi\, \frac{e^{i\phi}}{2\pi/n} = \frac{n}{\pi}\sin{\frac{\pi}{n}}$

# In[47]:


n=27


# In[48]:


def xi_r(phi):
    return np.exp(1j*phi)


# In[49]:


# sample 10**5 points uniformly in the range 0 to 4 and append to my vals_array
vals_array = np.empty(0)
for _ in range(10**5):
    phi = (2*np.pi/n)*np.random.rand(1)-(np.pi/n)
    vals_array = np.append(vals_array, xi_r(phi))


# In[50]:


vals_array.mean()


# In[51]:


(n/np.pi)*np.sin(np.pi/n)


# # Generate S(1080)

# In[27]:


omega = np.exp(2*np.pi*1j/3)
mu_p = (1/2)*(-1+np.sqrt(5))
mu_m = (1/2)*(-1-np.sqrt(5))
gA = np.array([[0,1,0],[0,0,1],[1,0,0]])
gF = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
gH = (1/2)*np.array([[-1, mu_m, mu_p], [mu_m, mu_p, -1], [mu_p, -1, mu_m]])
gQ = np.array([[-1,0,0],[0,0,-omega],[0,-omega**2,0]])


# In[28]:


# S_1080_start = np.array([np.identity(3), *[np.linalg.matrix_power(gA, i) for i in range(1, 3)], gF, gH, gQ,\
#                     gF@gQ, gA@gH, gA@gF, gF@gH, gH@gQ, *[np.linalg.matrix_power(gA@gQ, i) for i in range(1, 7)]])
# S_1080_start = np.round(S_1080_start, 6)
# # S_1080_start = np.unique(S_1080_start ,axis=0)


# In[29]:


try:
    S_1080_start=np.load('S_1080.npy')
except FileNotFoundError:
    S_1080_start = np.array([np.identity(3), *[np.linalg.matrix_power(gA, i) for i in range(1, 3)], gF, gH, gQ,\
                    gF@gQ, gA@gH, gA@gF, gF@gH, gH@gQ, *[np.linalg.matrix_power(gA@gQ, i) for i in range(1, 7)]])
    S_1080_start = np.round(S_1080_start, 6)
    S_1080_start = np.unique(S_1080_start ,axis=0)


# In[30]:


S_1080 = S_1080_start
S_1080_temp = S_1080_start


# In[31]:


S_1080.shape


# In[21]:


def is_mem(elem, array):
    is_member=False
    for item in array:
        if np.linalg.svd(elem-item).S.max() < 10**-4:
            is_member=True
            break
    return is_member


# In[22]:


for i in range(len(S_1080)):
    for j in range(len(S_1080)):
        print(i, j)
        if not is_mem(S_1080[i]@S_1080[j], S_1080_start):
            S_1080_start = np.append(S_1080_start, np.array([S_1080[i]@S_1080[j]]), axis=0)


# In[23]:


S_1080_start.shape


# In[24]:


S_1080_start=np.unique(S_1080_start, axis=0)


# In[25]:


S_1080_start.shape


# In[26]:


np.save('S_1080', S_1080_start)


# In[ ]:




