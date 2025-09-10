#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pythtb import TBModel, WFArray, Mesh, Lattice
from pythtb.models.haldane import haldane

import matplotlib.pyplot as plt


# In[2]:


# tight-binding parameters
delta = 1
t = 1
t2 = -0.3

model = haldane(delta, t, t2)
print(model) 


# In[3]:


k_path = [[0, 0], [2/3, 1/3], [.5, .5], [1/3, 2/3], [0, 0], [.5, .5]]
k_label = (r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $', r'$M$')
(k_vec, k_dist, k_node) = model.k_path(k_path, 101, report=False)

model.plot_bands(k_path, k_label=k_label, nk=501, scat_size=2, proj_orb_idx=[1], cmap='plasma')


# In[4]:


mesh = Mesh(dim_k=2, axis_types=['k', 'k'])
mesh.build_grid(shape=(20, 20))
print(mesh)


# In[5]:


wfa = WFArray(model, mesh)
wfa.solve_mesh()


# In[6]:


bflux = wfa.berry_flux(state_idx=[0], plane=(0,1))


# In[7]:


chern = wfa.chern_num(state_idx=[0], plane=(0,1))
print(f"Chern # occupied: {chern.real: .1f}")


# In[8]:


mesh_cart = mesh.grid @ model.recip_lat_vecs
KX, KY= mesh_cart[..., 0], mesh_cart[..., 1]


# In[9]:


im = plt.pcolormesh(KX, KY, bflux, cmap='plasma', shading='gouraud')
plt.colorbar(label=r'$\Omega(\mathbf{k})$')


# In[10]:


kpts = mesh.flat


# In[11]:


bflux_kubo = model.berry_curvature(kpts, occ_idxs=[0])[0, 1].reshape(mesh.shape_k)


# In[12]:


im = plt.pcolormesh(KX, KY, bflux_kubo.real, cmap='plasma', shading='gouraud')
plt.colorbar(label=r'$\Omega(\mathbf{k})$')


# In[ ]:




