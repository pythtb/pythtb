#!/usr/bin/env python
# coding: utf-8

# (mesh-nb)=
# # `Mesh` class
#
# This tutorial will show you how to create custom meshes and winding paths around the the combined Brillouin zone (BZ) and parameter space.

# In[6]:


from pythtb import Mesh
import numpy as np


# In[7]:


mesh = Mesh(dim_k=2, axis_types=["k"])
points = np.linspace([0, 0], [1, 1], 10, endpoint=False)  # path from (0,0) to (1, 1)
mesh.build_custom(points)
print(mesh)


# In[8]:


mesh.wind_bz(axis_idx=0, component_idx=0)  # mark as winding k_x
mesh.wind_bz(axis_idx=0, component_idx=1)  # mark as winding k_y
print(mesh)


# In[9]:


print(mesh.bz_winding_axes)
