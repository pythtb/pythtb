#!/usr/bin/env python
# coding: utf-8

# (graphene-nb)=
# # Graphene band structure
# 
# This is a toy model of a two-dimensional graphene sheet.

# In[1]:


from pythtb import TBModel 
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# define lattice vectors
lat = [[1, 0], [1/2, np.sqrt(3) / 2]]
# define coordinates of orbitals
orb = [[1/3, 1/3], [2/3, 2/3]]

# make two dimensional tight-binding graphene model
my_model = TBModel(2, 2, lat, orb)

# set model parameters
delta = 0.0
t = -1.0

# set on-site energies
my_model.set_onsite([-delta, delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t, 0, 1, [0, 0])
my_model.set_hop(t, 1, 0, [1, 0])
my_model.set_hop(t, 1, 0, [0, 1])

print(my_model)


# ## Generating k-points from `TBModel`
# 
# Generate list of k-points following a segmented path in the BZ list of nodes (high-symmetry points) using `TBModel.k_path`.
# 
# Outputs:
# - k_vec: list of interpolated k-points
# - k_dist: horizontal axis position of each k-point in the list
# - k_node: horizontal axis position of each original node

# In[ ]:


path = [[0, 0], [2/3, 1/3], [1/2, 1.2], [0, 0]]
label = (r"$\Gamma $", r"$K$", r"$M$", r"$\Gamma $")
nk = 121

k_vec, k_dist, k_node = my_model.k_path(path, nk)


# ## Band structure
# 
# We compute the band structure by solving the Hamiltonian at each k-point along the specified path.

# In[ ]:


evals = my_model.solve_ham(k_vec)


# In[4]:


fig, ax = plt.subplots()

ax.set_xlim(k_node[0], k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(label)

for n in range(len(k_node)):
    ax.axvline(x=k_node[n], linewidth=0.5, color="k")

ax.set_title("Graphene band structure")
ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy")

# plot bands
ax.plot(k_dist, evals)

