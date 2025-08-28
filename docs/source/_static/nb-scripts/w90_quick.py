#!/usr/bin/env python
# coding: utf-8

# (w90-quick-nb)=
# # Wannier90 quick example
# 
# In this example, we will demonstrate how to use the PythTB package to read the output from Wannier90 and plot the band structure of a silicon supercell.
# 
# To run the interface with Wannier90, you must first download the
# following :download:`wannier90 output example <../misc/wannier90_example.tar.gz>` 
# and unpack it with the following command in unix command
# 
# .. code-block:: bash
# 
#         tar -zxf wannier90_example.tar.gz
# 
# The example below will read the tight-binding model from the Wannier90
# calculation, create a simplified model in which some small hopping
# terms are ignored, and finally plot the interpolated band structure.

# In[1]:


from pythtb import W90  
import matplotlib.pyplot as plt


# Read output from `Wannier90` that should be in folder named "silicon_w90"

# In[2]:


silicon = W90(r"silicon_w90", r"silicon")


# Get tight-binding model without hopping terms above 0.01 eV

# In[3]:


my_model = silicon.model(min_hopping_norm=0.01)


# Solve model on a path and plot it

# In[4]:


path = [
    [0.5, 0.5, 0.5],
    [0.0, 0.0, 0.0],
    [0.5, -0.5, 0.0],
    [0.375, -0.375, 0.0],
    [0.0, 0.0, 0.0],
]
# labels of the nodes
k_label = (r"$L$", r"$\Gamma$", r"$X$", r"$K$", r"$\Gamma$")
# call function k_path to construct the actual path
(k_vec, k_dist, k_node) = my_model.k_path(path, 101)


# In[5]:


evals = my_model.solve_ham(k_vec)


# In[6]:


fig, ax = plt.subplots()

ax.plot(k_dist, evals, "k-")

for n in range(len(k_node)):
    ax.axvline(x=k_node[n], linewidth=0.5, color="k")

ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy (eV)")
ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(k_label)

