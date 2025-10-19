#!/usr/bin/env python
# coding: utf-8

# (w90-nb)=
# # Wannier90 example with silicon

# In[6]:


from pythtb import W90 
import matplotlib.pyplot as plt
import numpy as np


# In[7]:


silicon = W90("silicon_w90", "silicon")


# In[8]:


# hard coded fermi level in eV
fermi_ev = 6.2285135


# In[9]:


# all pair distances between the orbitals
print("Shells:\n", silicon.shells())


# In[10]:


# plot hopping terms as a function of distance on a log scale
(dist, ham) = silicon.dist_hop()
fig, ax = plt.subplots()
ax.scatter(dist, np.abs(ham))
ax.set_xlabel("Distance (A)")
ax.set_ylabel(r"$H$ (eV)")
ax.set_yscale('log')


# In[11]:


# get tb model in which some small terms are ignored
my_model = silicon.model(
    zero_energy=fermi_ev,
    min_hopping_norm=0.01,
    max_distance=None,
    ignorable_imaginary_part=0.01,
)


# :::{tip}
# It is advised to save the tight-binding model to disk with the cPickle module:
# ```python
# import cPickle
# cPickle.dump(my_model, open("store.pkl", "wb"))
# ```
# Later one can load in the model from disk in a separate script with
# ```python
# my_model = cPickle.load(open("store.pkl", "rb"))
# ```
# :::

# Solve and plot on the same path as used in Wannier90
# 
# :::{hint}
# Small discrepancies in the plot may arise due to the terms that were ignored in the silicon.model function call above.
# :::

# In[12]:


w90_kpt, w90_evals, w90_kdist = silicon.w90_bands(return_kdist=True)


# In[13]:


int_evals = my_model.solve_ham(w90_kpt)


# In[14]:


fig, ax = plt.subplots()

ax.plot(list(range(w90_evals.shape[0])), w90_evals - fermi_ev, "k-", zorder=0)
ax.plot(list(range(int_evals.shape[0])), int_evals, "r--", zorder=1)

ax.set_xlim(0, int_evals.shape[0] - 1)
ax.set_xlabel("K-path from Wannier90")
ax.set_ylabel("Band energy (eV)")

