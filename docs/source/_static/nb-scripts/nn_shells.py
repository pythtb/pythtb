#!/usr/bin/env python
# coding: utf-8

# (graphene-nn)=
# # Graphene n'th nearest neighbor hopping
#
# This graphene model calculation illustrates a case where one can use `TBModel.set_nn_hops` to get the n'th nearest neighbor hoppings.

# In[1]:


from pythtb import TBModel, Lattice
import numpy as np


# In[2]:


lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]

lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])


# ## Nearest neighbor hopping

# In[3]:


nn_model = TBModel(lat)
nn_model.set_nn_hops([-np.exp(-1)], [1])
print(nn_model)
nn_model.visualize()


# ## N'th nearest neighbor hoppings

# In[15]:


nnn_model = TBModel(lat)

N = 5
hops = [-np.exp(-t) for t in range(N)]
shells = [i + 1 for i in range(N)]

nnn_model.set_nn_hops(hop_amps=hops, nn_shells=shells, mode="set")


# In[16]:


print(nnn_model)
nnn_model.visualize()


# In[17]:


path = [[0, 0], [2 / 3, 1 / 3], [1 / 2, 1 / 2], [0, 0]]
label = (r"$\Gamma $", r"$K$", r"$M$", r"$\Gamma $")
nk = 100

fig, ax = nn_model.plot_bands(path, label, nk, bands_label="NN", fig=None, ax=None)
nnn_model.plot_bands(
    path, label, nk, bands_label=f"{N}'th NN", fig=fig, ax=ax, ls="--", lc="r"
)
ax.set_title("Graphene Tight-Binding Model")
