#!/usr/bin/env python
# coding: utf-8

# (plot-bands-nb)=
# # Plotting Bands with `TBModel.plot_bands`

# ## Importing graphene model with `pythtb.models`
# 
# ::: {versionadded} 2.0.0
# :::

# In[1]:


from pythtb.models import graphene

# set model parameters
delta = 0.3
t = -1.0

my_model = graphene(delta=delta, t=t)


# Generate list of high-symmetry k-points to interpolate path through

# In[2]:


path = [[0, 0], [2/3, 1/3], [1/2, 1/2], [0, 0]]
# labels of the nodes
label = (r"$\Gamma $", r"$K$", r"$M$", r"$\Gamma $")
# total number of interpolated k-points along the path
nk = 200
# index of orbitals to project onto
proj_orb_idx = [0]

my_model.plot_bands(
    k_path=path, k_label=label, nk=nk, show=True, 
    proj_orb_idx=proj_orb_idx
    )

