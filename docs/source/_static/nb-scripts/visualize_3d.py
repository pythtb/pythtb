#!/usr/bin/env python
# coding: utf-8

# (visualize-3d-nb)=
# # Visualize 3D tight-binding model
#
# This example demonstrates how to visualize a 3D tight-binding model using the `pythtb` library.

# In[ ]:


import plotly.io as pio

# switch to an HTML‐based renderer that MyST-NB understands
pio.renderers.default = "notebook"  # or "notebook", "browser", etc.


# In[4]:


from pythtb.models import fu_kane_mele

# Reference Model
t = 1  # spin-independent first-neighbor hop
soc = 1  # spin-dependent second-neighbor hop
m = 1  # magnetic field magnitude
beta = 1  # Adiabatic parameter
fkm_model = fu_kane_mele(t, soc, m, beta)

fig = fkm_model.visualize_3d(draw_hoppings=True)
fig
