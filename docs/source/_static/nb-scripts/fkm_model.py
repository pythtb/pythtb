#!/usr/bin/env python
# coding: utf-8

# (fkm-nb)=
# # Three-dimensional Fu-Kane-Mele model
# 
# :::{seealso}
# Fu, Kane and Mele, PRL 98, 106803 (2007)
# :::

# In[1]:


from pythtb import TBModel, WFArray, Mesh, Lattice
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


import plotly.io as pio
# switch to an HTML‐based renderer that MyST-NB understands
pio.renderers.default = "notebook"    # or "notebook", "browser", etc.


# In[3]:


def set_model(t, dt, soc):

  # set up Fu-Kane-Mele model
  lat_vecs = [[0, 1/2, 1/2], [1/2, 0, 1/2],[1/2, 1/2, 0]]
  orb_vecs = [[0, 0, 0],[1/4, 1/4, 1/4]]
  lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0,1,2])
  model = TBModel(lat, nspin=2)

  # spin-independent first-neighbor hops
  for lvec in ([0,0,0],[-1,0,0],[0,-1,0],[0,0,-1]):
    model.set_hop(t,0,1,lvec)
  model.set_hop(dt,0,1,[0,0,0],mode="add")

  # spin-dependent second-neighbor hops
  lvec_list=([1,0,0],[0,1,0],[0,0,1],[-1,1,0],[0,-1,1],[1,0,-1])
  dir_list=([0,1,-1],[-1,0,1],[1,-1,0],[1,1,0],[0,1,1],[1,0,1])
  for j in range(6):
    spin=np.array([0.]+dir_list[j])
    model.set_hop( 1.j*soc*spin,0,0,lvec_list[j])
    model.set_hop(-1.j*soc*spin,1,1,lvec_list[j])

  return model


# In[4]:


t = 1.0      # spin-independent first-neighbor hop
dt = 0.4     # modification to t for (111) bond
soc = 0.125  # spin-dependent second-neighbor hop

my_model = set_model(t, dt, soc)
print(my_model)
my_model.visualize_3d()


# ## Band structure

# In[6]:


nodes = [
    [0,0,0], [0, 1/2, 1/2], [1/4, 5/8, 5/8],
    [1/2, 1/2, 1/2],[3/4, 3/8, 3/8],[1/2, 0, 0]
    ]
label = (r'$\Gamma$',r'$X$',r'$U$',r'$L$',r'$K$',r'$L^\prime$')
my_model.plot_bands(k_path=nodes, nk=101, ktick_labels=label)


# ## Wannier flow

# Obtain eigenvectors on 2D grid on slices at fixed $k_3$. 
# 
# .. Note:: 
#    Physical $(k_1, k_2, k_3)$ have python indices ``(0, 1, 2)``.
# 
# We choose two slices at $k_3 = 0$ and $k_3 = \pi/2$. The following code block sets up the $k$-points for these two slices. Then, we construct a ``Mesh`` object to hold these $k$-points by using the ``build_custom`` method. This method allows us to create a mesh with arbitrary $k$-points. 
# 
# .. Warning:: 
#    When using ``build_custom`` on a semi-full k-mesh, i.e. one that does includes all k-points in the first Brillouin zone except for the boundary points, we must be sure to use `Mesh.wind_bz()` to tell the `WFArray` object to properly handle the periodic boundary conditions when computing overlaps. Here, we include the boundary points which is typically the safer option.

# In[7]:


# number of k-points along each direction in 2D grid
nk = 101  # choose nk odd when including endpoint to include k_i = 1/2, and nk even when excluding endpoint

# 0 <= k_i < 1 in reduced coordinates
# To include endpoint (k_i = 1), use endpoint=True
k_vals = np.linspace(0, 1, nk, endpoint=True) # <--- include endpoint
# k_vals = np.linspace(0, 1, nk, endpoint=False) # <--- exclude endpoint

k_points = np.zeros((nk, nk, 2, 3))
for j, k2 in enumerate([0, 1/2]):
  for idx0, k0 in enumerate(k_vals):
    for idx1, k1 in enumerate(k_vals):
      k_points[idx0, idx1, j, :] = [k0, k1, k2]

mesh = Mesh(dim_k=3, axis_types=['k', 'k', 'k'])
mesh.build_custom(points=k_points)
# mesh.wind_bz(0, 0) <--- used if excluding endpoint
# mesh.wind_bz(1, 1) <--- used if excluding endpoint
print(mesh)


# Solve for wavefunctions on mesh with `WFArray`

# In[8]:


wfa = WFArray(my_model, mesh)
wfa.solve_mesh()


# Compute hybrid Wannier functions

# In[9]:


hwfc = wfa.berry_phase(mu=1, state_idx=[0,1], contin=True, berry_evals=True)/(2*np.pi)


# In[10]:


# initialize plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

labels = [r'$\kappa_3$=0', r'$\kappa_3$=$\pi/2$']
for j in range(2):
  ax[j].set_xlim([0, 1])
  ax[j].set_xticks([0, 1/2, 1])
  ax[j].set_xlabel(r"$\kappa_1/2\pi$")
  ax[j].set_ylim(-0.5, 1.5)
  ax[j].text(0.08, 0.60, labels[j], size=12, bbox=dict(facecolor='w', edgecolor='k'))

  for n in range(2):
    for shift in [-1, 0, 1]:
      ax[j].plot(np.linspace(0, 1, nk), hwfc[:, j, n]+shift, color='k')

ax[0].set_ylabel(r"HWF center $\bar{s}_2$")

