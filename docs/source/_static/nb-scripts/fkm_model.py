#!/usr/bin/env python
# coding: utf-8

# (fkm-nb)=
# # Three-dimensional Fu-Kane-Mele model
# 
# :::{seealso}
# Fu, Kane and Mele, PRL 98, 106803 (2007)
# :::

# In[1]:


from pythtb import TBModel, WFArray, Mesh
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def set_model(t, dt, soc):

  # set up Fu-Kane-Mele model
  lat = [[0, 1/2, 1/2], [1/2, 0, 1/2],[1/2, 1/2, 0]]
  orb = [[0, 0, 0],[1/4, 1/4, 1/4]]
  model = TBModel(dim_k=3, dim_r=3, lat=lat, orb=orb, nspin=2)

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


# In[3]:


t = 1.0      # spin-independent first-neighbor hop
dt = 0.4     # modification to t for (111) bond
soc= 0.125  # spin-dependent second-neighbor hop

my_model = set_model(t,dt,soc)
print(my_model)


# ## Band structure

# In[ ]:


path = [
    [0,0,0], [0, 1/2, 1/2], [1/4, 5/8, 5/8],
    [1/2, 1/2, 1/2],[3/4, 3/8, 3/8],[1/2, 0, 0]
    ]
label = (r'$\Gamma$',r'$X$',r'$U$',r'$L$',r'$K$',r'$L^\prime$')
(k_vec, k_dist, k_node) = my_model.k_path(path, 101)

evals = my_model.solve_ham(k_vec)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(k_dist, evals, color='k')

ax.set_xlim([0, k_node[-1]])
ax.set_xticks(k_node)
ax.set_xticklabels(label)
for n in range(len(k_node)):
  ax.axvline(x=k_node[n], linewidth=0.5, color='k')
ax.set_ylabel("Energy")
ax.set_ylim(-4.9, 4.9)


# ## Wannier flow

# Construct mesh

# In[11]:


# Obtain eigenvectors on 2D grid on slices at fixed kappa_3
# Note physical (kappa_1,kappa_2,kappa_3) have python indices (0,1,2)
kappa2_values=[0, 1/2]
labs = [r'$\kappa_3$=0',r'$\kappa_3$=$\pi$']
nk = 41
dk = 1/(nk-1)

k_points = np.zeros((nk, nk, 2, 3))
for j in range(2):
  for k0 in range(nk):
    for k1 in range(nk):
      kvec = [k0*dk, k1*dk, kappa2_values[j]]
      k_points[k0, k1, j, :] = kvec

mesh = Mesh(my_model, axis_types=['k', 'k', 'k'])
mesh.build_custom(points=k_points)


# Solve for wavefunctions on mesh with `WFArray`

# In[13]:


wfa = WFArray(mesh)
wfa.solve_mesh()


# In[ ]:


wfa.get_states(flatten_spin=True).shape


# Compute hybrid Wannier functions

# In[15]:


hwfc = wfa.berry_phase(occ = [0,1], dir=1, contin=True, berry_evals=True)/(2*np.pi)


# In[17]:


# initialize plot
fig, ax = plt.subplots(1,2,figsize=(12, 6), sharey=True)

for j in range(2):
  ax[j].set_xlim([0.,1.])
  ax[j].set_xticks([0.,0.5,1.])
  ax[j].set_xlabel(r"$\kappa_1/2\pi$")
  ax[j].set_ylim(-0.5,1.5)

  for n in range(2):
    for shift in [-1.,0.,1.]:
      ax[j].plot(np.linspace(0, 1, nk), hwfc[:, j, n]+shift, color='k')
    ax[j].text(0.08,1.20,labs[j],size=12.,bbox=dict(facecolor='w',edgecolor='k'))

ax[0].set_ylabel(r"HWF center $\bar{s}_2$")

