#!/usr/bin/env python
# coding: utf-8

# # Wannier functions in trivial Haldane model

# In[ ]:


from pythtb import Mesh, Wannier, WFArray
from pythtb.models import haldane
import numpy as np


# ## Haldane model

# Setting up `pythTB` tight-binding model for the Haldane model parameterized by the onsite potential $\Delta$, nearest neighbor hopping $t_1$ and complex next nearest neighbor hopping $t_2$

# In[2]:


# tight-binding parameters
delta = t1 = 1
t2 = -0.1
n_super_cell = 2  # number of primitive cells along both lattice vectors

model = haldane(delta, t1, t2).make_supercell([[n_super_cell, 0], [0, n_super_cell]])


# ## `Wannier` class
# 
# The `Wannier`class contains the functions relevant for subspace selection, maximal-localization, and Wannier interpolation. We initialize it by passing the reference `Model` and number of k-points along each dimension in the mesh.

# In[3]:


nks = 20, 20 # number of k points along each dimension
mesh = Mesh(dim_k=2, axis_types=['k', 'k'])
mesh.build_grid(shape=nks)
mesh.loop_axis(0, 0)
mesh.loop_axis(1, 1)
print(mesh)


# In[4]:


wfa = WFArray(model, mesh)
wfa.solve_mesh()


# In[5]:


WF = Wannier(model, wfa)


# ## Setting up trial wavefunctions 
# 
# Now we must choose trial wavefunctions to construct our Bloch-like states. A natural choice is delta functions on the low-energy sublattice in the home cell. 
# 
# The trial wavefunctions are defined by lists of tuples specifying the trial wavefunction's probability amplitude over the orbitals `[(n, c_n), ...]`. 
# 
# $$ |t_i \ \rangle = \sum_n c_n |\phi_n\rangle $$
# 
# 
# _Note_: Normalization is handled internally so the square of the amplitudes do not need to sum to $1$. Any orbitals not specified are taken to have zero amplitude.

# In[ ]:


# model specific constants
n_orb = model.norb  # number of orbitals
n_occ = int(n_orb/2)  # number of occupied bands (assume half-filling)
low_E_sites = np.arange(0, n_orb, 2)  # low-energy sites defined to be indexed by even numbers

# defining the trial wavefunctions
tf_list = [ [(orb, 1)] for orb in low_E_sites] 
n_tfs = len(tf_list)

print(f"Trial wavefunctions: {tf_list}")
print(f"# of Wannier functions: {n_tfs}")
print(f"# of occupied bands: {n_occ}")
print(f"Wannier fraction: {n_tfs/n_occ}")


# In[7]:


WF.set_trial_wfs(tf_list)
WF.trial_wfs


# In[8]:


WF.num_twfs


# ## Projection step

# To obtain the initial Bloch-like states from projection we call the method `optimal_alignment` providing the trial wavefunctions we specified and the band-indices to construct Wannier functions from. This performs the operations,
# 1. Projection $$ (A_{\mathbf{k}})_{mn} = \langle \psi_{m\mathbf{k}} | t_n \rangle$$
# 2. SVD $$ A_{\mathbf{k}} = V_{\mathbf{k}} \Sigma_{\mathbf{k}} W_{\mathbf{k}}^{\dagger} $$
# 3. Unitary rotation$$ |\tilde{\psi}_{n\mathbf{k}} \rangle = \sum_{m\in \text{band idxs}} |\psi_{m\mathbf{k}} \rangle (V_{\mathbf{k}}W_{\mathbf{k}}^{\dagger})_{mn} $$
# 4. Fourier transformation $$  |\mathbf{R} n\rangle = \sum_{\mathbf{k}} e^{-i\mathbf{k}\cdot \mathbf{R}} |\tilde{\psi}_{n\mathbf{k}} \rangle  $$

# In[9]:


WF.optimal_alignment(band_idxs=list(range(n_occ)))


# This will already gives us quite localized Wannier functions. We can see their spreads by calling the function `report`.

# In[10]:


WF.report()


# In[11]:


WF.report()


# We can also directly access the attributes

# In[12]:


print(WF.spread)
print(WF.omega_i)
print(WF.omega_tilde)
print(WF.centers)

omega_tilde_ss = WF.omega_tilde


# We can visualize the Wannier density using `plot_density`. We specify which Wannier function to look at with `Wan_idx`.

# In[13]:


WF.plot_density(Wan_idx=1)
WF.plot_decay(Wan_idx=1, fit_rng=[5,20])
WF.plot_centers()


# ## Maximal Localization
# 
# _Maximal localization_ finds the optimal unitary rotation that minimizes the gauge dependent spread $\widetilde{\Omega}$ using the Marzari-Vanderbilt algorithm from PhysRevB.56.12847. 
# 
# To do so we call the `max_loc` function, specifying the following
# - `eps` is the step size for gradient descent
# - `iter_num` is the number of iterations before the calculation stops
# - Optionally we can set `tol` specifying the minimum change in the spread at subsequent iterations before convergence
# - For additional control we specify `grad_min` which sets the minimum gradient of the spread before convergence.

# In[14]:


iter_num = 1000

WF.max_localize(eps=1e-3, iter_num=iter_num, tol=1e-10, grad_min=1e-10, verbose=True)


# Now let's see how the localization improved.

# In[15]:


WF.report()

omega_tilde_ml = WF.omega_tilde
print()
print(f"Spread lowered by: {omega_tilde_ss-omega_tilde_ml}")

