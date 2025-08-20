#!/usr/bin/env python
# coding: utf-8

# # Reduced Wannier functions and disentanglement

# In[1]:


from pythtb import Mesh, Wannier, WFArray
from pythtb.models import haldane
import numpy as np


# ## Haldane model supercell construction
# 
# We begin by constructing the Haldane model in the topological phase, in which case the non-zero Chern number of the occupied bands enforces a topological obstruction to constructing exponentially localized Wannier functions that respect the lattice symmetries.

# In[2]:


# tight-binding parameters
delta = 1
t1 = 1
t2 = -0.4
prim_model = haldane(delta, t1, t2)

print(f"Chern number: {prim_model.chern():0.3f}")


# To circumvent the obstruction, we use the procedure of constructing "reduced Wannier" functions that are localized in a smaller subspace of the original Hilbert space. Since the occupied space is one-dimensional, there exists no subspace. This is why we must construct a supercell, folding the bands back into the first Brillouin zone, to obtain more occupied bands to choose from. Here we will use a 2x2 supercell, which will give us 4 occupied bands to work with.

# In[3]:


n_super_cell = 2
model = prim_model.make_supercell([[n_super_cell, 0], [0, n_super_cell]])
model.report(show=True, short=True)


# We construct the `WFArray` and diagonalize the model on a _semi-full_ k-mesh. It is important that the mesh not include the endpoints $k_i=1$, which correspond to the boundaries of the Brillouin zone. The Fourier transform requires a well-defined periodicity, which is disrupted by including these points. Therefore, we will use a k-mesh that spans the interior of the Brillouin zone, avoiding the boundaries. This is the default behavior of `Mesh.build_grid`. To specify that the k-mesh is periodic, with the topology of a torus, we can use the `Mesh.loop_axis` function, specifying the mesh axis and k-component that is wrapped.

# In[4]:


nks = 20, 20 # number of k points along each dimension
mesh = Mesh(dim_k=2, axis_types=['k', 'k'])
mesh.build_grid(shape=nks)
mesh.loop_axis(0, 0)
mesh.loop_axis(1, 1)
print(mesh)


# Now we pass this mesh to the `WFArray` constructor and solve the mesh.

# In[5]:


wfa = WFArray(model, mesh)
wfa.solve_mesh()


# We know that Wannierizing the full set of 4 occupied bands is obstructed by the topology of the band structure. We can try the next best thing and Wannierize a 3-dimensional subspace. To do this, we will choose a set of three trial wavefunctions centered on 3 of the low energy orbitals, where we would expect the localized Wannier functions of the trivial occupied bands to be located

# In[6]:


n_orb = model.norb # number of orbitals
n_occ = int(n_orb/2)  # number of occupied bands (assume half-filling)

low_E_sites = np.arange(0, n_orb, 2)  # low-energy sites defined to be indexed by even numbers
high_E_sites = np.arange(1, n_orb, 2)  # high-energy sites defined to be indexed by odd numbers

omit_site = 6  # omitting one of the low energy sites
sites = list(np.setdiff1d(low_E_sites, [omit_site])) 
tf_list = [ [(orb, 1)] for orb in sites]  # trial wavefunctions in form of [(orbital index, weight)]

n_tfs = len(tf_list)

print(f"Trial wavefunctions: {tf_list}")
print(f"# of Wannier functions: {n_tfs}")
print(f"# of occupied bands: {n_occ}")
print(f"Wannier fraction: {n_tfs/n_occ}")


# ## Optimal Alignment

# Next, we initialize the `Wannier` object with the `TBModel` and `WFArray` objects. We initialize the Bloch-like states with `optimal_alignment` function which aligns the trial wavefunctions with the target bands specified by `band_idxs`.

# In[7]:


WF = Wannier(model, wfa)

WF.optimal_alignment(tf_list, band_idxs=list(range(n_occ)))


# This already gives us a set of Wannier functions that are exponentially localized, showing that this is a trivial subsapce of the obstructed manifold.

# In[8]:


WF.report()


# ## Disentanglement

# We can make these states even more localized with subspace selectio via the disentanglement procedure. This picks the subspace of the 4-band manfiold that minimizes the gauge-independent spread.

# In[9]:


WF.disentangle(iter_num=2000, tol=1e-10, verbose=True, tf_speedup=True)


# ## Maximal localization
# 
# To obtain maximally localized Wannier functions, we follow this with another projection to initialize a smooth gauge, then maximal localization.
# - Note we must pass the flag `tilde=True` to indicate we are projecting the trial wavefunctions onto the tilde states and not the energy eigenstates

# In[10]:


WF.optimal_alignment(use_tilde=True)


# In[11]:


WF.max_localize(eps=1e-3, iter_num=2000, tol=1e-10, grad_min=1e-12, verbose=True)


# In[12]:


WF.report()


# Now the spreads have been minimized, and the Wannier functions are maximally localized. To help validate that the Wannier functions are indeed exponentially localized, we can plot the decay of each Wannier function's weight away from its center with `plot_decay`. This will plot the absolute value of each Wannier function as a function of distance from its center on a logarithmic scale.

# In[13]:


WF.plot_decay(0)


# In[14]:


WF.plot_density(0)


# Note that we have effectively broken the primitive translational symmetry of the underlying lattice by choosing a subset of trial wavefunctions on three out of the four low energy sites in the supercell. We can see their positions using `plot_centers`

# In[15]:


WF.plot_centers()


# ## Wannier interpolation
# 
# We can view the Wannier interpolated bands by calling `plot_interp_bands`. We specify a set of high-symmetry k-points that defines the one-dimensional path along which the bands are plotted. 

# In[16]:


k_path = [[0, 0], [2/3, 1/3], [.5, .5], [1/3, 2/3], [0, 0], [.5, .5]]
k_label = (r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $', r'$M$')


# In[17]:


n_interp = 501
interp_energies = WF.interp_bands(k_path, n_interp=n_interp, ret_eigvecs=False)


# In[20]:


fig, ax = model.plot_bands(k_path, nk=501, k_label=k_label, proj_orb_idx=high_E_sites, cmap='plasma')

(k_vec, k_dist, k_node) = model.k_path(k_path, nk=n_interp, report=False)
ax.plot(k_dist, interp_energies, ls='--', c='k', lw=2, zorder=5, alpha=0.8)

