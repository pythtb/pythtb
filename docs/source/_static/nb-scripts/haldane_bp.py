#!/usr/bin/env python
# coding: utf-8

# (haldane-bp-nb)=
# # Haldane model: Berry phases and curvatures

# In[1]:


from pythtb import TBModel, Lattice, WFArray, Mesh
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# define lattice vectors
lat_vecs = [[1, 0], [1/2, np.sqrt(3)/2]]
# define coordinates of orbitals
orb_vecs = [[1/3, 1/3], [2/3, 2/3]]
lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

# make two dimensional tight-binding Haldane model
my_model = TBModel(lat)

# set model parameters
delta = 0
t = -1
t2 = 0.15 * np.exp(1j*np.pi/2)
t2c = t2.conjugate()

# set on-site energies
my_model.set_onsite([-delta, delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t, 0, 1, [0, 0])
my_model.set_hop(t, 1, 0, [1, 0])
my_model.set_hop(t, 1, 0, [0, 1])
# add second neighbour complex hoppings
my_model.set_hop(t2, 0, 0, [1, 0])
my_model.set_hop(t2, 1, 1, [1, -1])
my_model.set_hop(t2, 1, 1, [0, 1])
my_model.set_hop(t2c, 1, 1, [1, 0])
my_model.set_hop(t2c, 0, 0, [1, -1])
my_model.set_hop(t2c, 0, 0, [0, 1])

print(my_model)
my_model.visualize()


# ## Inspect the band structure
# 
# A high-symmetry path through the hexagonal Brillouin zone highlights the gap opened by the complex second-neighbour hopping. We colour the bands by projection onto one sublattice to highlight the fact that a band-inversion occured at the $K^\prime$ point upon the gap closing and re-opening.

# In[3]:


k_nodes = [[0, 0], [2/3, 1/3], [.5, .5], [1/3, 2/3], [0, 0], [.5, .5]]
k_labels = (r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $', r'$M$')

my_model.plot_bands(k_nodes, k_node_labels=k_labels, nk=501, scat_size=2, proj_orb_idx=[1], cmap='plasma')


# ## Brillouin-zone mesh
# 
# To compute curvature we sample the full two-dimensional Brillouin zone. `Mesh(dim_k=2, axis_types=['k','k']).build_grid()` builds a Monkhorst–Pack grid; this gives a uniform sampling without the endpoints.

# In[4]:


mesh = Mesh(dim_k=2, axis_types=['k', 'k'])
mesh.build_grid(shape=(31, 31), gamma_centered=True)
print(mesh)


# ## Using `WFArray`
# 
# Generate object of type `WFArray` that will be used for Berry phase and curvature calculations

# In[5]:


wfa = WFArray(my_model.lattice, mesh)
wfa.solve_model(my_model)


# Calculate Berry phases around the BZ in the $k_x$ direction (which can be interpreted as the 1D hybrid Wannier center in the $x$ direction) and plot results as a function of $k_y$.

# In[6]:


# Berry phases along k_x for lower band
phi_0 = wfa.berry_phase(0, [0], contin=True)
# Berry phases along k_x for upper band
phi_1 = wfa.berry_phase(0, [1], contin=True)
# Berry phases along k_x for both bands
phi_both = wfa.berry_phase(0, [0, 1], contin=True)


# These results indicate that the two bands have equal and opposite Chern numbers.

# In[7]:


# plot Berry phases
fig, ax = plt.subplots()
ky = np.linspace(0, 1, len(phi_1))
ax.plot(ky, phi_0, "ro-", label="Lower band")
ax.plot(ky, phi_1, "go-", label="Upper band")
ax.plot(ky, phi_both, "bo-", label="Both bands")

ax.legend()
ax.set_xlabel(r"$k_y$")
ax.set_ylabel(r"Berry phase along $k_x$")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-7.0, 7.0)
ax.yaxis.set_ticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
ax.set_yticklabels((r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"))


# Verify with calculation of Chern numbers

# In[8]:


chern0 = wfa.chern_num(state_idx=[0], plane=(0,1))
chern1 = wfa.chern_num(state_idx=[1], plane=(0,1))

print("Chern number for lower band = ", chern0)
print("Chern number for upper band = ", chern1)


# ## Berry flux tiles
# 
# `WFArray.berry_flux(state_idx=[0], plane=(0, 1))` returns the discretized Berry flux through each plaquette for the chosen band (here the lowest). This is the gauge-invariant ingredient that sums to the band Chern number.

# In[9]:


bflux = wfa.berry_flux(state_idx=[0], plane=(0,1))


# ## Visualize the curvature
# 
# We map the mesh points into Cartesian momentum coordinates using the reciprocal lattice vectors, then plot the Berry flux density with `pcolormesh`. The peak at the $K^\prime$ point signals the topological character of the band.

# In[10]:


mesh_cart = mesh.grid @ my_model.recip_lat_vecs
KX, KY= mesh_cart[..., 0], mesh_cart[..., 1]

im = plt.pcolormesh(KX, KY, bflux, cmap='plasma', shading='gouraud')
plt.colorbar(label=r'$\Omega(\mathbf{k})$')

