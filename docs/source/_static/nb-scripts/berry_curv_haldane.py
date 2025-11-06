#!/usr/bin/env python
# coding: utf-8

# (berry-curv-nb)=
# # Berry curvature of the Haldane model
#
# We evaluate the Berry curvature of the Haldane model on a two-dimensional Brillouin-zone mesh, visualize its hot spots near the Dirac points, and integrate it to recover the Chern number of the valence band.
#
# :::{admonition} What you will learn to do
# :class: tip
# - Instantiate the Haldane tight-binding model using `pythtb.models`.
# - Sample a 2D Brillouin zone with `Mesh` and populate a `WFArray`.
# - Compute Berry flux tiles with `WFArray.berry_flux` and integrate them to obtain a Chern number.
# - Plot the curvature in Cartesian momentum coordinates to locate topological features.
# :::

# In[ ]:


from pythtb import WFArray, Mesh
from pythtb.models.haldane import haldane

import matplotlib.pyplot as plt


# ## Initialise the model
#
# We import the Haldane model factory and choose staggered mass `delta`, nearest-neighbour hopping `t`, and complex next-nearest-neighbour hopping `t2`. With these parameters the lower band is topological ($\mathcal{C}=+1$).

# In[12]:


# tight-binding parameters
delta = 1
t = 1
t2 = -0.3

model = haldane(delta, t, t2)
print(model)


# ## Inspect the band structure
#
# A high-symmetry path through the hexagonal Brillouin zone highlights the gap opened by the complex second-neighbour hopping. We colour the bands by projection onto one sublattice to highlight the fact that a band-inversion occured at the $K^\prime$ point upon the gap closing and re-opening.

# In[13]:


k_nodes = [[0, 0], [2 / 3, 1 / 3], [0.5, 0.5], [1 / 3, 2 / 3], [0, 0], [0.5, 0.5]]
k_labels = (r"$\Gamma $", r"$K$", r"$M$", r"$K^\prime$", r"$\Gamma $", r"$M$")

model.plot_bands(
    k_nodes,
    k_node_labels=k_labels,
    nk=501,
    scat_size=2,
    proj_orb_idx=[1],
    cmap="plasma",
)


# ## Brillouin-zone mesh
#
# To compute curvature we sample the full two-dimensional Brillouin zone. `Mesh(dim_k=2, axis_types=['k','k']).build_grid()` builds a Monkhorst–Pack grid; this gives a uniform sampling without the endpoints.

# In[4]:


mesh = Mesh(dim_k=2, axis_types=["k", "k"])
mesh.build_grid(shape=(20, 20))
print(mesh)


# ## Solve the Hamiltonian on the mesh
#
# `WFArray` stores eigenvectors (and energies) at every mesh point. Passing the `TBModel` directly to `solve_model` diagonalizes the Hamiltonian on the predefined grid and caches the overlaps required for geometric quantities.

# In[5]:


wfa = WFArray(model.lattice, mesh)
wfa.solve_model(model)


# ## Berry flux tiles
#
# `WFArray.berry_flux(state_idx=[0], plane=(0, 1))` returns the discretized Berry flux through each plaquette for the chosen band (here the lowest). This is the gauge-invariant ingredient that sums to the band Chern number.

# In[ ]:


bflux = wfa.berry_flux(state_idx=[0], plane=(0, 1))


# ## Visualize the curvature
#
# We map the mesh points into Cartesian momentum coordinates using the reciprocal lattice vectors, then plot the Berry flux density with `pcolormesh`. The peak at the $K^\prime$ point signals the topological character of the band.

# In[14]:


mesh_cart = mesh.grid @ model.recip_lat_vecs
KX, KY = mesh_cart[..., 0], mesh_cart[..., 1]

im = plt.pcolormesh(KX, KY, bflux, cmap="plasma", shading="gouraud")
plt.colorbar(label=r"$\Omega(\mathbf{k})$")


# ## Integrate to obtain the Chern number
#
# Summing the Berry flux over the torus yields the Chern number. Numerical noise should be negligible; the result should round to $+1$ for the parameters chosen above.

# In[15]:


chern = wfa.chern_num(state_idx=[0], plane=(0, 1))
print(f"Chern # occupied: {chern.real: .1f}")


# ## Next steps
#
# :::{admonition} Next steps
# :class: seealso
# - Vary `delta` and `t2` across the phase diagram and watch the curvature redistribute as the Chern number changes.
# - Increase the mesh density to study convergence of the discrete Berry flux.
# - Compute curvature for both bands (`state_idx=[0,1]`) to confirm the total Chern number vanishes.
# :::
#
