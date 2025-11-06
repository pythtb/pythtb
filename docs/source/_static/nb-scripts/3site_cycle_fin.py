#!/usr/bin/env python
# coding: utf-8

# (3site-cycle-fin-nb)=
# # Three-site Thouless pump
#
# We revisit the three-orbital 1D model whose onsite energies slide around the unit cell as the adiabatic parameter $\lambda$ sweeps from 0 to 1. In the periodic geometry this produces a quantized charge pump; here we cut a finite chain to reveal the edge modes that accompany the bulk Chern numbers.
#

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pythtb import Lattice, Mesh, TBModel, WFArray


# ## Model builder
#
# `set_model` constructs the periodic three-site Hamiltonian for a given hopping `t`, onsite amplitude `delta`, and pump parameter `lmbda`. The onsite terms follow a cosine profile delayed by $2\pi/3$ on each orbital so the deepest well moves from site 0 → 1 → 2 as $\lambda$ advances.

# In[2]:


def set_model(t: float, delta: float, lmbda: float) -> TBModel:
    """Periodic three-site model at parameter lmbda."""
    lattice = Lattice(
        lat_vecs=[[1.0]], orb_vecs=[[0.0], [1 / 3], [2 / 3]], periodic_dirs=[0]
    )
    model = TBModel(lattice=lattice)

    # nearest-neighbour hoppings (last hop wraps to the next cell)
    model.set_hop(t, 0, 1, [0])
    model.set_hop(t, 1, 2, [0])
    model.set_hop(t, 2, 0, [1])

    onsite = [delta * -np.cos(2 * np.pi * (lmbda - idx / 3)) for idx in range(3)]
    model.set_onsite(onsite)
    return model


# ## Mesh and wavefunctions
#
# We sample the two-dimensional parameter space $(k,\lambda)$ with a `Mesh`. Axis names must match the keyword arguments in `set_model`; otherwise the solver would not vary the correct parameter. In our case we should name the parametric axis `"lmbda"`.
#
# :::{tip}
# `dim_k` and `dim_lambda` do not need to match the number of k- and $\lambda$- axes. For example, the mesh may define a path through a higher-dimensional Brillouin zone.
# :::

# In[3]:


mesh = Mesh(
    dim_k=1,
    dim_lambda=1,
    axis_types=["k", "l"],  # first axis: crystal momentum; second: adiabatic parameter
    axis_names=["kx", "lmbda"],
)


# Build a uniform $(k,\lambda)$ grid and enforce periodic boundary conditions along the $\lambda$ axis so that $\lambda=0$ and $\lambda=1$ are identified.
#
# A few additional parameters are available to control the range of the parameter axis and whether to loop it.
# In `mesh.build_grid`, we specify the start and stop values for the lambda parameter axis. By default, the endpoint is included along the lambda axes, while it is not included along the k axes.
#
# :::{versionchanged} 2.0.0
# In previous versions, `mesh.loop_axis(1,1)` would have been specified in `wf_array` by `impose_loop(1,1)`.
# :::

# In[4]:


mesh.build_grid(shape=(31, 21), gamma_centered=True, lambda_start=0.0, lambda_stop=1.0)
mesh.loop_axis(axis_idx=1, component_idx=1)  # form the lambda axis into a loop
mesh.close_axis(
    axis_idx=1, component_idx=1
)  # indicate that the end of the loop completes the cycle (endpoint included)
print(mesh)


# Initialise a `WFArray` with the mesh, seed it with a reference model (only the orbital metadata is used), and solve the Hamiltonian for every $(k,\lambda)$ point while keeping `t` and `delta` fixed.

# In[6]:


reference_model = set_model(t=0.0, delta=0.0, lmbda=0.0)
wfa = WFArray(reference_model, mesh)


# In[7]:


fixed_params = {"t": -1.3, "delta": 2.0}
wfa.solve(model_func=set_model, fixed_params=fixed_params)


# ## Berry flux and Chern numbers
#
# The charge pumped per cycle equals the Chern number computed on the $(k,\lambda)$ torus. We evaluate the Berry curvature with `WFArray.chern_num` for individual bands and for cumulative fillings.

# In[8]:


fillings = {
    "band 0": [0],
    "bands 0–1": [0, 1],
    "bands 0–2": [0, 1, 2],
}

cherns = {
    label: wfa.chern_num(state_idx=indices, plane=(0, 1))
    for label, indices in fillings.items()
}
band_cherns = {band: wfa.chern_num(state_idx=[band], plane=(0, 1)) for band in range(3)}

print("Chern numbers by filling:")
for label, value in cherns.items():
    print(f"  {label:<10} = {value:+5.2f}")

print("\nIndividual band Chern numbers:")
for band, value in band_cherns.items():
    print(f"  band {band}      = {value:+5.2f}")


# ## Finite chain setup

# To expose the edge physics behind the pump, we cut a chain of `num_cells` unit cells from the periodic model. `cut_piece` removes the $k$ degree of freedom while keeping the hopping pattern intact.

# In[12]:


num_cells = 10
num_orb = 3 * num_cells


def finite_model_builder(t: float, delta: float, lmbda: float) -> TBModel:
    periodic = set_model(t, delta, lmbda)
    return periodic.cut_piece(num_cells, 0)


# The finite system has no crystal momentum axis, so the mesh tracks only the adiabatic parameter. We sample $\lambda$ densely enough to resolve the edge-state crossings.

# In[13]:


finite_mesh = Mesh(dim_k=0, dim_lambda=1, axis_types=["l"], axis_names=["lmbda"])
finite_mesh.build_grid(shape=(241,), lambda_start=0.0, lambda_stop=1.0)
finite_mesh.loop_axis(0, 0)  # lambda axis and component now indexed by 0
finite_mesh.close_axis(0, 0)
print(finite_mesh)


# As before, create a `WFArray` tied to this mesh and populate it by sweeping $\lambda$ across the cycle.

# In[16]:


finite_reference = finite_model_builder(t=0.0, delta=0.0, lmbda=0.0)
finite_wfa = WFArray(finite_reference, finite_mesh)
finite_wfa.solve(model_func=finite_model_builder, fixed_params=fixed_params)


# ## Position expectation values
#
# `WFArray.position_expectation(dir=0)` returns the center of each eigenstate in units of the lattice spacing. Bulk states span the chain and sit near the midpoint, whereas edge states hug either boundary.

# In[17]:


x_expectation = finite_wfa.position_expectation(dir=0)


# ### Spectrum versus $\lambda$
#
# Eigenenergies of the finite chain traced over the adiabatic cycle. Point colour encodes the position expectation value $\langle x \rangle$: bulk states (green at the chain centre) stay in the gap, while edge-localised states (dark/light extremes) thread the gap and connect the valence and conduction manifolds. This matches the non-zero Chern number found for the periodic system.

# In[31]:


lambda_points = finite_mesh.get_param_points()
vmin, vmax = x_expectation.min(), x_expectation.max()
cmap = matplotlib.colormaps.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(8, 5))

for orb in range(num_orb):
    sc = ax.scatter(
        lambda_points,
        finite_wfa.energies[:, orb],
        c=x_expectation[:, orb],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=12,
        edgecolors="none",
        alpha=0.85,
    )

cbar = fig.colorbar(sc, ax=ax, pad=0.02, label=r"$\langle x \rangle$ (cells)")

ax.text(0.18, -1.7, rf"$\mathcal{{C}}_0 = {cherns['band 0']:+1.0f}$")
ax.text(0.46, 1.6, rf"$\mathcal{{C}}_{{(0,1)}} = {cherns['bands 0–1']:+1.0f}$")

ax.set_title("Finite-chain spectrum of the three-site pump")
ax.set_xlabel(r"Adiabatic parameter $\lambda$")
ax.set_ylabel("Energy")
ax.set_xlim(0.0, 1.0)

plt.show()
