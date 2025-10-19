#!/usr/bin/env python
# coding: utf-8

# (boron-nitride-nb)=
# # Boron-nitride ribbon
# 
# We compare two descriptions of a boron-nitride nanoribbon: one in the original oblique unit cell, and another after reorienting the non-periodic lattice vector so it is orthogonal to the periodic direction. Both models share the same band structure, yet only the orthogonalized cell yields a Berry phase consistent with the ribbon’s physical polarization.
# 
# :::{admonition} What you will learn
# :class: tip
# - Build a 2D honeycomb model and cut out a ribbon with `TBModel.cut_piece`.
# - Re-define the non-periodic lattice vector with `TBModel.change_nonperiodic_vector` to clean up geometric invariants.
# - Use `Mesh` and `WFArray` to solve a 1D Brillouin-zone problem and compute Berry phases.
# - Diagnose how cell geometry influences Wilson loops and polarization.
# :::

# In[1]:


from pythtb import TBModel, WFArray, Mesh, Lattice
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# define lattice vectors
lat_vecs = [[1, 0], [1/2, np.sqrt(3)/2]]
# define coordinates of orbitals
orb_vecs = [[1/3, 1/3], [2/3, 2/3]]

lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

# make two dimensional tight-binding boron nitride model
my_model = TBModel(lat)

# set periodic model
delta = 0.4
t = -1.0
my_model.set_onsite([-delta, delta])
my_model.set_hop(t, 0, 1, [0, 0])
my_model.set_hop(t, 1, 0, [1, 0])
my_model.set_hop(t, 1, 0, [0, 1])
print(my_model)
my_model.visualize()


# ## `TBModel.cut_piece`
# 
# We first carve a ribbon by repeating the boron-nitride sheet three times along the second lattice vector and opening the boundary in that direction. `cut_piece` replicates the hoppings and onsite terms while removing the corresponding crystal-momentum axis.

# In[3]:


model_orig = my_model.cut_piece(3, 1, glue_edges=False)
print(model_orig)
model_orig.visualize()


# ## `TBModel.change_nonperiodic_vector`
# 
# Next we redefine the non-periodic lattice vector so it points exactly normal to the periodic direction. This leaves the tight-binding spectrum unchanged but simplifies the geometry that underlies Berry-phase calculations. 
# 
# :::note
# The default behavior of `change_nonperiodic_vector` is to orthogonalize the non-periodic vector to the periodic one. One can also specify a custom vector.
# :::

# In[4]:


model_perp = model_orig.copy()
model_perp.change_nonperiodic_vector(1, to_home=True)
model_perp.report(short=True)
model_perp.visualize()


# ## Bands and Berry phase
# 
# We solve both ribbon models on the same 1D $k$-mesh using `WFArray.solve_model`, verify that their band structures coincide, and then compute the Berry phase along the periodic direction using `WFArray.berry_phase`. The orthogonalized cell produces the expected zero Berry phase (by mirror symmetry), while the oblique cell does not.

# In[5]:


fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

def run_model(model, panel, title):
    k_nodes = [[-0.5], [0.5]]

    model.plot_bands(k_nodes=k_nodes, nk=100, fig=fig, ax=ax[panel], lw=1)
    ax[panel].set_xticklabels([-0.5, 0.5])

    mesh = Mesh(dim_k=model.dim_k, axis_types=["k"])
    mesh.build_grid(shape=(40,), gamma_centered=True)
    wf = WFArray(model.lattice, mesh)
    wf.solve_model(model)

    n_occ = model.nstate // 2
    berry_phase = wf.berry_phase(mu=0, state_idx=range(n_occ))
    ax[panel].set_title(fr"{title}: $\phi=${berry_phase: .3f}")

    print(f"Berry Phase {title} = {berry_phase:.7f}")

run_model(model_orig, 0, "Original")
run_model(model_perp, 1, "Shifted")

ax[1].set_ylabel(None)
fig.tight_layout()
plt.show()


# ## Note about mirror symmetry
# 
# Let $x$ denote the ribbon direction (periodic) and $y$ the transverse direction (finite). The ribbon has an $M_x$ mirror symmetry, so the polarization—and hence the Berry phase—must be either $0$ or $\pi$.  
# 
# In the original oblique cell the reciprocal loop vector $\mathbf{b}_0$ used in the Wilson loop has both $x$ and $y$ components. The Berry phase therefore mixes shifts of Wannier centers along $y$ with the desired polarization along $x$, yielding a spurious result.  
# 
# After `change_nonperiodic_vector`, $\mathbf{b}_0$ points purely along $x$. The Wilson loop samples overlaps with phase factors depending only on $x$, so the Berry phase reflects the true ribbon polarization and vanishes as dictated by symmetry. The band structure remains unchanged because it depends only on the periodic direction.

# ## Next steps
# 
# :::{admonition} Next steps
# :class: seealso
# - Tilt the non-periodic lattice vector by a controlled angle and plot how the Berry phase deviates from the mirror-symmetric value.
# - Compute edge Wannier centers with `WFArray.position_expectation` to visualise how the polarization migrates when the cell is oblique.
# :::
# 
