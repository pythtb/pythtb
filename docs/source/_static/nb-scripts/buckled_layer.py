#!/usr/bin/env python
# coding: utf-8

# (buckled-layer-nb)=
# # Buckled layer slab
#
# We examine a buckled layer slab model. Orbitals live in 3D real space, but the system is periodic only in the $x$–$y$ plane so the crystal momentum remains two-dimensional.
#
# :::{admonition} What you will learn
# :class: tip
# - Build a slab-style `TBModel` whose real space is lower dimensional than reciprocal space.
# - Define a high-symmetry path in the 2D Brillouin zone for plotting.
# - Compare the convenience of `TBModel.plot_bands` with manual diagonalization using `TBModel.solve_ham`.
# :::

# In[1]:


from pythtb import TBModel, Lattice
import matplotlib.pyplot as plt


# In[2]:


import plotly.io as pio

# switch to an HTML‐based renderer that MyST-NB understands
pio.renderers.default = "notebook"  # or "notebook", "browser", etc.


# In[3]:


# define 3D real-space lattice vectors
lat_vecs = [[1, 0, 0], [0, 1.25, 0], [0, 0, 3]]
# define coordinates of orbitals in reduced units
orb_vecs = [[0, 0, -0.15], [0.5, 0.5, 0.15]]

# only first two lattice vectors repeat, so k-space is 2D
lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

my_model = TBModel(lat)

delta = 1.1
t = 0.6

# set on-site energies
my_model.set_onsite([-delta, delta])
# set hoppings (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t, 1, 0, [0, 0, 0])
my_model.set_hop(t, 1, 0, [1, 0, 0])
my_model.set_hop(t, 1, 0, [0, 1, 0])
my_model.set_hop(t, 1, 0, [1, 1, 0])

print(my_model)
my_model.visualize_3d()


# ## Band structure calculation

# Now we specify the k-space path for the band structure calculation by listing a set of nodes. The path will consist of straight line segments connecting these nodes.

# ## High-symmetry path
#
# We prescribe a sequence of nodes in the 2D Brillouin zone and let `TBModel.plot_bands` interpolate straight segments between them. This path will feed both plotting workflows below.

# In[4]:


path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
# specify labels for these nodal points
label = (r"$\Gamma $", r"$X$", r"$M$", r"$\Gamma $")


# ## Quick look with `TBModel.plot_bands`
#
# `plot_bands` handles path construction, diagonalization, and plotting in one call. It is great for rapid inspection of gaps, degeneracies, and orbital projections without managing k-point arrays or diagonalization manually.

# In[5]:


my_model.plot_bands(k_nodes=path, k_node_labels=label, nk=100)


# ## Manual diagonalisation via `TBModel.solve_ham`
#
# For finer control we can manually compute the interpolated k-points along path using `TBModel.k_path`. This will return the k-vectors along the path, the cumulative k-point distances from the first node normalized to a maximum of 1, and the cumulative distances of the high-symmetry k-points. We can then use the k-vectors to diagonalize the model.

# In[6]:


(k_vec, k_dist, k_node) = my_model.k_path(path, 81)


# In[7]:


evals = my_model.solve_ham(k_vec)


# In[8]:


fig, ax = plt.subplots()
ax.set_title("Bandstructure for buckled rectangular layer")
ax.set_ylabel("Band energy")

# specify horizontal axis details
ax.set_xlim(k_node[0], k_node[-1])
# put tickmarks and labels at node positions
ax.set_xticks(k_node)
ax.set_xticklabels(label)
# add vertical lines at node positions
for n in range(len(k_node)):
    ax.axvline(x=k_node[n], linewidth=0.5, color="k")

ax.plot(k_dist, evals)


# ## Next steps

# :::{admonition} Next steps
# :class: seealso
# - Slice the slab with `cut_piece` to form a ribbon and study edge modes along the remaining periodic direction.
# :::
#
