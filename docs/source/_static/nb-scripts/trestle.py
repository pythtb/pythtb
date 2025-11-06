#!/usr/bin/env python
# coding: utf-8

# (trestle-nb)=
# # Ribbon geometry

# In[2]:


from pythtb.tb_model import TBModel, Lattice
import matplotlib.pyplot as plt


# We start with a simple model that has one-dimensional k-space and two-dimensional r-space. The model also includes complex hoppings between orbitals.

# In[3]:


# define lattice vectors
lat_vecs = [[2, 0], [0, 1]]
# define coordinates of orbitals
orb_vecs = [[0, 0], [1 / 2, 1]]

lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0])

# make one dimensional tight-binding model of a trestle-like structure
my_model = TBModel(lat)

# set model parameters
t_first = 0.8 + 0.6j
t_second = 2

# leave on-site energies to default zero values
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t_second, 0, 0, [1, 0])
my_model.set_hop(t_second, 1, 1, [1, 0])
my_model.set_hop(t_first, 0, 1, [0, 0])
my_model.set_hop(t_first, 1, 0, [1, 0])

print(my_model)
my_model.visualize()


# ## Band structure calculation
#
# We will calculate the band structure of the model by solving the tight-binding Hamiltonian on a grid of k-points in the Brillouin zone. To do so, we will call the `k_path` method with `"fullc"` to generate a path centered at the Gamma point.

# In[4]:


# generate list of k-points following some high-symmetry line in
(k_vec, k_dist, k_node) = my_model.k_path("fullc", 100)
k_label = [r"$-\pi$", r"$0$", r"$\pi$"]


# Now solve for eigenenergies of Hamiltonian on the set of k-points from above.

# In[5]:


evals = my_model.solve_ham(k_vec)


# Plotting bandstructure...

# In[6]:


# First make a figure object
fig, ax = plt.subplots()
# specify horizontal axis details
ax.set_xlim(k_node[0], k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(k_label)
ax.axvline(x=k_node[1], linewidth=0.5, color="k")

# plot bands together
ax.plot(k_dist, evals)

# set titles
ax.set_title("Trestle band structure")
ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy")
