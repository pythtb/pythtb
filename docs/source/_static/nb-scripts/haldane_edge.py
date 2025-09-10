#!/usr/bin/env python
# coding: utf-8

# (haldane-edge-nb)=
# # Haldane model edge states
# 
# Plots the edge state-eigenfunction for a finite Haldane model that
# is periodic either in both directions or in only one direction.

# In[1]:


from pythtb import TBModel, Lattice 
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
delta = 0.0
t = -1.0
t2 = 0.15 * np.exp(1j*np.pi/2)
t2c = t2.conjugate()

my_model.set_onsite([-delta, delta])
my_model.set_hop(t, 0, 1, [0, 0])
my_model.set_hop(t, 1, 0, [1, 0])
my_model.set_hop(t, 1, 0, [0, 1])
my_model.set_hop(t2, 0, 0, [1, 0])
my_model.set_hop(t2, 1, 1, [1, -1])
my_model.set_hop(t2, 1, 1, [0, 1])
my_model.set_hop(t2c, 1, 1, [1, 0])
my_model.set_hop(t2c, 0, 0, [1, -1])
my_model.set_hop(t2c, 0, 0, [0, 1])

print(my_model)


# In[ ]:


# cut finite models with slices in both directions
# first direction open, second direction glued
fin_model = my_model.make_finite(dirs=[0,1], num_cells=[10, 10], glue_edges=[False, False])
fin_model_half = my_model.make_finite(dirs=[0,1], num_cells=[10, 10], glue_edges=[True, False])


# In[9]:


# solve finite models
(evals, evecs) = fin_model.solve_ham(return_eigvecs=True)
(evals_half, evecs_half) = fin_model_half.solve_ham(return_eigvecs=True)


# In[16]:


# pick index of state in the middle of the gap
ed = fin_model.norb // 2

# draw one of the edge states in both cases
(fig, ax) = fin_model.visualize(proj_plane=[0, 1], eig_dr=evecs[ed, :], draw_hoppings=False)

ax.set_title("Edge state for finite model without periodic direction")
ax.set_xlabel("x coordinate")
ax.set_ylabel("y coordinate")
plt.show()


# In[17]:


(fig, ax) = fin_model_half.visualize(
    proj_plane=[0, 1], eig_dr=evecs_half[ed, :], draw_hoppings=False
)
ax.set_title("Edge state for finite model periodic in one direction")
ax.set_xlabel("x coordinate")
ax.set_ylabel("y coordinate")
fig.tight_layout()
plt.show()

