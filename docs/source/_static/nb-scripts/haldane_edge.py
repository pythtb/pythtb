#!/usr/bin/env python
# coding: utf-8

# (haldane-edge-nb)=
# # Haldane model edge states
# 
# Plots the edge state-eigenfunction for a finite Haldane model that
# is periodic either in both directions or in only one direction.

# In[1]:


from pythtb import TBModel  
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# define lattice vectors
lat = [[1, 0], [1/2, np.sqrt(3)/2]]
# define coordinates of orbitals
orb = [[1/3, 1/3], [2/3, 2/3]]

# make two dimensional tight-binding Haldane model
my_model = TBModel(2, 2, lat, orb)

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


# In[3]:


# cutout finite model first along direction x with no PBC
tmp_model = my_model.cut_piece(10, 0, glue_edgs=False)
# cutout also along y direction with no PBC
fin_model = tmp_model.cut_piece(10, 1, glue_edgs=False)

# cutout finite model first along direction x with PBC
tmp_model_half = my_model.cut_piece(10, 0, glue_edgs=True)
# cutout also along y direction with no PBC
fin_model_half = tmp_model_half.cut_piece(10, 1, glue_edgs=False)

# solve finite models
(evals, evecs) = fin_model.solve_ham(return_eigvecs=True)
(evals_half, evecs_half) = fin_model_half.solve_ham(return_eigvecs=True)


# In[4]:


# pick index of state in the middle of the gap
ed = fin_model.norb // 2

# draw one of the edge states in both cases
(fig, ax) = fin_model.visualize(proj_plane=[0, 1], eig_dr=evecs[ed, :], draw_hoppings=False)

ax.set_title("Edge state for finite model without periodic direction")
ax.set_xlabel("x coordinate")
ax.set_ylabel("y coordinate")
plt.show()


# In[5]:


(fig, ax) = fin_model_half.visualize(
    proj_plane=[0, 1], eig_dr=evecs_half[ed, :], draw_hoppings=False
)
ax.set_title("Edge state for finite model periodic in one direction")
ax.set_xlabel("x coordinate")
ax.set_ylabel("y coordinate")
fig.tight_layout()
plt.show()

