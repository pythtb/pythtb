#!/usr/bin/env python
# coding: utf-8

# (ssh-edge-nb)=
# # SSH model topological edge states
# 
# This example demonstrates the emergence of topological edge states in a finite Su-Schrieffer-Heeger (SSH) chain as a function of the hopping parameters.

# In[1]:


from pythtb import TBModel, WFArray, Mesh
from pythtb.utils import pauli_decompose
import matplotlib.pyplot as plt
import numpy as np


# Define the model generator
# 
# :::{hint}
# You can also get the SSH model using the `pythtb.models` library
# :::

# In[2]:


def ssh(v, w):
    lat = [[1]]
    orb = [[-1/4], [1/4]]
    my_model = TBModel(1, 1, lat, orb)

    my_model.set_hop(v, 0, 1, [0])
    my_model.set_hop(w, 1, 0, [1])

    return my_model


# In[3]:


model = ssh(1, 1)
model.visualize()


# ## Finite Model
# 
# Get the eigenvalues and eigenvectors for each intracell hopping $w$

# In[4]:


nlam = 100
w_values = np.linspace(0, 1, nlam)
v = 0.5
evals_w = []
evecs_w = []
n_cells = 30
for w in w_values:
    finite_model = ssh(v, w).cut_piece(n_cells, 0)
    evals, evecs = finite_model.solve_ham(return_eigvecs=True)
    evals_w.append(evals)
    evecs_w.append(evecs)


# Plotting the energies as a function of $w$ shows that two zero-energy levels emerge at the topological phase transition where $|w| > |v|$.
# These zero energy states are edge modes, as visible in the edge state density plot.

# In[5]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the eigenvalues
ax1.plot(w_values, evals_w, c='r')

# Mark the v hopping on the plot
ax1.axvline(x=v, color='b', linestyle='--', lw=1, label=r'$v$ hopping')

ax1.set_xlabel(r"$w$")
ax1.set_ylabel(r"$E$")
ax1.set_xlim(0, 1.0)
ax1.set_ylim(-2, 2)
ax1.set_title(f"Finite SSH Chain with {n_cells} unit cells")
ax1.legend()

# Plot edge state density at last w=1.0
band_idx = n_cells
density = np.abs(evecs_w[-1][band_idx, :])**2
position = np.arange(len(density)) / 2
ax2.plot(position, density)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(rf"$\psi_{{{band_idx}}}(x)$")
ax2.set_title(r"Edge state density at $w=1.0$")


# ## Bulk polarization and Berry phase

# In[10]:


model = ssh(v, 0)
nk = 100

mesh = Mesh(model, axis_types=['k'])
mesh.build_full_grid(shape=(nk,))

wfa = WFArray(mesh)
wfa.solve_mesh()
P1 = wfa.berry_phase([1], 0) / (2 * np.pi)
P1


# In[12]:


model = ssh(v, 1)
nk = 100

mesh = Mesh(model, axis_types=['k'])
mesh.build_full_grid(shape=(nk,))

wfa = WFArray(mesh)
wfa.solve_mesh()
P2 = wfa.berry_phase([0], 0) / (2 * np.pi)
P2

