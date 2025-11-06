#!/usr/bin/env python
# coding: utf-8

# (lattice-nb)=
# # The `Lattice` class
#
# In version 2.0.0 of `pythtb`, we introduced the `Lattice` class to encapsulate all information about the lattice geometry, including lattice vectors, orbital positions, and periodicity. This modular approach allows for a separation of concerns, where the `Lattice` class handles geometric details, while the `TBModel` class focuses on the tight-binding model itself. This design enhances code readability, maintainability, and reusability.

# In[1]:


from pythtb import Lattice
import numpy as np


# Below, we demonstrate how to use the `Lattice` class to define a lattice and then use it to create a tight-binding model.
#
# We start by defining the lattice vectors and orbital positions for a honeycomb lattice with two orbitals per unit cell. The lattice vectors are given by:
#
# $$
# \mathbf{a}_{1} = a \hat{x}, \quad \mathbf{a}_{2} = \frac{a}{2} \hat{x} + \frac{a \sqrt{3}}{2} \hat{y}
# $$
#
# where $a$ is the lattice constant, which we set to 1 for simplicity. The orbital positions are given in reduced coordinates as:
#
# $$
# \mathbf{\tau}_{1} = 0, \quad \mathbf{\tau}_{2} = \frac{1}{3} \mathbf{a}_{1} + \frac{1}{3} \mathbf{a}_{2}
# $$
#
# giving us a graphene-like structure.
#

# In[2]:


lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]


# We pass this information to the `Lattice` class to create a lattice object. Additionally, we specify which directions are periodic. In our case, both directions are periodic.
#

# In[3]:


lat = Lattice(orb_vecs=orb_vecs, lat_vecs=lat_vecs, periodic_dirs=[0, 1])


# We can see a report of the lattice object to verify its properties by printing it.

# In[4]:


print(lat)


# If we want to get the positions of the orbitals in Cartesian coordinates, we can use the `get_orb_vecs` method of the `Lattice` class and set the `cartesian` argument to `True`.

# In[5]:


lat.get_orb_vecs(cartesian=True)


# The `Lattice` class internally generates the reciprocal lattice vectors based on the provided lattice vectors and periodicity. We can access these reciprocal lattice vectors using the `get_recip_lat_vecs` method or the `recip_lat_vecs` attribute.

# In[6]:


print(lat.get_recip_lat_vecs())
print(lat.recip_lat_vecs)


# Let's verigy that the reciprocal lattice vectors satisfy the orthogonality condition with the real-space lattice vectors.
#
# $$\mathbf{a}_{i} \cdot \mathbf{b}_{j} = 2 \pi \delta_{ij}$$

# In[7]:


overlap_mat = lat.lat_vecs @ lat.recip_lat_vecs.T
print(overlap_mat / (2 * np.pi))


# If we like, we can also get the volume of the unit cell in both real and reciprocal space. These are stored as attributes of the `Lattice` class.

# In[8]:


print(lat.recip_volume, lat.cell_volume)


# In[ ]:
