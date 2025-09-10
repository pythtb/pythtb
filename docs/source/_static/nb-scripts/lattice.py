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
# $$
# \mathbf{a}_{1} = a \hat{x}, \quad \mathbf{a}_{2} = \frac{a}{2} \hat{x} + \frac{a \sqrt{3}}{2} \hat{y}
# $$
# where $a$ is the lattice constant, which we set to 1 for simplicity. The orbital positions are given in reduced coordinates as:
# $$
# \mathbf{\tau}_{1} = 0, \quad \mathbf{\tau}_{2} = \frac{1}{3} \mathbf{a}_{1} + \frac{1}{3} \mathbf{a}_{2}
# $$ 
# giving us a graphene-like structure.
# 

# In[2]:


lat_vecs = [[1, 0], [1/2, np.sqrt(3)/2]]
orb_vecs = [[1/3, 1/3], [2/3, 2/3]]


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
print(overlap_mat/(2*np.pi))


# If we like, we can also get the volume of the unit cell in both real and reciprocal space. These are stored as attributes of the `Lattice` class.

# In[8]:


print(lat.recip_volume, lat.cell_volume)


# In[9]:


from pythtb.lattice import Lattice, SymmetryOperation
import numpy as np


# In[10]:


avec = [[1,0,0],[0,1,0],[0,0,1]]
orbs = [[0,0,0],[1/3,2/3,0]]
per  = [0,1]  # 2D periodic slab with a nonperiodic z


# In[11]:


# C3z rotation in reduced coords around z (acts only on x,y block)
theta = 2*np.pi/3
R = np.array(
    [[np.cos(theta), -np.sin(theta), 0],
     [np.sin(theta),  np.cos(theta), 0],
     [0,              0,             1]], 
    float)


# In[12]:


# In reduced lattice basis a proper C3 on a hex lattice is integer-like on the periodic subspace.
# For a generic basis, you’ll typically use the integer form (e.g., [[0,-1],[1,-1]]) in the periodic block.
lat = Lattice(avec, orbs, periodic_dirs=per)
lat.add_symmetry_operation(R, t=[0,0,0], label='C3z')


# In[13]:


print(lat)


# In[14]:


ok = lat.check_symmetry()
print(ok)


# In[15]:


# # k-space helpers
# Gamma = np.array([0.0,0.0,0.0])
# Kstar = lat.star_of_k(np.array([1/3,1/3,0.0]))
# keepers = lat.little_group(Gamma)


# In[16]:


# 2D periodic (x,y), z nonperiodic example
lat = Lattice(
    lat_vecs=np.array([[1,0,0],[0,1,0],[0,0,1]], float),
    orb_vecs=np.array([[0,0,0],[1/3,2/3,0]], float),
    periodic_dirs=[0,1]
)

# Proper reduced-basis C3 around a lattice site
lat.add_symmetry_operation(
    R=[[0,-1,0],[1,-1,0],[0,0,1]],
    t=[0,0,0],
    label="C3z"
)

print(lat.check_symmetry())
# Expect {'lattice_ok': True, 'orbitals_ok': True, ...}


# In[17]:


from pythtb.models import fu_kane_mele

model = fu_kane_mele(1, 1, 1, 1)


# In[19]:


model.lattice.visualize_3d(n_cells=3)

