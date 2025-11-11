#!/usr/bin/env python
from __future__ import print_function # python3 style print

# Construct and compute Berry phases of hybrid Wannier functions
# for a simple slab model

from pythtb import * # import TB model class

import matplotlib.pyplot as plt

# set up model on bcc motif (CsCl structure)
# nearest-neighbor hopping only, but of two different strengths
def set_model(delta,ta,tb):
  lat=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
  orb=[[0.0,0.0,0.0],[0.5,0.5,0.5]]
  model=tb_model(3,3,lat,orb)
  model.set_onsite([-delta,delta])
  for lvec in ([-1,0,0],[0,0,-1],[-1,-1,0],[0,-1,-1]):
    model.set_hop(ta, 0, 1, lvec)
  for lvec in ([0,0,0],[0,-1,0],[-1,-1,-1],[-1,0,-1]):
    model.set_hop(tb, 0, 1, lvec)

  # Symmetry is actually orthorhombic with a simple m_y mirror
  # and two diagonal mirror planes containing the y axis

  return model

# set model parameters and construct bulk model
delta=1.0   # site energy shift
ta=0.4      # six weaker hoppings
tb=0.7      # two stronger hoppings
bulk_model=set_model(delta,ta,tb)

# bulk_model.display()

# make slab model
nl=9 # number of layers
slab_model=bulk_model.cut_piece(nl,2,glue_edgs=False)
# remove top orbital so top and bottom have the same termination
slab_model=slab_model.remove_orb(2*nl-1)

print('\nConstructed %2d-layer slab model\n'%nl)

# slab_model.display()

# solve on grid to check insulating
nk=10
k_1d=np.linspace(0.,1.,nk,endpoint=False)
kpts=[]
for kx in k_1d:
  for ky in k_1d:
    kpts.append([kx,ky])
evals=slab_model.solve_all(kpts)
# delta > 0, so there are nl valence and nl-1 conduction bands
e_vb=evals[:nl,:]
e_cb=evals[nl+1:,:]

print("VB min,max = %6.3f , %6.3f"%(np.min(e_vb),np.max(e_vb)))
print("CB min,max = %6.3f , %6.3f"%(np.min(e_cb),np.max(e_cb)))

# initialize and fill wf_array object for Bloch functions
nk=9
bloch_arr=wf_array(slab_model,[nk,nk])
bloch_arr.solve_on_grid([0.0, 0.0])
#
# initalize wf_array to hold HWFs, and Numpy array for HWFCs
hwf_arr=bloch_arr.empty_like(nsta_arr=nl)
hwfc=np.zeros([nk,nk,nl])

# loop over k points and fill arrays with HW centers and vectors
for ix in range(nk):
  for iy in range(nk):
    (val,vec)=bloch_arr.position_hwf([ix,iy],occ=list(range(nl)),
        dir=2,hwf_evec=True,basis="orbital")
    hwfc[ix,iy]=val
    hwf_arr[ix,iy]=vec
# impose periodic boundary conditions
hwf_arr.impose_pbc(0,0)
hwf_arr.impose_pbc(1,1)

# compute and print mean and standard deviation of Wannier centers by layer
print('\nLocations of hybrid Wannier centers along z:\n')
print('  Layer      '+nl*'  %2d    '%tuple(range(nl)))
print('  Mean   '+nl*'%8.4f'%tuple(np.mean(hwfc,axis=(0,1))))
print('  Std Dev'+nl*'%8.4f'%tuple(np.std(hwfc,axis=(0,1))))

# compute and print layer contributions to polarization along x, then y
px=np.zeros((nl,nk))
py=np.zeros((nl,nk))
for n in range(nl):
  px[n,:]=hwf_arr.berry_phase(dir=0,occ=[n])/(2.*np.pi)

print('\nBerry phases along x (rows correspond to k_y points):\n')
print('  Layer      '+nl*'  %2d    '%tuple(range(nl)))
for k in range(nk):
  print('         '+nl*'%8.4f'%tuple(px[:,k]))
# when averaging, don't count last k-point
px_mean=np.mean(px[:,:-1],axis=1)
print('\n  Ave    '+nl*'%8.4f'%tuple(px_mean))

# Similar calculations along y give zero due to m_y mirror')

nlh=nl//2
sum_top=np.sum(px_mean[:nlh])
sum_bot=np.sum(px_mean[-nlh:])
print('\n  Surface sums: Top, Bottom = %8.4f , %8.4f\n'%(sum_top,sum_bot))

# These quantities are essentially the "surface polarizations" of the
# model as defined within the hybrid Wannier gauge.  See, e.g.,
# S. Ren, I. Souza, and D. Vanderbilt, "Quadrupole moments, edge
# polarizations, and corner charges in the Wannier representation,"
# Phys. Rev. B 103, 035147 (2021).

# Make bar chart
fig = plt.figure()
plt.bar(range(nl),px_mean)
plt.axhline(0.,linewidth=0.8,color='k')
plt.xticks(range(nl))
plt.xlabel("Layer index of hybrid Wannier band")
plt.ylabel(r"Contribution to $P_x$")
fig.tight_layout()
fig.savefig("cubic_slab_hwf.pdf")
