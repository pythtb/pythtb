#!/usr/bin/env python

# Boron nitride ribbon: Compute Berry phase

# Copyright under GNU General Public License 2010, 2012, 2016, 2021
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt

# define lattice vectors
lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
# define coordinates of orbitals
orb=[[1./3.,1./3.],[2./3.,2./3.]]

# ------

# make two dimensional tight-binding boron nitride model
my_model=tb_model(2,2,lat,orb)

# set periodic model
delta=0.4
t=-1.0
my_model.set_onsite([-delta,delta])
my_model.set_hop(t, 0, 1, [ 0, 0])
my_model.set_hop(t, 1, 0, [ 1, 0])
my_model.set_hop(t, 1, 0, [ 0, 1])

# cut out 3 unit cells along second direction with open boundary
#   conditions to make ribbon model
model_orig=my_model.cut_piece(3,1,glue_edgs=False)

print('\n========================================================')
print('construct and display original model with tilted')
print('nonperiodic lattice vector')
print('========================================================\n')
model_orig.display()

# ------

# reset second lattice vector normal to the first one
# model_perp=model_orig.change_nonperiodic_vector(1,to_home_suppress_warning=True)
print('\n========================================================')
print('construct and display new model with nonperiodic lattice')
print('vector changed to be normal to the periodic direction')
print('========================================================\n')

model_perp=model_orig.change_nonperiodic_vector(1)
model_perp.display()

# ------

# initialize figure with subplots
fig, ax = plt.subplots(1,2,figsize=(6.5,2.8))

# ------------------------------------------------------------------
# function to print model, plot band structure, and compute Berry phase
def run_model(model,panel):
  numk=41
  (k_vec,k_dist,k_node)=model.k_path([[-0.5],[0.5]],numk,report=False)
  (eval,evec)=model.solve_all(k_vec,eig_vectors=True)
  #
  # plot band structure
  ax[panel].set_title("Band structure - "+["original","modified"][panel])
  ax[panel].set_xlabel("Reduced wavevector")
  ax[panel].set_ylabel("Band energy")
  ax[panel].set_xlim(-0.5,0.5)
  n_bands=eval.shape[0]
  for band in range(n_bands):
    ax[panel].plot(k_vec,eval[band,:],"k-",linewidth=0.5)
  #
  # compute and print Berry phase at half filling
  wf=wf_array(model,[numk])
  wf.solve_on_grid([0.])
  n_occ=n_bands//2
  berry_phase=wf.berry_phase(range(n_occ),dir=0)
  print('  Berry phase = %10.7f\n'%(berry_phase,))
  return()
# ------------------------------------------------------------------

print('\n========================================================')
print('solve both models, showing that the band structures are')
print('the same, but Berry phases are different')
print('========================================================\n')

print('Original model\n')
run_model(model_orig,0)

print('Revised model\n')
run_model(model_perp,1)

# save figure
fig.tight_layout()
fig.savefig("bn_ribbon_berry.pdf")
print('Band structures have been saved to "bn_ribbon_berry.pdf"\n')

# Notes
#
# Let x be along the extended direction and y be normal to it.
#
# This model has an M_x mirror symmetry, so the Berry phase is
# expected to be 0 or pi.  We find it to be zero, but only after the
# 'change_nonperiodic_vector' method is used to force the nonperiodic
# "lattice vector" to be perpedicular to the extended direction.
# 
# The physical meaning of the Berry phase in the original model
# calculation is rather subtle.  It is related to the position of
# the joint Wannier center (i.e., summed over occupied bands) in
# the direction of reciprocal lattice vector 0, which has a
# y component as well as an x component (since it must be normal
# to real space lattice vector 1).  The joint Wannier center gets
# displaced along y as the hopping 't' is changed, so the Berry
# phase calculation gets "contaminated" by this displacement.
