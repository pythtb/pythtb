#!/usr/bin/env python

# Toy graphene model

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb.tb_model import *  # import TB model class
import numpy as np
import matplotlib.pyplot as plt

# define lattice vectors
lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
# define coordinates of orbitals
orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

# make two dimensional tight-binding graphene model
my_model = tb_model(2, 2, lat, orb)

# set model parameters
delta = 0.3
t = -1.0

# set on-site energies
my_model.set_onsite([-delta, delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t, 0, 1, [0, 0])
my_model.set_hop(t, 1, 0, [1, 0])
my_model.set_hop(t, 1, 0, [0, 1])

# print tight-binding model
my_model.report()

# generate list of k-points following a segmented path in the BZ
# list of nodes (high-symmetry points) that will be connected
path = [[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.5, 0.5], [0.0, 0.0]]
# labels of the nodes
label = (r"$\Gamma $", r"$K$", r"$M$", r"$\Gamma $")
# total number of interpolated k-points along the path
nk = 121


print("---------------------------------------")
print("starting calculation")
print("---------------------------------------")
print("Calculating bands...")

my_model.plot_bands(k_path=path, k_label=label, nk=nk, show=True, proj_orb_idx=[0])

print("Done.\n")
