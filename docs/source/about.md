# Motivations and capabilities

The ``PythTB`` package was written in Python for several reasons, including

- The ease of learning and using Python
- The wide availability of Python in the community
- The flexibility with which Python can be interfaced with graphics and visualization modules
- In general, the easy extensibility of Python programs

You can get an idea of the capabilities of the package by browsing {doc}`API <api>` 
for code documentation and the {doc}`tutorials <tutorials>` for notebooks that 
demonstrate these workflows.

## Tight-binding models

The [tight binding](http://en.wikipedia.org/wiki/Tight_binding) method is an approximate approach for solving for the electronic wave 
functions for electrons in solids assuming a basis of localized atomic-like orbitals. We assume here that the orbitals are orthonormal, 
and focus on the “empirical tight binding” approach in which the Hamiltonian matrix elements are simply parametrized, as opposed to being 
computed ab-initio.

The ``PythTB`` package is intended to set up and solve tight-binding models for the electronic structure of

- 0D clusters
- 1D chains and ladders
- 2D layers (square lattice, hexagonal lattice, honeycomb lattice, etc.)
- 3D crystals
- clusters, ribbons, slabs, etc., cut from higher-dimensional crystals
- etc.

It provides tools for setting up more complicated tight-binding models, e.g., by “cutting” a cluster, ribbon, or slab out of a higher-dimensional crystal, 
and for visualizing the connectivity of a tight-binding model once it has been constructed.

As currently written, it is not intended to handle realistic chemical interactions. So for example, the 
[Slater-Koster forms](http://en.wikipedia.org/wiki/Tight_binding#Table_of_interatomic_matrix_elements) for interactions 
between *s*, *p* and *d* orbitals are *not currently coded*, although the addition of such features could be considered for a future release.

## Topology and quantum geometry
The ``PythTB`` package has a particular focus on computing topological and quantum geometric properties of tight-binding models. 

The {class}`~pythtb.tbmodel.TBModel` class is the core class for constructing and solving tight-binding models.
It provides methods to compute a variety of observables related to topology and quantum geometry, including:

- velocity operator, 
- quantum geometric tensor
- Berry curvature
- quantum metric
- Chern number
- local Chern marker
- axion angle 
- second Chern number
- position operator matrix elements
- hybrid Wannier functions

These quantities can be computed either at individual k-points, or on meshes of k-points in the Brillouin zone. 
Each of these observables can also be computed on arrays of adiabatic parameters that modify the Hamiltonian. 

{class}`~pythtb.wfarray.WFArray` cooperates with {class}`~pythtb.mesh.Mesh` and {class}`~pythtb.tbmodel.TBModel` 
to store states computed on structured k-point and/or parameter meshes. This is convenient for 
automatically managing periodic boundary conditions, and for storing states in a consistent gauge across the mesh. 
Once states are stored in a {class}`~pythtb.wfarray.WFArray`, one can use its methods to evaluate:

- Wilson loops
- Berry phases
- Berry connections
- Berry curvatures
- Chern numbers 
- hybrid Wannier functions
- position matrix elements

## Wannier functions and Wannier90 interface

Starting with Version 1.7, ``PythTB`` provides an interface to the 
popular [Wannier90](http://www.wannier.org) package via the {class}`~pythtb.w90.W90` class,
- {class}`~pythtb.w90.W90` reads tight-binding Hamiltonians from Wannier90 output files (``*.win``, ``*_hr.dat``, ``*_centres.xyz``). 
  You can combine imported data with the broader PythTB ecosystem, run band-structure checks, or feed the states into 
  {class}`~pythtb.wfarray.WFArray` and {class}`~pythtb.wannier.Wannier` for further processing.

Starting with Version 2.0, ``PythTB`` includes the {class}`~pythtb.wannier.Wannier` class, which constructs maximally localized Wannier 
functions directly from {class}`~pythtb.wfarray.WFArray` wavefunctions.

- {class}`~pythtb.wannier.Wannier` constructs maximally localized Wannier functions directly from PythTB wavefunctions. 
  It supports single-shot projections, disentanglement, maximal localization, spread and center analysis, 
  and plotting helpers such as {meth}`~pythtb.wannier.Wannier.plot_centers`, 
  {meth}`~pythtb.wannier.Wannier.plot_decay`, and {meth}`~pythtb.wannier.Wannier.plot_density`.

(history)=
# History

This code package had its origins in a simpler package that was developed for use in a special-topics course on “Berry Phases in Solid State Physics” offered by D. Vanderbilt in Fall 2010 at Rutgers University. The students were asked to use the code as provided, or to make extensions on their own as needed, in order to compute properties of simple systems, such as a 2D honeycomb model of graphene, in the tight-binding (TB) approximation. Sinisa Coh, who was a PhD student with Vanderbilt at the time, was the initial developer and primary maintainer of the package. Since then, many others have contributed to its development, including those listed below.

(Acknowledgments)=
# Acknowledgments

`PythTB` has benefited from the contributions of many individuals over the years. 
Below is a list of the current maintainers and contributors, along with their affiliations.
We apologize for any omissions, and welcome feedback and corrections. 

## Maintainers
- [Trey Cole](mailto:trey@treycole.me) - Rutgers University
- [David Vanderbilt](mailto:dhv@physics.rutgers.edu) - Rutgers University
- [Sinisa Coh](mailto:sinisacoh@gmail.com) - University of California at Riverside (formerly Rutgers University)

## Contributors
We gratefully acknowledge additional contributions to PythTB from:

- Wenshuo Liu - formerly Rutgers University
- Victor Alexandrov - formerly Rutgers University
- Tahir Yusufaly - formerly Rutgers University
- Maryam Taherinejad - formerly Rutgers University

# Funding

This Web page is based in part upon work supported by the US National Science Foundation under Grants DMR-1005838, DMR-1408838, DMR-1954856, and DMR-2421895. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author and do not necessarily reflect the views of the National Science Foundation.

# License

``PythTB`` is freely distributed under the terms of the [GNU public license](../../LICENSE). A copy of the license is included with the code distribution, and is also available online. You are free to use it for your own research and educational purposes, or pass it on to others for similar use. You may modify it, but if you do so you must include a prominent notice stating that you have changed the code and include a copy of this license.