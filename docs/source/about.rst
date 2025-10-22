
Motivations and capabilities
----------------------------

The ``PythTB`` package was written in Python for several reasons, including

- The ease of learning and using Python
- The wide availability of Python in the community
- The flexibility with which Python can be interfaced with graphics and visualization modules
- In general, the easy extensibility of Python programs

You can get an idea of the capabilities of the package by browsing the :doc:`PythTB examples <examples>`.

Tight-binding models
^^^^^^^^^^^^^^^^^^^^^

The `tight binding <http://en.wikipedia.org/wiki/Tight_binding>`_ method is an approximate approach for solving for the electronic wave 
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
`Slater-Koster forms <http://en.wikipedia.org/wiki/Tight_binding#Table_of_interatomic_matrix_elements>`_ for interactions 
between *s*, *p* and *d* orbitals are *not currently coded*, although the addition of such features could be considered for 
a future release.

Topology and quantum geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``WFArray`` cooperates with ``Mesh`` to evaluate Berry phases, Berry connections, curvature, and Chern numbers on closed loops, 
uniform grids, or mixed k–parameter meshes. Hybrid Wannier functions, polarization, adiabatic pumping, and related observables 
follow naturally once the states are stored in a consistent gauge.

Wannier functions and Wannier90 interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting with Version 1.7, ``PythTB`` provides two complementary Wannier toolchains:

- ``Wannier`` constructs maximally localized Wannier functions directly from PythTB wavefunctions. 
  It supports single-shot projections, disentanglement, maximal localization, spread and center analysis, 
  and plotting helpers such as ``plot_centers``, ``plot_decay``, and ``plot_density``.
- ``W90`` reads tight-binding Hamiltonians from Wannier90 output files (``*.win``, ``*_hr.dat``, ``*_centres.xyz``). 
  You can combine imported data with the broader PythTB ecosystem, run band-structure checks, or feed the states into 
  ``WFArray`` and ``Wannier`` for further processing.

See :doc:`usage <usage>` for an end-to-end walkthrough and :doc:`examples <examples>` for notebooks that demonstrate these workflows.


.. _history:

History
-------

This code package had its origins in a simpler package that was
developed for use in a special-topics course on “Berry Phases in Solid
State Physics” offered by D. Vanderbilt in Fall 2010 at Rutgers
University. The students were asked to use the code as provided, or to
make extensions on their own as needed, in order to compute properties
of simple systems, such as a 2D honeycomb model of graphene, in the
tight-binding (TB) approximation. Sinisa Coh, who was a PhD student
with Vanderbilt at the time, was the initial developer and primary maintainer
of the package. Since then, many others have contributed to its development,
including those listed below.

.. _Acknowledgments:

Acknowledgments
----------------
`PythTB` has benefited from the contributions of many individuals over the years. 
Below is a list of the current maintainers and contributors, along with their affiliations.
We apologize for any omissions, and welcome feedback and corrections. 

Maintainers
^^^^^^^^^^^^^^^^
- `Trey Cole <mailto: trey@treycole.me>`_ - Rutgers University
- `David Vanderbilt <mailto: dhv@physics.rutgers.edu>`_ - Rutgers University
- `Sinisa Coh <mailto: sinisacoh@gmail.com>`_ - University of California at Riverside (formerly Rutgers University)

Contributors
^^^^^^^^^^^^^^^^
We gratefully acknowledge additional contributions to PythTB from:

- Wenshuo Liu - formerly Rutgers University
- Victor Alexandrov - formerly Rutgers University
- Tahir Yusufaly - formerly Rutgers University
- Maryam Taherinejad - formerly Rutgers University


Funding
-------

This Web page is based in part upon work supported by the US National
Science Foundation under Grants DMR-1005838, DMR-1408838, DMR-1954856,
and DMR-2421895.  Any opinions, findings, and
conclusions or recommendations expressed in this material are those of
the author and do not necessarily reflect the views of the National
Science Foundation.


License
-------

Note that the ``PythTB`` code is freely distributed under the terms of
the :download:`GNU GPL public license <misc/LICENSE>`. You may
use it for your own research and educational purposes, or pass it on
to others for similar use. You may modify it, but if you do so
you must include a prominent notice stating that you have changed the
code and include a copy of this license.