.. _pythtb_mainpage:

.. meta::
   :keywords: PythTB, PyTB, python, tight binding, Wannier, Berry,
              topological insulator, Chern, Haldane, Kane-Mele, Z2, graphene,
              band structure, wavefunction, bloch, periodic insulator,
              wannier90, wannier function, density functional theory,
              DFT, first-principles

=============================
Python Tight Binding (PythTB)
=============================

PythTB is a pure-Python toolbox for building and analyzing tight-binding models. With a few lines of code, you can define lattices, 
assign hopping parameters, diagonalize the Hamiltonians on custom meshes, plot band structures, and evaluate quantum-geometry objects
such as Berry phases, curvatures, and Chern numbers. The package also reads Wannier90 output so you can work directly with Wannierized 
models coming from first-principles calculations.

.. admonition:: Quick Links
    :class: seealso

    - :doc:`Installation <install>` - install instructions and dependencies
    - :doc:`Usage guide <usage>` - APIs, workflows, and tips
    - :doc:`Examples <examples>` - example scripts and notebooks

Core functionality
------------------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: :material-regular:`build` Model construction
      :link: usage
      :link-type: doc

      Build lattices with ``Lattice`` and populate hoppings via ``TBModel``.

   .. grid-item-card:: :material-regular:`play_circle` Sampling & eigenstates
      :link: usage
      :link-type: doc

      Create k-space or parameter meshes with ``Mesh.build_grid`` / ``Mesh.build_path`` and store eigenvectors in a ``WFArray`` that tracks phases, energies, overlaps, and quantum geometry.

   .. grid-item-card:: :material-regular:`category` Topology & quantum geometry
      :link: usage
      :link-type: doc

      Evaluate Berry connections/curvature, compute Berry phases & Chern numbers, follow hybrid Wannier centers, and analyze adiabatic cycles with ``WFArray`` tools.

   .. grid-item-card:: :material-regular:`integration_instructions` Wannier90 Integration
      :link: usage
      :link-type: doc

      Import Wannier90 tight-binding Hamiltonians via ``W90`` and continue analysis inside PythTB.

   .. grid-item-card:: :material-regular:`widgets` Wannier workflows
      :link: usage
      :link-type: doc

      Build maximally localized Wannier functions with ``Wannier``; do single-shot projections, evaluate spreads, and visualize centers, densities, and decay profiles.

   .. grid-item-card:: :material-regular:`image` Visualization and export
      :link: usage
      :link-type: doc

      Plot bands, DOS, lattices, and TB graphs with ``pythtb.plotting`` and export data for downstream workflows.

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
  It supports single-shot projections, gauge refinement, spread and center analysis, and plotting helpers such as ``plot_centers``, 
  ``plot_decay``, and ``plot_density``.
- ``W90`` reads tight-binding Hamiltonians from Wannier90 output files (``*.win``, ``*_hr.dat``, ``*_centres.xyz``). 
  You can combine imported data with the broader PythTB ecosystem, run band-structure checks, or feed the states into 
  ``WFArray`` and ``Wannier`` for further processing.

See :doc:`usage <usage>` for an end-to-end walkthrough and :doc:`examples <examples>` for notebooks that demonstrate these workflows.


Get started with PythTB
-----------------------
This is a simple example showing how to define graphene tight-binding
model with first neighbour hopping only. Below is the source code and
plot of the resulting band structure. Here you can find :doc:`more examples <examples>`.

.. thebe-button:: Launch interactive session

.. container:: thebe

   .. code-block:: python
      :class: thebe thebe-init

      from pythtb import TBModel
      import numpy as np
      import matplotlib.pyplot as plt

      lat = [[1, 0], [1/2, np.sqrt(3)/2]]
      orb = [[1/3, 1/3], [2/3, 2/3]]

      model = TBModel(2, 2, lat, orb)
      model.set_hop(-1.0, 0, 1, [0, 0])
      model.set_hop(-1.0, 1, 0, [1, 0])
      model.set_hop(-1.0, 1, 0, [0, 1])

      k_nodes = [[0.0, 0.0], [1./3., 2./3.], [0.5, 0.5]]
      nk = 100
      k_vec, k_dist, k_node = model.k_path(k_nodes, nk, report=False)
      evals = model.solve_ham(k_vec)

      fig, ax = plt.subplots()
      ax.plot(k_dist, evals)
      ax.set_xticks(k_node)
      ax.set_xticklabels([r"$\Gamma$", r"$K$", r"$M$"])
      plt.show()

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
including those listed in the :ref:`Acknowledgments <Acknowledgments>` section.

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
^^^^^^^^^^

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


Feedback
--------

Please send comments or suggestions for improvement to `these email
addresses <mailto: trey@treycole.me, dhv@physics.rutgers.edu, sinisacoh@gmail.com>`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   Home <self>
   install
   usage
   examples
   CHANGELOG
   formalism
   resources
   citation