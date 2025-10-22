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

   .. grid-item-card:: :material-outlined:`hub` Model construction
      :link: generated/pythtb.TBModel
      :link-type: doc

      Build tight-binding models with ``TBModel``. Define unit cells and
      lattice geometry with ``Lattice``.

   .. grid-item-card:: :material-outlined:`blur_on` State sampling
      :link: generated/pythtb.Mesh
      :link-type: doc

      Create k-space and parameter sampling-meshes with ``Mesh``.

   .. grid-item-card:: :material-outlined:`image` Visualization
      :link: usage
      :link-type: doc

      Plot bands, DOS, lattices, and TB graphs with built-in plotting helpers.

   .. grid-item-card:: :material-outlined:`all_inclusive` Topology & quantum geometry
      :link: generated/pythtb.WFArray
      :link-type: doc

      Evaluate Berry connections, curvature, and phases, compute Chern numbers, follow hybrid Wannier centers, 
      and analyze adiabatic cycles with ``WFArray`` tools.

   .. grid-item-card:: :material-outlined:`extension` Wannier90 Integration
      :link: generated/pythtb.W90
      :link-type: doc

      Import Wannier90 tight-binding Hamiltonians via ``W90`` and continue analysis inside PythTB.

   .. grid-item-card:: :material-outlined:`token` Wannier workflows
      :link: generated/pythtb.Wannier
      :link-type: doc

      Build maximally localized Wannier functions with ``Wannier``; do single-shot projections, evaluate spreads, and visualize centers, 
      densities, and decay profiles.


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


Get started with PythTB
-----------------------
This is a simple example showing how to define graphene tight-binding
model with first neighbour hopping only. Below is the source code and
plot of the resulting band structure. Here you can find :doc:`more examples <examples>`.

.. literalinclude:: get_started/graphene_bands.py
   :language: python
   :linenos:

.. raw:: html

   <div style="display: flex; gap: 1rem;">
     <figure style="flex: 1; margin: 0;">
       <img src="_images/graphene_lattice.png" alt="Graphene lattice" style="width:100%;"/>
       <figcaption>Graphene lattice</figcaption>
     </figure>
     <figure style="flex: 1; margin: 0;">
       <img src="_images/graphene_bands.png" alt="Graphene band structure" style="width:100%;"/>
       <figcaption>Graphene band structure</figcaption>
     </figure>
   </div>


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


Feedback
--------

Please send comments or suggestions for improvement to `these email
addresses <mailto: trey@treycole.me, dhv@physics.rutgers.edu, sinisacoh@gmail.com>`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   Home <self>
   About <about>
   install
   usage
   examples
   CHANGELOG
   formalism
   resources
   citation