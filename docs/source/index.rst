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

      Define and manipulate tight-binding Hamiltonians using ``TBModel``.
      Specify on-site terms, hoppings, spin structure, and
      parameter-dependent contributions.

   .. grid-item-card:: :material-outlined:`blur_on` State sampling
      :link: generated/pythtb.WFArray
      :link-type: doc

      Build structured k-space and parameter meshes with ``Mesh``.
      Sample Hamiltonians and store resulting states in ``WFArray`` for
      further analysis.

   .. grid-item-card:: :material-outlined:`all_inclusive` Topology & quantum geometry
      :link: usage
      :link-type: doc

      Compute Berry phases, connections, and curvature; Chern numbers;
      the axion angle; local Chern markers; hybrid Wannier
      functions; and other quantum-geometric observables using
      ``WFArray`` and ``TBModel`` methods.

   .. grid-item-card:: :material-outlined:`extension` Wannier90 Integration
      :link: generated/pythtb.W90
      :link-type: doc

      Import Wannier90 tight-binding Hamiltonians via ``W90`` 
      for post-processing and topological/quantum-geometric analysis.

   .. grid-item-card:: :material-outlined:`token` Wannier workflows
      :link: generated/pythtb.Wannier
      :link-type: doc

      Construct maximally localized Wannier functions with ``Wannier``.
      Perform projections, disentanglement, evaluate spreads, and analyze centers
      and localization properties.
   
   .. grid-item-card:: :material-outlined:`image` Visualization
      :link: usage
      :link-type: doc

      Plot band structures, density of states, lattice geometries, and
      hopping graphs with built-in visualization utilities.


Get started with PythTB
-----------------------
This is a simple example showing how to define graphene tight-binding
model with first neighbour hopping only. Below is the source code and
plot of the resulting band structure. Here you can find :doc:`more examples <examples>`.

.. literalinclude:: get_started/graphene_bands.py
   :language: python

.. list-table::
   :widths: 50 50

   * - .. figure:: get_started/graphene_lattice.png
           :width: 100%

           Graphene lattice
     - .. figure:: get_started/graphene_bands.png
           :width: 100%

           Graphene band structure


Feedback
--------

Please send comments or suggestions for improvement to `these email
addresses <mailto: trey@treycole.me, dhv@physics.rutgers.edu, sinisacoh@gmail.com>`_.
If you find bugs, please report them on the `GitHub Issues page
<https://github.com/pythtb/PythTB/issues>`_. 

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