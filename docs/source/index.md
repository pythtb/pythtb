---
myst:
  html_meta:
    "description lang=en": "PythTB is a Python package for constructing and analyzing tight-binding models with a focus on topology and quantum geometry."
    "keywords": "PythTB, PyTB, Python, tight binding, Wannier, Berry,
                topological insulator, Chern, Haldane, Kane-Mele, Z2, graphene,
                band structure, wavefunction, bloch, periodic insulator,
                wannier90, wannier function, density functional theory,
                DFT, first-principles"
    "property=og:locale": "en_US"
---

(_pythtb_mainpage)=
# Python Tight Binding (PythTB)

PythTB is a Python library for constructing and analyzing tight-binding models, built for modern topological band theory applications. It provides a streamlined path from model specification to physical interpretation, making it useful for both learning electronic structure and conducting research-level studies. With only a few lines of code, you can define lattice models, build tight-binding Hamiltonians, and compute electronic properties.

```{admonition} PythTB 2.0.0 Released!
:class: important

[Release Notes](release/2.0.0-notes)  
[Changelog](CHANGELOG.md)
```

```{admonition} Quick Links
:class: seealso

- [GitHub](https://github.com/pythtb/pythtb) - source code and issue tracker
- {doc}`Installation <install>` - install instructions and dependencies
- {doc}`API <api>` - detailed API reference
- {doc}`Tutorials <tutorials>` - Jupyter notebooks demonstrating key features
- {doc}`Development <development>` - contributing guidelines and developer docs
- {doc}`Release Notes <release>` - discussion of new features by version
- {doc}`Changelog <CHANGELOG>` - list of changes by version
- {doc}`Formalism <formalism>` - theoretical background 
- {doc}`Citation <citation>` - how to cite PythTB in publications
```


## Core functionality

```{eval-rst}
.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: :material-outlined:`hub` Model construction
      :link: generated/pythtb.TBModel
      :link-type: doc

      Construct and manipulate tight-binding Hamiltonians with ``TBModel``.
      Define lattice geometry with ``Lattice``, on-site and hopping terms, spin structure,
      and parameter-dependent model components.

   .. grid-item-card:: :material-outlined:`blur_on` State sampling
      :link: generated/pythtb.WFArray
      :link-type: doc

      Create structured k-space and parameter meshes, sample model
      Hamiltonians, and store eigenstates in ``WFArray`` objects for
      downstream analysis.

   .. grid-item-card:: :material-outlined:`all_inclusive` Topology & quantum geometry
      :link: api
      :link-type: doc

      Compute topological invariants and quantum-geometric quantities
      with ``WFArray`` and ``TBModel``: Berry connection, phases and
      curvature; quantum metric and geometric tensor; Wilson loops,
      Chern numbers, local Chern markers, hybrid Wannier centers, and
      Chern-Simons axion-angle pumping.

   .. grid-item-card:: :material-outlined:`extension` Wannier90 Integration
      :link: generated/pythtb.W90
      :link-type: doc

      Import Wannier90 tight-binding Hamiltonians via ``W90`` 
      for post-processing, band-structure evaluation, and detailed analysis of
      topological and quantum-geometric properties.

   .. grid-item-card:: :material-outlined:`token` Wannier workflows
      :link: generated/pythtb.Wannier
      :link-type: doc

      Perform projection, disentanglement, and construct maximally 
      localized Wannier functions in a tight-binding framework 
      using ``Wannier`` and ``WFArray``. Evaluate spreads, centers, 
      localization properties, and topological obstructions.
   
   .. grid-item-card:: :material-outlined:`image` Visualization
      :link: api
      :link-type: doc

      Plot band structures, density of states, lattice geometries,
      hopping graphs, and interactive 3D models with built-in 
      visualization utilities.
```

## Get started with PythTB

This is a simple example showing how to define graphene tight-binding
model with first neighbor hopping only. Below is the source code and
plot of the resulting band structure. Here you can find {doc}`more examples <tutorials>`.

```{literalinclude} _static/get_started/graphene_bands.py
:language: python
```

```{eval-rst}
.. list-table::
   :widths: 50 50

   * - .. figure:: _static/get_started/graphene_lattice.png
           :width: 100%

           Graphene lattice
     - .. figure:: _static/get_started/graphene_bands.png
           :width: 100%

           Graphene band structure
```

## Feedback

Please send comments or suggestions for improvement to [these email addresses](mailto:trey@treycole.me;dhv@physics.rutgers.edu;sinisacoh@gmail.com) or start
a discussion on the [GitHub Discussions page](https://github.com/orgs/pythtb/discussions). If you find bugs, please report them on the [GitHub Issues page](https://github.com/pythtb/PythTB/issues). 

```{toctree}
:maxdepth: 1
:hidden:

About <about>
install
API <api>
Tutorials <tutorials>
Development <development>
release
CHANGELOG
formalism
resources
citation
```