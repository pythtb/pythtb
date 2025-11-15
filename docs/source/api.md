# PythTB API Reference

```{eval-rst}
.. currentmodule:: pythtb
.. module:: pythtb
```

## Core Classes
PythTB centers on a small set of core classes for constructing and analyzing tight-binding models. This page highlights what each component is for and links to the API reference in the tables below.

- {py:class}`pythtb.TBModel` 
   Tight-binding Hamiltonians on arbitrary lattices. Use it to set hoppings, sweep parameters, compute spectra, and evaluate derived quantities such as Berry curvature, quantum geometric tensors, axion angles, or local Chern markers.
- {py:class}`pythtb.WFArray` 
   Mesh-aware wavefunction storage. Solve a model across $(k, \lambda)$ grids, then evaluate Wilson loops, Berry phases, Chern numbers, and other objects with consistent periodic boundary conditions.
- {py:class}`pythtb.W90` 
   Interface with [Wannier90](http://www.wannier.org) and export a Wannierized tight-binding model in the form of a 
   {py:class}`TBModel`. 

:::{versionadded} 2.0.0
The classes below were introduced in PythTB version 2.0.0.
:::

- {py:class}`pythtb.Mesh` 
   Describes combined crystal momentum and parameter meshes, $(k, \lambda)$. Can construct uniform grids, paths, and meshes defined on a custom set of points. Encodes boundary conditions 
   (loops, endpoints, adiabatic cycles) for {py:class}`WFArray` to apply appropriate 
   gauge conditions downstream.
- {py:class}`pythtb.Lattice`
   Holds real- and reciprocal-space geometry, orbital positions, and nearest-neighbor shells. 
   Every model and wavefunction array references the same lattice instance to ensure consistent coordinates.
- {py:class}`pythtb.Wannier` 
   Build Wannier gauges directly inside PythTB from a {py:class}`WFArray`, 
   perform projections, disentanglement and maximal localization, and analyze 
   spreads and centers.


```{eval-rst}
.. autosummary::
   :toctree: generated/
   :caption: PythTB Classes
   :template: autosummary/public_class.rst

   TBModel
   WFArray
   Lattice
   Mesh
   W90
   Wannier
```

## Predefined Models
PythTB also provides a collection of predefined tight-binding models. Import and use these models
using the following syntax:
```python
from pythtb.models import haldane, graphene, ssh
```

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :caption: PythTB Models
   :recursive:

   models
```

## I/O Utilities

PythTB ships with lightweight readers for importing data from external software packages such as Wannier90 and Quantum ESPRESSO.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :caption: I/O Utilities
   :template: autosummary/public_module.rst

   io.w90
   io.qe
```
