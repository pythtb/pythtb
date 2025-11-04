# Usage

- {py:class}`pythtb.TBModel` 
   Class for constructing tight-binding models and their Hamiltonians. It provides methods to compute band structures,
   density of states, quantum geometric tensors, local Chern markers, axion angles, and other related quantities.
- {py:class}`pythtb.WFArray` 
   Class for storing wavefunctions on a parameter mesh, and computing Wilson loops, Berry phases, Berry connections, Berry curvatures, Chern numbers, and other related quantities.
- {py:class}`pythtb.Mesh` 
   Class for constructing meshes or paths of k-points and parameter points. This class 
   stores information about grid topology, such as periodic boundary conditions, which get
   passed on to the {py:class}`pythtb.WFArray` class for wavefunction storage and manipulation.
- {py:class}`pythtb.Lattice`
   Class for storing information about the lattice structure of the tight-binding model, including
   lattice vectors, reciprocal lattice vectors, and nearest neighbor vectors. This class is used
   by the {py:class}`pythtb.TBModel` and {py:class}`pythtb.WFArray` classes.
- {py:class}`pythtb.W90` 
   Class for interfacing `PythTB` with [Wannier90](http://www.wannier.org) allowing for the construction
   of tight-binding models based on first-principles density functional theory calculations.
- {py:class}`pythtb.Wannier` 
   Class for constructing Wannier functions from Bloch wavefunctions defined on a full k-mesh with 
   the help of the {py:class}`pythtb.WFArray` class. The quadratic spread can then be minimized using the disentanglement and maximal localization algorithms.
- {py:mod}`pythtb.models` 
   A collection of predefined tight-binding models. Import and use these models
   using the following syntax:
   ```python
   from pythtb.models import haldane, graphene, ssh
   ```

```{eval-rst}
.. currentmodule:: pythtb
```

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
```{eval-rst}
.. autosummary::
   :toctree: generated/
   :caption: PythTB Models

   models.ssh
   models.checkerboard
   models.graphene
   models.haldane
   models.kane_mele
   models.fu_kane_mele
```


