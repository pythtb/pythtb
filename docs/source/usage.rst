Usage
=====

PythTB module consists of these primary classes:

* :class:`pythtb.TBModel` 
   Class for constructing tight-binding models and their Hamiltonians.
* :class:`pythtb.WFArray` 
   Class for storing wavefunctions on a parameter mesh, and computing Berry phases, Berry curvatures,
   Chern numbers, and other related quantities.
* :class:`pythtb.Mesh` 
   Class for constructing meshes or paths of k-points and parameter points. This class 
   stores information about grid topology, such as periodic boundary conditions, which get
   passed on to the :class:`pythtb.WFArray` class for wavefunction storage and manipulation.
* :class:`pythtb.W90` 
   Class for interfacing `PythTB` with `Wannier90 <http://www.wannier.org>`_ allowing for the construction
   of tight-binding models based on first-principles density functional theory calculations.
* :class:`pythtb.Wannier` 
   Class for constructing Wannier functions from Bloch wavefunctions defined on a full mesh. These Bloch
   wavefunctions can be obtained from either tight-binding models or from first-principles calculations using the
   :class:`pythtb.W90` class. The Wannier functions' spread can then be minimized using the disentanglement and
   maximal localization algorithms implemented in the :class:`pythtb.Wannier` class.

.. currentmodule:: pythtb

.. autosummary::
   :toctree: generated/
   :caption: PythTB Classes
   :recursive:

   TBModel
   WFArray
   Mesh
   W90
   Wannier

In addition, PythTB provides a visualization module :mod:`pythtb.plotting` 
for plotting and analyzing the results obtained 
from the tight-binding models. This module includes functions for visualizing band structures, 
density of states, and the geometry of the tight-binding model. 

.. autosummary::
   :toctree: generated/
   :caption: PythTB Plotting

   plotting.plot_bands
   plotting.plot_tb_model
   plotting.plot_tb_model_3d

Lastly, there is a collection of predefined tight-binding models available in PythTB in
:mod:`pythtb.models`.

.. autosummary::
   :toctree: generated/
   :caption: PythTB Models

   models.ssh
   models.checkerboard
   models.graphene
   models.haldane
   models.kane_mele
   models.fu_kane_mele

.. .. automodule:: pythtb
..    :undoc-members:
..    :show-inheritance:
..    :noindex:
..    :members:



