About
=====

The "**Pyth**\ on **T**\ ight **B**\ inding" (``PythTB``) code was
developed and is maintained by Sinisa Coh (University of California at
Riverside) and David Vanderbilt (Rutgers University) with assistance
from :ref:`many other individuals <history>`.

The primary location for this package is at
`<http://www.physics.rutgers.edu/pythtb>`_ where the most up-to-date
releases and information can be found.

Motivations and capabilities
----------------------------

The `tight binding <http://en.wikipedia.org/wiki/Tight_binding>`_
method is an approximate approach for solving for the electronic wave
functions for electrons in solids assuming a basis of localized
atomic-like orbitals. We assume here that the orbitals are
orthonormal, and focus on the “empirical tight binding” approach in
which the Hamiltonian matrix elements are simply parametrized, as
opposed to being computed ab-initio.

The ``PythTB`` package is intended to set up and solve tight-binding
models for the electronic structure of

-  0D clusters
-  1D chains and ladders
-  2D layers (square lattice, hexagonal lattice, honeycomb lattice,
   etc.)
-  3D crystals
-  clusters, ribbons, slabs, etc., cut from higher-dimensional crystals
-  etc.

As currently written, it is not intended to handle realistic chemical
interactions. So for example, the `Slater-Koster forms
<http://en.wikipedia.org/wiki/Tight_binding#Table_of_interatomic_matrix_elements>`_
for interactions between *s*, *p* and *d* orbitals are *not currently
coded*, although the addition of such features could be considered for
a future release.

The ``PythTB`` package includes capabilities for

-  computing electron eigenvalues and eigenvectors at selected k-points
   or on a mesh of k-points
-  generating band-structure plots
-  generating density-of-states plots

It can also calculate `Berry phases, connections and curvatures
<http://en.wikipedia.org/wiki/Berry_connection_and_curvature>`_ in
ways that are useful for calculations of

-  adiabatic charge transport
-  electric polarization
-  anomalous Hall conductivity
-  topological indices
-  etc.

Finally, it provides tools for setting up more complicated
tight-binding models, e.g., by “cutting” a cluster, ribbon, or slab
out of a higher-dimensional crystal, and for visualizing the
connectivity of a tight-binding model once it has been
constructed. You can get an idea of the capabilities of the package by
browsing the :doc:`PythTB examples <examples>`.

The code is intended for pedagogical as well as research purposes. For
example, it should be suitable for use in an undergraduate- or
graduate-level solid-state physics course as a tool for illustrating the
calculation of electronic band structructures, and it is simple enough
that it can be considered for use in homework sets or special projects
in such a course.

The ``PythTB`` package was written in Python for several reasons,
including

-  The ease of learning and using Python
-  The wide availability of Python in the community
-  The flexibility with which Python can be interfaced with graphics and
   visualization modules
-  In general, the easy extensibility of Python programs

On the other hand, please note that Python is not a computationally
efficient platform when applied to large systems requiring heavy
computation.

Getting started
---------------

The best way to explore the capabilities of ``PythTB`` and to get
started in using it is to read through the :doc:`installation
instructions <install>` and :doc:`PythTB examples <examples>`.

If you are unfamiliar with Python or are not sure whether Python and the
needed Python modules are installed on your system, see our
:doc:`python introduction <resources>` and :doc:`installation
instructions <install>`.

Note that the ``PythTB`` code is freely distributed under the terms of
the :download:`GNU GPL public license <misc/gpl-3.0.txt>`. You may
use it for your own research and educational purposes, or pass it on
to others for similar use. However, the code is not guaranteed to be
bug-free, and we do not promise active support for the package.

Wannier90 interface
-------------------

Starting with Version 1.7, ``PythTB`` also provides an interface
to the `Wannier90 <http://wannier.org>`_ code, which can
be used to take the output of a first-principles density-functional
calculation and construct from it a tight-binding model, in
the basis of Wannier functions, that accurately reproduces the
first-principles bandstructure.  See :doc:`usage <usage>`.

.. _history:

History
-------

This code package had its origins in a simpler package that was
developed for use in a special-topics course on “Berry Phases in Solid
State Physics” offered by D. Vanderbilt in Fall 2010 at Rutgers
University. The students were asked to use the code as provided, or to
make extensions on their own as needed, in order to compute properties
of simple systems, such as a 2D honeycomb model of graphene, in the
tight-binding (TB) approximation.

From the beginning, Sinisa Coh, who was a PhD student with Vanderbilt at
the time, has been the primary developer of the package. However, many
other individuals made contributions to this code, including
Wenshuo Liu, Victor Alexandrov, Tahir Yusufaly, and Maryam Taherinejad.

Feedback
--------

Please send comments or suggestions for improvement to `these email
addresses <mailto:dhv@physics.rutgers.edu,sinisacoh@gmail.com>`_.

Acknowledgments and Disclaimer
------------------------------

This Web page is based in part upon work supported by the US National
Science Foundation under Grants DMR-1005838, DMR-1408838, DMR-1954856,
and DMR-2421895.  Any opinions, findings, and
conclusions or recommendations expressed in this material are those of
the author and do not necessarily reflect the views of the National
Science Foundation.
