Examples
========

If you are unfamiliar with Python or are not sure whether Python and
the needed Python modules are installed on your system, see our
:doc:`python introduction <resources>` and :doc:`installation
instructions <install>`.

You can download each example script below indvidually by clicking on
the 'Source code' link under each example.  Alternatively, you can 
download all example files as a single tar file

   :download:`pythtb-examples.tar.gz <misc/pythtb-examples.tar.gz>`

to untar this file use the following unix command::

        tar -zxf pythtb-examples.tar.gz

.. _simple-example:

Note that there is also a useful `collection of PythTB sample programs
<https://minisites.cambridgecore.org/berryphases/ptb_samples.html>`_ that
were developed in connection with David Vanderbilt's book `Berry Phases in
Electronic Structure Theory <https://www.cambridge.org/9781107157651>`_
(Cambridge University Press, 2018).

Simple example
--------------

After installing PythTB you can run the following :download:`simple example
<examples/simple.py>` either as::

    python simple.py

or by making script executable::

    chmod u+x simple.py

and then executing it with::

    ./simple.py
    
After you have executed the script, in that same folder file band.pdf
should appear and it should look similar to the figure below. 

.. plot :: examples/simple.py
   :include-source:

.. _checkerboard-example:

Checkerboard model
------------------

Simple two-dimensional model.

.. plot :: examples/checkerboard.py
   :include-source:

.. _trestle-example:

Trestle
-------

Simple model with one-dimensional k-space, two-dimensional r-space,
and with complex hoppings.

.. plot :: examples/trestle.py
   :include-source:

.. _0dim-example:

Molecule
--------

This is a zero-dimensional example of a molecule.

.. plot:: examples/0dim.py
   :include-source:

.. _graphene-example:   

Graphene model
--------------

This is a toy model of a two-dimensional graphene sheet.

.. plot :: examples/graphene.py
   :include-source:
  
.. _cone-example:

Berry phase around Dirac cone in graphene
-----------------------------------------

This example computes Berry phases for a circular path (in reduced
coordinates) around the Dirac point of the graphene band structure. In
order to have a well defined sign of the Berry phase, a small on-site
staggered potential is added in order to open a gap at the Dirac point.

After computing the Berry phase around the circular loop, it also computes
the integral of the Berry curvature over a small square patch in the
Brillouin zone containing the Dirac point, and plots individual phases
for each plaquette in the array.

.. plot :: examples/cone.py
   :include-source:

.. _3site_cycle-example:

One-dimensional cycle of 1D tight-binding model
-----------------------------------------------

This example considers a simple three-site one-dimensional tight-binding
model parametrized by some parameter :math:`\lambda`. As :math:`\lambda`
is changed from 0 to 1, the deepest onsite term is moved from the first
to second, then to the third, and then back to the first tight-binding
orbital. Therefore, we expect that Wannier center of the lowest band will
shift by one lattice vector as :math:`\lambda` changes from 0 to 1.

Also plots the individual on-site energies, band structure, and Wannier
center of lowest band.

.. plot :: examples/3site_cycle.py
   :include-source:


.. _3site_cycle_fin-example:

One-dimensional cycle on a finite 1D chain
------------------------------------------

This example is based on the same model as the one above but it
considers the effect of the one-dimensional cycle on the edge states
of a finite chain.

.. plot :: examples/3site_cycle_fin.py
   :include-source:


.. _haldane-example: 

Haldane model
-------------

`Haldane model <http://link.aps.org/doi/10.1103/PhysRevLett.61.2015>`_
is a graphene model with complex second neighbour hoppings.

Here we have calculated the density of states as well.

.. plot :: examples/haldane.py
   :include-source:

.. _haldane_fin-example:   

Finite Haldane model
--------------------

Now let us calculate the density of states for a finite piece of the Haldane
model with and without periodic boundary conditions (PBC).

.. plot :: examples/haldane_fin.py
   :include-source:

.. _edge-example:

Edge states
-----------

Plots the edge state-eigenfunction for a finite Haldane model that
is periodic either in both directions or in only one direction.

.. plot :: examples/edge.py
   :include-source:

.. _haldane_bp-example:

Berry phases in Haldane model
-----------------------------

Calculate Berry phases along :math:`k_x` (which are proportional
to the 1D Wannier center positions along :math:`x`) as a function
of :math:`k_y` for the Haldane model.  This is done first for each
band separately, then for both together.  Two different approaches,
one less and one more automated, are illustrated.  The results
indicate that the two bands have equal and opposite Chern
numbers. Finally, the Berry curvature is calculated and printed.

.. plot :: examples/haldane_bp.py
   :include-source:

.. _haldane_hwf-example:

Hybrid Wannier centers in Haldane model
---------------------------------------

Calculates Berry phases for the Haldane model and compares it to the
hybrid Wannier centers for a ribbon of the Haldane model.

.. plot :: examples/haldane_hwf.py
   :include-source:



.. _kane_mele-example:

Kane-Mele model using spinor features
-------------------------------------

Calculate the band structure of the Kane-Mele model, illustrating
the use of spinor features of the code.  Also compute the 1D
Wannier centers along :math:`x` as a function of :math:`k_y`,
illustrating the determination of the :math:`Z_2` invariant.

.. plot :: examples/kane_mele.py
   :include-source:


.. _visualize-example:

Visualization example
---------------------

Demonstrates visualization capabilities of the code.

.. plot :: examples/visualize.py
   :include-source:

   
.. _supercell:

Arbitrary graphene surface
--------------------------

Using supercell generator, one can calculate and plot the
surface band structure for an arbitrary surface orientation.

.. plot :: examples/supercell.py
   :include-source:


.. _buckled_layer:

Buckled layer
-------------

This is a very simple illustration of a slab geometry in which
the orbitals are specified in a 3D space, but the system is only
extensive in 2D, so that k-space is only 2D.

.. plot :: examples/buckled_layer.py
   :include-source:


.. _cubic_slab_hwf:

Hybrid Wannier functions on a cubic slab
----------------------------------------

This code illustrates a calculation of the Berry phases along x
of individual z-localized hybrid Wannier bands for a slab model
(finite in z but extended in x and y), using a *wf_array* structure
to simplify the calculation.

.. plot :: examples/cubic_slab_hwf.py
   :include-source:


.. _bn_ribbon_berry:

BN ribbon with change of nonperiodic vector
-------------------------------------------

This boron nitride ribbon calculation illustrates a case where
it is a good idea to change a nonperiodic lattice vector to be
normal to the periodic direction(s).  While the model is
physically the same before and after the change, as shown by the
identical band structures, the Berry phase is consistent with
the polarization in the extended direction only after the change.

.. plot :: examples/bn_ribbon_berry.py
   :include-source:


.. _w90_quick:

Quick Wannier90 example
-----------------------

To run the interface with Wannier90, you must first download the
following :download:`wannier90 output example
<misc/wannier90_example.tar.gz>` and unpack it with the following
command in unix command::

        tar -zxf wannier90_example.tar.gz

The example below will read the tight-binding model from the Wannier90
calculation, create a simplified model in which some small hopping
terms are ignored, and finally plot the interpolated band structure.

.. plot :: examples/w90_quick.py
   :include-source:


.. _w90_long:

Longer Wannier90 example
------------------------

This is a somewhat longer example showing how to use the interface to
Wannier90.  Unlike the example above, this one includes some
diagnostics as well.

To run the interface with Wannier90, first download
:download:`wannier90 output example <misc/wannier90_example.tar.gz>`
and unpack it with the following command in unix command::

        tar -zxf wannier90_example.tar.gz

Here is the source code of the example.

.. plot :: examples/w90.py
   :include-source:
