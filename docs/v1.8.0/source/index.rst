.. meta::
   :keywords: PythTB, PyTB, python, tight binding, Wannier, Berry,
              topological insulator, Chern, Haldane, Kane-Mele, Z2, graphene,
              band structure, wavefunction, bloch, periodic insulator,
	      wannier90, wannier function, density functional theory,
	      DFT, first-principles

Python Tight Binding (PythTB)
=============================

PythTB is a software package providing a Python implementation
of the tight-binding approximation. It can be used to construct
and solve tight-binding models of the electronic structure of
systems of arbitrary dimensionality (crystals, slabs, ribbons,
clusters, etc.), and is rich with features for computing Berry
phases and related properties.

.. toctree::
   :maxdepth: 1

   about
   install
   examples
   formalism
   usage
   resources
   citation

Quick installation
==================

Type in terminal::

    pip install pythtb --upgrade

or if you need more assistance follow these :doc:`instructions <install>`.

Quick example
=============

This is a simple example showing how to define graphene tight-binding
model with first neighbour hopping only. Below is the source code and
plot of the resulting band structure. Here you can find :doc:`more
examples <examples>`.


.. raw:: html

    <table style="margin-left: 20px;" align="center" border="0"><tbody><tr>
    <td width="50%">

.. literalinclude:: simple_fig/simple_fig.py

.. raw:: html

    </td>
    <td width="50%">

.. image:: simple_fig/band.png

.. raw:: html

    </td></tr>
    </tbody></table>

