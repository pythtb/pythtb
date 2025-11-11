.. _installation:

Installation
============

To install the latest version of PythTB simply type in terminal::

    pip install pythtb --upgrade

If you don't have root access try installing PythTB into your home 
folder by executing::

  pip install pythtb --upgrade --user

You should now be ready to download and run any of the  :doc:`examples <examples>`.
If you wish, you could also download all example files as a single tar file

   :download:`pythtb-examples.tar.gz <misc/pythtb-examples.tar.gz>`

to untar this file use the following unix command::

        tar -zxf pythtb-examples.tar.gz

If you need more assistance with the installation process, please 
continue reading.

More detailed instructions
--------------------------

PythTB is compatible with Python 2.7 and 3.x (Python 2.6 and
below are not recommended).  To check which version of Python is
currently installed on your system, just type::

   python -V

If your Python is not at version 2.7 or 3.x, see
:ref:`Installing or upgrading Python<install-python>`.

These instructions also assume that
`pip <https://pip.pypa.io/en/stable/>`_, the standard package
manager for Python, is installed on your system.  If not, we recommend
that you
`install it <https://pip.pypa.io/en/stable/installing/>`_.
Otherwise, see instructions for
:ref:`Installing without pip <install-alternative>`.

Once you have python and pip you can install PythTB simply by
typing the following into your terminal::

    pip install pythtb --upgrade

Note that pip command should automatically install/upgrade the
following python packages if they were not previously installed
and upgraded to the latest version:

* numpy
* matplotlib

To bypass the upgrade of these packages::

    pip install --upgrade --upgrade-strategy only-if-needed pythtb

You should now be ready to run any of the  :doc:`examples <examples>`.

Older versions of PythTB
------------------------

If for some reason you wish to install a specific older version of PythTB you
can type the following into terminal (replace 1.x.x with version number you
wish to install)::

    pip install pythtb==1.x.x

To check which version of PythTB you have installed, type::

    pip show pythtb

or print the value of the version number from within a PythTB program::

    print(pythtb.__version__)


.. _install-alternative:

Alternative installation, without 'pip'
---------------------------------------

If for any reason you can't install PythTB using 'pip' you can
follow these instructions.

PythTB is compatible with Python 2.7 and 3.x (Python 2.6 and
below are not recommended).  To check which version of Python is
currently installed on your system, just type::

   python -V

If your Python is not at version 2.7 or 3.x, see
:ref:`Installing or upgrading Python<install-python>`.

In addition, you will need to have numpy and matplotlib installed.
On a Unix/Linux/Mac system, you may be able to install these using your
Package Manager.  Otherwise, follow the installation instructions
at the numpy and matplotlib official websites:

*  `numpy official website <http://numpy.scipy.org>`_

*  `matplotlib official website <http://www.matplotlib.org>`_

or install `Anaconda <https://www.continuum.io/downloads>`_ version of python.

To install PythTB without 'pip' first download either of the following 
two archived forms

* :download:`pythtb-1.8.0.tar.gz <misc/pythtb-1.8.0.tar.gz>`
* :download:`pythtb-1.8.0.zip <misc/pythtb-1.8.0.zip>`

and unpack it with one of these commands::

   tar -zxf pythtb-1.8.0.tar.gz
   unzip pythtb-1.8.0.zip

Then move into the working directory and install it::

   cd pythtb-1.8.0
   python setup.py install

(You may have to `sudo` the second command.)  To check that the
installation was successful, type::

   python

and then execute the following command in python interpreter::

   import pythtb

If you do not see any error message, the installation was
successful. Now you can try executing any of the :doc:`example scripts <examples>`.

.. _install-python:

Installing or upgrading Python
------------------------------

If you do not have Python installed, or it is not at Version 2.7 or
higher, follow these instructions.

In Unix/Linux or Mac, use the package manager provided by your system
to download and install or upgrade Python and
any needed modules.  It is recommended to install ‘python2.7’
(or python3.5) and related packages such as ‘python2.7-doc’.
If you do not have a package manager, you can try going to the
official `Python Download Page <http://www.python.org/download/>`_ and
follow instructions there to download and install it. Or, try `Anaconda
<https://www.continuum.io/downloads>`_, which provides SciPy,
NumPy, and Matplotlib already in the distribution.

For Windows, follow the instructions at the
`official Python site <https://www.python.org/downloads/windows/>`_ or
try `Anaconda <https://www.continuum.io/downloads>`_.

Additional software
-------------------

You may wish to try installing `ipython <http://www.ipython.org>`_; it
provides a more user-friendly interactive interface than ‘python’ does.


Release notes and version list
------------------------------

We recommend you always use latest available version of PythTB.
Note that versions up to 1.7.0 are incompatible with Python3.
Versions 1.7.1 and above are compatible with both Python 2.7
and 3.x.

However, if you need
to look up an old version of the code, you can find it below.

.. include:: ../local/release/release.rst
