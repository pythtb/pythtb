:tocdepth: 0

.. _installation:

Install
=======

PythTB >= v2.0.0 is supported for Python >= 3.10. Versions up to 1.7.0 are
only supported on Python < 3.0, while version 1.8.0 is compatible with 
Python 2.7 - 3.10 (Python 2.6 and below are not recommended).
While other versions of Python may work, they are not guaranteed to be compatible.

To check which version of Python is
currently installed on your system, just type

.. code-block:: bash

   python -V

If you don't have Python 3.10 or higher, you can follow the instructions in
:ref:`Installing or upgrading Python <install-python>` to install or upgrade.


Dependencies
------------

PythTB requires the following Python packages to be installed:

*  `numpy <https://numpy.org/>`_
*  `matplotlib <https://matplotlib.org/stable/>`_

Optionally, you may also want to install the following packages
to enhance your experience with PythTB:

*  `ipython <http://www.ipython.org>`_ - a powerful interactive shell.
*  `jupyter <https://jupyter.org/>`_ - for running Jupyter notebooks.
*  `plotly <https://plotly.com/python/>`_ - for interactive plotting capabilities with 3D models.
*  `tensorflow <https://www.tensorflow.org/>`_ - for faster computations.

Installing from PyPI with ``pip``
----------------------------------

These instructions also assume that
`pip <https://pip.pypa.io/en/stable/>`_, the standard package
manager for Python, is installed on your system. If not, we recommend
that you `install it <https://pip.pypa.io/en/stable/installing/>`_.

To install the latest version of PythTB with pip simply type in terminal

.. code-block:: bash

   pip install pythtb --upgrade

To install an older version of PythTB, you can specify the version number like this:

.. code-block:: bash

   pip install pythtb==X.Y.Z

To install all the optional dependencies, you can run

.. code-block:: bash

   pip install .[optional] 
   
.. tip::

   If you don't have root access try installing PythTB into your home 
   folder by executing

   .. code-block:: bash

      pip install pythtb --upgrade --user


.. note::
   The ``pip install`` command should automatically install/upgrade the
   following python packages if they were not previously installed
   and upgraded to the latest version. To bypass the upgrade of these 
   packages

   .. code-block:: bash

      pip install --upgrade --upgrade-strategy only-if-needed pythtb

After the installation is complete, you can check that PythTB
is installed correctly by typing in terminal

.. code-block:: bash

   python -c "import pythtb; print(pythtb.__version__)"

If you need more assistance with the installation process, please 
continue reading.


.. _install-alternative:

Installing from source
-------------------------

If you'd like to install PythTB from source, you can do so by cloning the
repository from `GitHub <https://github.com/pythtb/pythtb>`_. This is useful if you want 
to contribute to the project or if you want to use the latest development version. 

First, make sure you have Python 3.10 or higher installed on your system.
You can check your Python version by typing in terminal
 
.. code-block:: bash

   python -V

If you don't have Python 3.10 or higher, you can follow the instructions in
:ref:`Installing or upgrading Python <install-python>` to install or upgrade.

Next, clone the PythTB repository from GitHub by typing in terminal

.. code-block:: bash

   git clone https://github.com/pythtb/pythtb.git

This will create a directory called `pythtb` in your current working directory.
Next, navigate to the `pythtb` directory by typing

.. code-block:: bash

   cd pythtb

Now, you can install PythTB by running the following command in terminal

.. code-block:: bash

   pip install .

This will install PythTB and its dependencies. 

If you want to install PythTB in editable mode 
(so that changes you make to the source code are immediately
reflected in your installation), you can run

.. code-block:: bash

   pip install -e .

If you want to install PythTB with optional dependencies, you can run

.. code-block:: bash

   pip install .[optional]

This will install PythTB with optional dependencies such as `ipython` and `plotly`. 

Older versions of PythTB
------------------------

If for some reason you wish to install an older version of PythTB you
can type the following into terminal (replace X.Y.Z with version number you
wish to install)

.. code-block:: bash

   pip install pythtb==X.Y.Z

To check which version of PythTB you have installed, type

.. code-block:: bash

   pip show pythtb

or print the value of the version number from within a Python environment

.. code-block:: python

   import pythtb
   print(pythtb.__version__)

.. _install-python:

Installing or upgrading Python
------------------------------
To use PythTB, you need Python 3.10 or higher. 
If you don't have it installed, or your version is outdated, 
follow the steps below

macOS and Linux
^^^^^^^^^^^^^^^^^^^^^
The recommended way to install or upgrade Python is via your system's package manager. 
For example, on Ubuntu or Debian-based systems, you can run

.. code-block:: bash

   sudo apt-get install python3

For macOS, you can use Homebrew

.. code-block:: bash

   brew install python

Alternatively, for any Unix-like system, you can download the latest Python installer
from the official `Python Download Page <http://www.python.org/download/>`_.

Windows
^^^^^^^^^^^^^^^
On Windows, the recommended way to install or upgrade Python is to download the
official installer from the `Python Download Page <http://www.python.org/download/>`_.
Be sure to select the option to add Python to your PATH during the installation process.

Anaconda and Miniconda
------------------------
PythTB is currently not available on the Anaconda Cloud, but you can still install it using pip
within an Anaconda environment. 

Anaconda is a popular distribution of Python that includes many scientific computing packages.
It is recommended to use Anaconda if you are working with scientific computing or data science, 
as it simplifies package management and deployment. You can install Anaconda from the
`Anaconda Download Page <https://www.anaconda.com/products/distribution>`_.

A minimal installation of Anaconda can be done using the
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ installer.
This will allow you to create isolated environments and manage packages easily without
installing the full Anaconda distribution.

To do this, first create a new environment with Python 3.10 or higher

.. code-block:: bash

   conda create -n pythtb_env python=3.10

Then activate the environment

.. code-block:: bash

   conda activate pythtb_env

After activating the environment, you can install PythTB using pip as you would normally

.. code-block:: bash

   pip install pythtb --upgrade

The previous steps in section `Installing from source <install-alternative>`_ will also work
as expected within the activated conda environment.

Version list
---------------
If you would like to install a specific version of PythTB directly from the
list of available versions, you can do so below.

.. include:: ../local/release/release.rst
