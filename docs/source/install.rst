:tocdepth: 0

.. _installation:

Install
=======

PythTB >= v2.0.0 supports **Python >= 3.12** 
(see `SPEC-0 <https://scientific-python.org/specs/spec-0000/#support-window>`_).  
Versions up to v1.8.0 are compatible with **Python 2.7–3.10**, 
while v1.7.0 and below are limited to **Python 2.x**.  
Although other versions may work, they are not officially supported.

To check your Python version, run:

.. code-block:: bash

   python -V

If you do not have Python 3.12 or higher, 
see :ref:`Installing or upgrading Python <install-python>` below.
If you are unfamiliar with Python or are not sure whether Python and the
needed Python modules are installed on your system, see our
:doc:`python introduction <resources>`.

Dependencies
------------

PythTB requires the following Python packages to be installed:

*  `numpy <https://numpy.org/>`_
*  `matplotlib <https://matplotlib.org/stable/>`_

Optionally, you may also want to install the following packages 
to enhance your experience with PythTB (install with `pip install .[group]`):

- `[plotting]`: plotly (interactive 3D plots)
- `[speedup]`: tensorflow (GPU-assisted routines)
- `[notebooks]`: ipython ≥ 8.17, ipykernel, notebook, jupyter, jupyterlab (Jupyter support)
- `[docs]`: sphinx toolchain (build the documentation)
- `[tests]`: pytest
- `[dev]`: pytest, black, pre-commit

Quick Installation
------------------
You can install PythTB directly from either **PyPI** or **Conda-Forge**.

.. code-block:: bash

   # Using pip
   pip install pythtb --upgrade

   # Or using conda
   conda install -c conda-forge pythtb

To verify your installation:

.. code-block:: bash

   python -c "import pythtb; print(pythtb.__version__)"

If you encounter issues or missing dependencies, see :ref:`Troubleshooting <install-troubleshooting>`.


.. _install-alternative:

Installing from Source
----------------------

If you'd like to install PythTB from source, you can do so by cloning the
repository from `GitHub <https://github.com/pythtb/pythtb>`_. This is useful if you want 
to contribute to the project or if you want to use the latest development version. 

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/pythtb/pythtb.git
      cd pythtb

2. Install the package:

   .. code-block:: bash

      pip install .

This installs PythTB and its dependencies into your current Python environment.

If you want to install PythTB with optional dependencies, you can run

.. code-block:: bash

   pip install -e .[group] # replace [group] with optional groups as needed


Editable (Development) Installation
-----------------------------------

For contributors or developers who wish to modify the source code and see 
changes take effect immediately, install in **editable mode**:

1. Create a virtual environment using `conda` (recommended):

   .. code-block:: bash

      conda create -n pythtb-dev python=3.12
      conda activate pythtb-dev

2. Clone and install in editable mode by using the `-e` flag:

   .. code-block:: bash

      git clone https://github.com/pythtb/pythtb.git
      cd pythtb
      pip install -e ".[group]"  # replace [group] with optional groups as needed

3. Verify installation:

   .. code-block:: python

      import pythtb
      print(pythtb.__version__)

If you modify the source code, those changes will immediately take effect in your local environment.  
If you don't see updates reflected, restart the interpreter or Jupyter kernel.

For more details, see the 
`Developer Installation Wiki 
<https://github.com/pythtb/pythtb/wiki/Installation-Instructions-for-Developers>`_.


Older Versions
--------------

To install a specific version of PythTB:

.. code-block:: bash

   pip install pythtb==X.Y.Z

To list installed versions:

.. code-block:: bash

   pip show pythtb

Or from Python:

.. code-block:: python

   import pythtb
   print(pythtb.__version__)


.. _install-python:

Installing or Upgrading Python
------------------------------

If you don’t already have Python 3.12 or higher, follow one of the options below.

**Anaconda / Miniconda (Recommended)**

If you prefer to manage environments separately, install Python via Miniconda:

.. code-block:: bash

   conda create -n pythtb-env python=3.12

`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ provides 
a lightweight version of Anaconda, ideal for managing clean environments 
for scientific packages like PythTB.

**macOS and Linux**

Use your system's package manager:

.. code-block:: bash

   # Ubuntu / Debian
   sudo apt-get install python3

   # macOS (via Homebrew)
   brew install python

Alternatively, download the latest release from the
`official Python website <https://www.python.org/downloads/>`_.

**Windows**

Download and run the official installer from
`python.org <https://www.python.org/downloads/>`_.  
Make sure to check *“Add Python to PATH”* during installation.

.. _install-troubleshooting:

Troubleshooting
---------------

Common issues and fixes:

* **`ModuleNotFoundError` after installation:**  
  Make sure you are installing inside the correct environment and have activated it.

  .. code-block:: bash

     conda activate pythtb-env

* **Conflicts between pip and conda:**  
  Avoid installing global packages. Keep each environment isolated.

* **Editable install not updating:**  
  Ensure you used `pip install -e .` and restart your interpreter.

  .. code-block:: bash

     which python
     conda list | grep pythtb

If problems persist, open an issue on the
`GitHub repository <https://github.com/pythtb/pythtb/issues>`_.

Version list
---------------
If you would like to install a specific version of PythTB directly from the
list of available versions, you can do so below.

.. include:: release.rst
