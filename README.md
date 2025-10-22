# PythTB

 [![Conda Version](https://anaconda.org/conda-forge/pythtb/badges/version.svg)](https://anaconda.org/conda-forge/pythtb/) [![Conda Downloads](https://anaconda.org/conda-forge/pythtb/badges/downloads.svg)](https://anaconda.org/conda-forge/pythtb/) [![readthedocs status](https://app.readthedocs.org/projects/pythtb/badge/?version=dev)](https://pythtb.readthedocs.io/en/latest/)

`PythTB` is a software package providing a Python implementation of the
tight-binding approximation. It can be used to construct and solve
tight-binding models of the electronic structure of systems of
arbitrary dimensionality (crystals, slabs, ribbons, clusters, etc.),
and is rich with features for computing Berry phases and related
properties. For more details, please see:

   http://www.physics.rutgers.edu/pythtb/

## Installation

PythTB can be installed from **Conda-Forge** or **PyPI**:

```bash
# via Conda
conda install -c conda-forge pythtb

# or via pip
pip install pythtb
```

PythTB ≥ v2.0.0 requires Python ≥ 3.11 and the following dependencies:
- numpy
- matplotlib

Optional dependencies:
- ipython, jupyter, plotly, tensorflow

For detailed installation instructions, editable/development setup, and troubleshooting, see:
- [Full Installation Guide](https://pythtb.readthedocs.io/en/latest/install.html)
- [Wiki](https://github.com/pythtb/pythtb/wiki/Editable-Conda-Install)
