<p align="center">
 <img src="docs/source/_static/pythtb_logo2_dark.svg" width="300"/>
</p>

--------

[![Conda Version](https://anaconda.org/conda-forge/pythtb/badges/version.svg)](https://anaconda.org/conda-forge/pythtb/) 
[![Conda Downloads](https://anaconda.org/conda-forge/pythtb/badges/downloads.svg)](https://anaconda.org/conda-forge/pythtb/) 
[![readthedocs status](https://app.readthedocs.org/projects/pythtb/badge/?version=dev)](https://pythtb.readthedocs.io/en/dev/) 
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![Run examples on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythtb/pythtb/dev?urlpath=lab/tree/docs/source/examples/)

`PythTB` is a software package providing a Python implementation of the tight-binding approximation. It can be used to construct and solve tight-binding models of the electronic structure of systems of arbitrary dimensionality (crystals, slabs, ribbons, clusters, etc.), and is rich with features for computing Berry phases and related properties. For more details, please see the [documentation](https://pythtb.readthedocs.io/en/latest/).

## Installation

PythTB can be installed from **Conda-Forge** (PythTB ≥ 1.8.0) or **PyPI**:

```bash
# via Conda for pythtb >= v1.8.0
conda install -c conda-forge pythtb

# or via pip
pip install pythtb
```

Or install from source (editable) after cloning the repository:

```bash
git clone https://github.com/pythtb/pythtb.git
cd pythtb
pip install -e .
```

PythTB ≥ 2.0.0 requires Python ≥ 3.11 and the core dependencies:
- numpy ≥ 2.0
- matplotlib ≥ 3.9

Optional extras (install with `pip install .[group]`):

- `[plotting]`: plotly (interactive 3D plots)
- `[speedup]`: tensorflow (GPU-assisted routines)
- `[notebooks]`: ipython ≥ 8.17, ipykernel, notebook, jupyter, jupyterlab (Jupyter support)
- `[docs]`: sphinx toolchain (build the documentation)
- `[tests]`: pytest
- `[dev]`: pytest, black, pre-commit

For detailed installation instructions, editable/development setup, and troubleshooting, see:
- [Full Installation Guide](https://pythtb.readthedocs.io/en/latest/install.html)
- [Wiki](https://github.com/pythtb/pythtb/wiki/Installation-Instructions-for-Developers)
