<p align="center">
 <img src="https://raw.githubusercontent.com/pythtb/pythtb/main/docs/source/_static/pythtb_logo.svg" width="300"/>
</p>

--------

[![Conda Version](https://anaconda.org/conda-forge/pythtb/badges/version.svg)](https://anaconda.org/conda-forge/pythtb/) 
[![Conda Downloads](https://anaconda.org/conda-forge/pythtb/badges/downloads.svg)](https://anaconda.org/conda-forge/pythtb/) 
[![PyPI Downloads](https://img.shields.io/pypi/dm/pythtb.svg?label=PyPI%20downloads)](
https://pypi.org/project/pythtb/)
[![readthedocs status](https://app.readthedocs.org/projects/pythtb/badge/?version=dev)](https://pythtb.readthedocs.io/en/dev/) 
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![Run examples on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythtb/pythtb/dev?urlpath=lab/tree/docs/source/examples/)

PythTB is a Python library for constructing and analyzing tight-binding models, built for modern topological band theory applications. It provides a streamlined path from model specification to physical interpretation, making it useful for both learning electronic structure and conducting research-level studies. With only a few lines of code, you can define lattice models, build tight-binding Hamiltonians, and compute electronic properties.

PythTB provides tools for:
- Band structures and density of states
- Berry phases and Berry curvature
- Chern numbers, Wilson loops, and related invariants
- Quantum geometric tensors and local Chern markers
- Chern-Simons axion angle
- Maximally localized Wannier functions
- Wannier-based tight-binding models generated through Wannier90

📘 [Documentation](https://pythtb.readthedocs.io/en/latest/)

## Installation

PythTB is available through conda-forge (recommended) and PyPI.

```bash
# Conda (pythtb >= 1.8.0)
conda install -c conda-forge pythtb

# pip
pip install pythtb
```

To install from source in editable mode:

```bash
git clone https://github.com/pythtb/pythtb.git
cd pythtb
pip install -e .
```

PythTB ≥ 2.0.0 requires Python ≥ 3.12 and the core dependencies:
- numpy ≥ 2.0
- matplotlib ≥ 3.9

Optional extras can be installed via `pip install .[group]`:

- `[plotting]`: Plotly for interactive visualization
- `[speedup]`: TensorFlow for GPU-accelerated routines
- `[notebooks]`: Jupyter support (IPython ≥ 8.17, ipykernel, notebook, jupyter, jupyterlab)
- `[docs]`: Sphinx toolchain for documentation
- `[tests]`: pytest
- `[dev]`: developer tools (pytest, ruff, pre-commit)

For more detailed instructions, see:
- [Full Installation Guide](https://pythtb.readthedocs.io/en/latest/install.html)
- [Wiki](https://github.com/pythtb/pythtb/wiki/Installation-Instructions-for-Developers)
