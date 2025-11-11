<p align="center">
 <img src="docs/source/_static/pythtb_logo.svg" width="300"/>
</p>

--------

[![Conda Version](https://anaconda.org/conda-forge/pythtb/badges/version.svg)](https://anaconda.org/conda-forge/pythtb/) 
[![Conda Downloads](https://anaconda.org/conda-forge/pythtb/badges/downloads.svg)](https://anaconda.org/conda-forge/pythtb/) 
[![PyPI Downloads](https://img.shields.io/pypi/dm/pythtb.svg?label=PyPI%20downloads)](
https://pypi.org/project/pythtb/)
[![readthedocs status](https://app.readthedocs.org/projects/pythtb/badge/?version=dev)](https://pythtb.readthedocs.io/en/dev/) 
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![Run examples on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythtb/pythtb/dev?urlpath=lab/tree/docs/source/examples/)

PythTB is a Python library for constructing and analyzing tight-binding models with an emphasis on clarity, flexibility, and modern band-topology workflows. With just a few lines of code, you can define lattices, add hoppings, diagonalize Hamiltonians, and extract geometric and topological observables across arbitrary parameter spaces.

PythTB supports systems ranging from textbook toy models to research-grade simulations. You can mix periodic and open directions to model crystals, slabs, ribbons, and molecules; sweep adiabatic parameters; and evaluate a wide range of electronic-structure quantities, including:

- Band structures and density of states
- Berry phases and Berry curvature
- Chern numbers, Wilson loops, and related invariants
- Quantum geometric tensors and local Chern markers
- Axion response in 3D
- Maximally localized Wannier functions
- Wannier-based tight-binding models generated through Wannier90

The package is designed to be readable and extensible, making it easy to explore new models, automate workflows, and integrate with first-principles calculations. Whether you are learning the basics of topological band theory or running high-throughput studies, PythTB gives you a transparent and lightweight framework to experiment, prototype, and investigate.

📘 [Documentation](https://pythtb.readthedocs.io/en/latest/)

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

PythTB ≥ 2.0.0 requires Python ≥ 3.12 and the core dependencies:
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
