<p align="center">
 <img src="docs/source/_static/pythtb_logo.svg" width="300"/>
</p>

--------

[![Conda Version](https://anaconda.org/conda-forge/pythtb/badges/version.svg)](https://anaconda.org/conda-forge/pythtb/) 
[![Conda Downloads](https://anaconda.org/conda-forge/pythtb/badges/downloads.svg)](https://anaconda.org/conda-forge/pythtb/) 
[![readthedocs status](https://app.readthedocs.org/projects/pythtb/badge/?version=dev)](https://pythtb.readthedocs.io/en/dev/) 
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![Run examples on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythtb/pythtb/dev?urlpath=lab/tree/docs/source/examples/)

PythTB is a Python toolkit for constructing and analyzing tight-binding models. It provides a flexible interface for exploring electronic structure and band topology in both simple and research-scale systems. With PythTB, you can:

- Build models for crystals, slabs, ribbons, and molecules - any combination of periodic and open directions.
- Sweep adiabatic parameters and evaluate band structure, quantum geometric tensors, local Chern markers, and axion angles.
- Compute Berry phases, Berry curvature, Chern numbers, Wilson loops, and other band-topology diagnostics on structured meshes.
- Interface with Wannier90 to construct tight-binding models from first-principles calculations.

Whether you are prototyping a textbook model or conducting research, PythTB is designed to remain readable, reproducible, and easy to extend.
For more details, please refer to the [documentation](https://pythtb.readthedocs.io/en/latest/).

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
