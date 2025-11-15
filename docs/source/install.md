(installation)=
# Install

PythTB can be installed in a variety of ways depending on your workflow. This page covers system requirements, quick installation methods, and optional
developer setups.


## Dependencies

PythTB follows the [SPEC-0](https://scientific-python.org/specs/spec-0000/#support-window) support window for scientific Python packages.

**Required dependencies**:

- [Python](https://www.python.org/) ≥ 3.12
- [NumPy](https://numpy.org/) ≥ 2.0
- [Matplotlib](https://matplotlib.org/stable/) ≥ 3.9

**Optional extras** (install via `pip install .[$GROUP]`):

- `[plotting]`: plotly (interactive 3D plots)
- `[speedup]`: tensorflow
- `[notebooks]`: ipython ≥ 8.17, ipykernel, notebook ≥ 7.0, jupyter, jupyterlab
- `[docs]`: sphinx toolchain for building documentation
- `[tests]`: pytest
- `[dev]`: pytest, black, ruff, pre-commit, nbstripout

Check your Python version by running:

```bash
python -V
```

If you need to upgrade Python, see [Installing or Upgrading Python](#install-python) below. If you are new to Python see our [resources](resources).

## Quick Installation

Install PythTB from **PyPI** or **conda-forge**.

```bash
# Using pip
pip install pythtb --upgrade

# Using conda
conda install -c conda-forge pythtb
```

Verify installation:

```bash
python -c "import pythtb; print(pythtb.__version__)"
```

If you encounter issues, see [Troubleshooting](install-troubleshooting).

(install-source)=
## Installing from Source

Installing from source is recommended for contributors or for using the latest development version.

1. Create a virtual environment using `conda` (recommended):

```bash
conda create -n pythtb-dev python=3.12
conda activate pythtb-dev
```

2. Clone the repository:

```bash
git clone https://github.com/pythtb/pythtb.git
cd pythtb
```

3. Install the package along with any optional dependencies you need:

```bash
pip install .  # or pip install .[$GROUP] for optional dependencies
```

For an editable install, 

```bash
pip install -e .
```

Editable installs allow immediate reflection of code changes. If you modify the source code, those changes will immediately take effect in your local environment. If you don't see updates reflected, restart the interpreter or Jupyter kernel.

For more details about setting up your development environment, see the [Developer Installation Wiki](https://github.com/pythtb/pythtb/wiki/Installation-Instructions-for-Developers).

## Older Versions

The latest stable release is always available on PyPI and conda-forge.


Install a specific version using conda:

```bash
conda install -c conda-forge pythtb=X.Y.Z
```

:::{note}
Currently only PythTB >= 1.8.0 are available on Conda-Forge. 
:::

Using pip, you can install all older versions from PyPI:

```bash
pip install pythtb==X.Y.Z
```

Check the installed version with:

```bash
pip show pythtb
```

The source code of all previous releases of PythTB can be downloaded below in [Version List](install-index). 

## Verify installation

After installation, verify by running:

```python
import pythtb
print(pythtb.__version__)
```

This should print the installed version of PythTB without errors.

(install-python)=
## Installing or Upgrading Python

If you don’t already have Python 3.12 or higher, follow one of the options below.

### Anaconda / Miniconda (Recommended)

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) provides 
a lightweight version of Anaconda, ideal for managing clean environments 
for scientific packages like PythTB.

If you prefer to manage environments separately, install Python via Miniconda:

```bash
conda create -n pythtb-env python=3.12
conda activate pythtb-env
```

### macOS and Linux

Use your system's package manager:

```bash
# Ubuntu / Debian
sudo apt-get install python3

# macOS (via Homebrew)
brew install python
```

Alternatively, download the latest release from the
[official Python website](https://www.python.org/downloads/).

### Windows

Download and run the official installer from [python.org](https://www.python.org/downloads/). Make sure to check *“Add Python to PATH”* during installation.


(install-troubleshooting)=
## Troubleshooting

Common issues and fixes:

- `ModuleNotFoundError`

    Make sure you are using the correct Python environment where PythTB is installed. If using `conda`, activate the environment:

    ```bash
    conda activate your-env-name
    ```

    Check installation with:

    ```bash
    conda list | grep pythtb
    ```

- Conflicting installations

    If you have multiple installations of PythTB, uninstall them first:
    ```bash
    pip uninstall pythtb
    conda remove pythtb
    ```
    Then reinstall using one method (pip or conda).

- Conflicts between pip and conda

    Avoid mixing `pip` and `conda` installations in the same environment. Prefer using one package manager consistently. The exception is using `pip` to install packages not available via `conda` or when installing in editable mode.

- Editable mode issues

    If you installed PythTB in editable mode and changes are not reflected, restart your Python interpreter or Jupyter kernel. Make sure you installed with the `-e` flag:

    ```bash
    pip install -e . 
    ```

If problems persist, open an issue on the [GitHub repository](https://github.com/pythtb/pythtb/issues).

(install-index)=
## Version List

See [changelog](CHANGELOG) for a complete list of changes.

### Version 2.0.0 (current)
11 November 2025: [pythtb-2.0.0.tar.gz](_static/versions/v2.0.0/pythtb-2.0.0.tar.gz)

### Version 1.8.0

20 September 2022: [pythtb-1.8.0.tar.gz](_static/versions/v1.8.0/pythtb-1.8.0.tar.gz)

### Version 1.7.2

1 August 2017: [pythtb-1.7.2.tar.gz](_static/versions/v1.7.2/pythtb-1.7.2.tar.gz)

### Version 1.7.1

22 December 2016: [pythtb-1.7.1.tar.gz](_static/versions/v1.7.1/pythtb-1.7.1.tar.gz)

### Version 1.7.0

7 June 2016: [pythtb-1.7.0.tar.gz](_static/versions/v1.7.0/pythtb-1.7.0.tar.gz)

### Version 1.6.2

25 February 2013: [pythtb-1.6.2.tar.gz](_static/versions/v1.6.2/pythtb-1.6.2.tar.gz)

### Version 1.6.1

15 November 2012: [pythtb-1.6.1.tar.gz](_static/versions/v1.6.1/pythtb-1.6.1.tar.gz)

### Version 1.5

4 June 2012: [pytb-1.5.tar.gz](_static/versions/v1.5/pytb-1.5.tar.gz)
