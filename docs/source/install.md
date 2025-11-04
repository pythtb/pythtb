(installation)=
# Install

**PythTB ≥ 2.0.0** supports **Python ≥ 3.12** (see [SPEC-0](https://scientific-python.org/specs/spec-0000/#support-window)).  
Versions up to v1.8.0 are compatible with **Python 2.7–3.10**, while v1.7.0 and below are limited to **Python 2.x**. Although other versions may work, they are not officially supported.

To check your Python version, run:

```bash
python -V
```

If you do not have Python 3.12 or higher, see [Installing or Upgrading Python](#install-python) below. If you are unfamiliar with Python see our [resources](resources).

## Dependencies

PythTB requires:

- [NumPy](https://numpy.org/) ≥ 2.0
- [Matplotlib](https://matplotlib.org/stable/) ≥ 3.9

Optional extras (install via `pip install .[group]`):

- `[plotting]`: plotly (interactive 3D plots)
- `[speedup]`: tensorflow
- `[notebooks]`: ipython ≥ 8.17, ipykernel, notebook, jupyter, jupyterlab
- `[docs]`: sphinx toolchain for building documentation
- `[tests]`: pytest
- `[dev]`: pytest, black, pre-commit

## Quick Installation

You can install PythTB directly from either **PyPI** or **Conda-Forge**.

```bash
# Using pip
pip install pythtb --upgrade

# Or using conda
conda install -c conda-forge pythtb
```

Verify installation:

```bash
python -c "import pythtb; print(pythtb.__version__)"
```
If you encounter issues or missing dependencies, see [Troubleshooting](install-troubleshooting).

(install-source)=
## Installing from Source

If you'd like to install PythTB from source, you can do so by cloning the repository from [GitHub](https://github.com/pythtb/pythtb). This is useful if you want to contribute to the project or if you want to use the latest development version. 

1. Clone the repository:
```bash
git clone https://github.com/pythtb/pythtb.git
cd pythtb
```

2. Install the package:

```bash
pip install .
```

This installs PythTB and its dependencies into your current Python environment. If you want to install PythTB with optional dependencies, you can run

```bash
pip install .[group] # replace [group] with optional groups as needed
```

### Editable (Development) Installation
For contributors or developers who wish to modify the source code and see 
changes take effect immediately, install in **editable mode**:

1. Create a virtual environment using `conda` (recommended):

```bash
conda create -n pythtb-dev python=3.12
conda activate pythtb-dev
```

2. Clone and install in editable mode by using the `-e` flag:

```bash
git clone https://github.com/pythtb/pythtb.git
cd pythtb
pip install -e .[group]  # replace [group] with optional groups as needed
```

3. Verify installation:

```python
import pythtb
print(pythtb.__version__)
```
If you modify the source code, those changes will immediately take effect in your local environment. If you don't see updates reflected, restart the interpreter or Jupyter kernel.

For more details, see the [Developer Installation Wiki](https://github.com/pythtb/pythtb/wiki/Installation-Instructions-for-Developers).


## Older Versions

PyPI and Conda-Forge always host the latest stable release of PythTB.
Conda-Forge allows installing specific versions using:

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

To list installed versions:

```bash
pip show pythtb
```

Or in Python:

```python
import pythtb
print(pythtb.__version__)
```

All previous releases of PythTB can be found below in [Version List](install-index) and the source code can be downloaded from the links provided there. 

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
