# Installation Instructions

This is a step-by-step guide to installing the latest version of PythTB, particularly useful for developers who would like an editable version of PythTB.

## Anaconda

When developing a new version of PythTB, it is useful to maintain several different versions simultaneously without interaction between the two that could lead to dependency issues. 
This is where package managers help keep things neat and organized. It allows one to create virtual environments that have their own independent versions of Python and its packages. 

### Why `conda`?
For most purposes, `conda` is a great choice for a virtual environment manager. While the `venv` package manager is native to Python, it lacks many of the nice features that come with `conda`. For instance, `venv` only isolates Python packages, where `conda` isolates both packages _and_ Python itself.

Like `pip`, `conda` is also a package manager. It searches for packages from _conda channels_ rather than `pip`'s PyPI. Many packages live in both, but not always the same versions or builds.

### Installing `conda`

`conda` is installable through either Anaconda or Miniconda. Anaconda comes with many scientific and data science-related packages, and is several GB in size. This is often unnecessary 
for development purposes. Instead, it is recommended to install Miniconda, which includes only `conda` and `python`. This will still allow you to install packages and manage independent
virtual environments. The Miniconda installation instructions can be obtained from the [Anaconda website](https://www.anaconda.com/docs/getting-started/miniconda/install).

### Using `conda` to create a virtual environment
Once `conda` is installed, follow these steps to set up a dedicated development environment for PythTB:

```bash
# Create a new environment with a specific Python version
conda create -n pythtb-dev python=3.11

# Activate the environment
conda activate pythtb-dev

# The environment should now be visible: `$(pythtb-dev) USER`

# Install required dependencies
conda install numpy matplotlib

# (Optional) If using Jupyter notebooks:
conda install jupyterlab
```

### Installing PythTB in Editable Mode

After setting up the environment, clone the repository and install it in editable mode so that changes to the source code are reflected immediately without reinstalling:

```bash
# Clone the repository
git clone https://github.com/pythtb/pythtb.git
cd pythtb

# Install in editable/development mode
pip install -e .
```

The `-e` flag will install PythTB in editable mode, allowing you to make changes to the source code that will immediately propagate the package installation.

### Verifying Installation

To test that everything is set up correctly, try importing the package from a Python shell or Jupyter notebook
```python
import pythtb
print(pythtb.__version__)
```
You should see the version number of your local install. You can also do
```bash
conda list
```
to see all of the packages installed in your activated virtual environment.


### Troubleshooting

If you encounter issues such as `ModuleNotFoundEror` or conflicts between pip and conda package managers, try:
- Ensuring you're not installing packages globally. You need to activate the `conda` environment before installing packages.
  
  ```bash
  conda activate pythtb-dev
  ```
  
- Avoid installing packages outside the environment. You can check your active environment with
  
  ```bash
  conda info --envs
  ```
- Always try installing packages from the `conda-forge` channel first before using `pip`

  ```bash
  conda install package-name
  ```

  PythTB is currently only installable with `pip`.
  
If you've edited the source code but don't see the changes take effect:
- Make sure the package was installed with the `-e` (editable) flag:
  ```bash
  pip install -e .
  ```
- Restart the Python interpreter or Jupyter kernel to clear cached states.
- Check you're importing from the right location. The active Python should be from the `conda` virtual environment
  where you installed PythTB. You can check with
  ```bash
  which python
  ```

For building documentation, running tests and contributing to development, refer to [CONTRIBUTING.md](CONTRIBUTING.md)





