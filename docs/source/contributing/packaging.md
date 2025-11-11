# Packaging and Distribution

PythTB is distributed on [PyPI](https://pypi.org/project/pythtb/) and [conda-forge](https://anaconda.org/conda-forge/pythtb). This guide outlines how to maintain and publish new versions through both channels.

It covers:

- Dependency management (pyproject.toml)
- Versioning guidelines
- Pre-release checks
- Releasing on PyPI
- Releasing on conda-forge through the feedstock

## `pyproject.toml`

The project's [pyproject.toml](https://github.com/pythtb/pythtb/blob/dev/pyproject.toml) defines:

- Build backend configuration
- Package metadata (name, version, author, license, description)
- Package dependencies (with optional groups)

It is required for modern Python packaging and ensures tools such as pip, PyPI, and conda-forge can build the package from source. See [this guide for information on writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

## Versioning

PythTB follows [Semantic Versioning](https://semver.org/). Versions are in the format `X.Y.Z` where:

Type | Example | Reason
-----|---------|-------
PATCH | X.Y.Z → X.Y.Z+1 | Bugfixes only
MINOR | X.Y.Z → X.Y+1.0 | New features, no breaking changes
MAJOR | X.Y.Z → X+1.0.0 | Breaking API changes

## Pre-release checklist

### 1. Tests pass

```bash
pytest -n auto
```

### 2. Documentation builds cleanly (no warnings or errors)

```bash
sphinx-build docs/source docs/build/html
```

### 3. `CHANGELOG.md` updated
### 4. Release notes added 

- Add release notes in `docs/local/release/` and update toctree in `docs/local/release/release.rst`

## Releasing on PyPI

Once a new version is ready for release to PyPI, follow these steps:

### 1. Update the version

- Update the `__version__` variable and date in the header of `pythtb.py`.

### 2. Commit and tag the release:

```bash
git commit -m "Release X.Y.Z"
git tag vX.Y.Z
git push
git push --tags
```

### 3. Build distribution artifacts

```bash
pip install build
python -m build
```

Creates `dist/` folder with:
```bash
dist/
  pythtb-X.Y.Z.tar.gz
  pythtb-X.Y.Z-py3-none-any.whl
```

### 4. Upload to PyPI

First do a test upload to TestPyPI:

```bash
pip install twine
twine upload dist/* -r testpypi
```

Test the installation from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ pythtb==X.Y.Z
```

Then upload to the real PyPI:

```bash
twine upload dist/*
```

This uploads to PyPI. Verify the release appears at [https://pypi.org/project/pythtb/](https://pypi.org/project/pythtb/). 

| Note: This procedure could be automated with GitHub Actions.

## Releasing on conda-forge

PythTB is maintained on conda-forge via its __feedstock__ repository (See [conda-forge documentation](https://conda-forge.org/docs/maintainer/updating_pkgs/#updating-the-maintainer-list) for more details.)

The conda-forge bot usually:

- Detects the new PyPI release
- Opens a version-bump PR
- Updates SHA + metadata

If the bot does not trigger, update manually.

### 1. Open feedstock
Navigate to the feedstock at [https://github.com/conda-forge/pythtb-feedstock](https://github.com/conda-forge/pythtb-feedstock)

### 2. Fork + clone
Create your own fork of the feedstock, clone it to your computer.

### 3. Create branch and make changes

- Update version
- Update SHA256 from PyPI
- Reset `build:` number to 0 when shipping a new version
- Increment `build:` number when metadata changes without version bump

### 4. Rerender

Options:

- Comment on PR

```
@conda-forge-admin, please rerender
``` 

- Or run locally:

```bash
conda install -c conda-forge conda-smithy
conda-smithy rerender
```

### 5. Create PR
Push changes and create a pull request to the original upstream feedstock.

## Post-release steps
  
- Add folder in `docs/_static/versions/vX.Y.Z` for the new version. Add the `pythtb-X.Y.Z.tar.gz` file there. 
To get the `.tar.gz` file you can either download it from PyPI or create it locally by doing:

```bash
git checkout master
git pull
rm -rf dist
python setup.py sdist
```

- Update the `docs/source/install.md` file to include downloadable `.tar.gz`.
- Announce the new release on relevant channels .




