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

1. Tests pass

```bash
pytest -n auto
```

2. Documentation builds cleanly (no warnings or errors)

```bash
sphinx-build docs/source docs/build/html
```

3. Version number updated in `pythtb/__init__.py`
4. `CHANGELOG.md` updated
5. Release notes added 
- Add release notes in `docs/source/release/` and update toctree in `docs/source/release.rst`

Once all of the changes are in place, we can proceed to release.

## Releasing on PyPI

Our releases are driven by GitHub Actions plus PyPI Trusted Publishing. No long-lived PyPI token exists in the repository (or on contributor machines); instead the workflow requests a short-lived token via OIDC and targets the protected `pypi-prod` environment. Because only release managers can push tags to the upstream repo _and_ approve that environment, accidental or manual uploads are effectively prevented.

1. Commit the version bump, changelog, and release notes:
   ```bash
   git commit -m "Release X.Y.Z"
   ```
2. Create an annotated tag that matches either the stable (`vX.Y.Z`) or pre-release (`vX.Y.ZrcN`) pattern:
   ```bash
   git tag vX.Y.Z
   ```
3. Push the branch and tag:
   ```bash
   git push
   git push --tags
   ```
4. Monitor the `Publish to PyPI (Trusted Publishing)` workflow for stable tags or `Publish to TestPyPI` for RC tags. GitHub will pause the job until an approver for the `pypi-prod` environment approves the deployment step. Once approved, `pypa/gh-action-pypi-publish` builds the sdist/wheel, runs `twine check`, and uploads them using Trusted Publishing.

Additional notes:

- RC tags (`vX.Y.ZrcN`) land on TestPyPI via `.github/workflows/testpypi-publish.yml`. Stable tags go straight to PyPI via `.github/workflows/pypi-publish.yml`.
- Both workflows can be re-run via *Re-run failed jobs* or `workflow_dispatch` in GitHub's UI; re-running reuses the same tag so nothing new needs to be pushed.
- Because we do not mint PyPI API tokens for local use, nobody can `twine upload` manually unless the PyPI owners deliberately create and share a token for an emergency.

### Manual fallback (only if automation cannot publish)

In the rare case that PyPI is down or Trusted Publishing is temporarily unavailable, coordinate with the PyPI project owners to obtain a temporary API token and then follow the commands below. Clean up any locally stored credentials afterwards.

#### Build distribution artifacts

```bash
pip install build
python -m build
```

This creates `dist/` with:

```
dist/
  pythtb-X.Y.Z.tar.gz
  pythtb-X.Y.Z-py3-none-any.whl
```

#### Upload to PyPI

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

Verify the release appears at [https://pypi.org/project/pythtb/](https://pypi.org/project/pythtb/). Remove any temporary API tokens when finished.


## Releasing on conda-forge

PythTB is maintained on conda-forge via its __feedstock__ repository (See [conda-forge documentation](https://conda-forge.org/docs/maintainer/updating_pkgs/#updating-the-maintainer-list) for more details.)

The conda-forge bot usually:

- Detects the new PyPI release
- Opens a version-bump PR
- Updates SHA + metadata

If the bot does not trigger, update manually.

1. Open feedstock
- Navigate to the feedstock at [https://github.com/conda-forge/pythtb-feedstock](https://github.com/conda-forge/pythtb-feedstock)

2. Fork + clone
- Create your own fork of the feedstock, clone it to your computer.

3. Create branch and make changes

- Update version
- Update SHA256 from PyPI
- Reset `build:` number to 0 when shipping a new version
- Increment `build:` number when metadata changes without version bump

4. Create PR
- Push changes and create a pull request to the original upstream feedstock.

5. Rerender

  - Comment on PR

  ```
  @conda-forge-admin, please rerender
  ``` 

  - Or run locally:

  ```bash
  conda install -c conda-forge conda-smithy
  conda-smithy rerender
  ```

## Post-release steps
  
 Add folder in `docs/_static/versions/vX.Y.Z` for the new version. Add the `pythtb-X.Y.Z.tar.gz` file there. 
To get the `.tar.gz` file you can either download it from PyPI or create it locally by doing:

1. Make a release on GitHub
- Go to the [Releases page](https://github.com/pythtb/pythtb/releases) and click "Draft a new release"
- Select the tag version
- Copy the release notes from `docs/source/release/X.Y.Z-notes.md` into the GitHub release description
- Publish the release

2. Download the source code `.tar.gz` from the GitHub release page.
- Update the `docs/source/install.md` file to include downloadable `.tar.gz`.

3. Update `docs/source/index.md` to highlight the new release.
4. Update Zenodo record with new version.
5. Announce the new release on relevant channels.



