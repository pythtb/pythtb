# Packaging and Distribution

PythTB is currently available from PyPI and conda-forge. This guide explains how to maintain the package across these channels, including:

- Managing dependencies via pyproject.toml
- Publishing releases on PyPI
- Updating the conda-forge feedstock

## The `pyproject.toml`

The [pyproject.toml](https://github.com/pythtb/pythtb/blob/dev/pyproject.toml) declares the project’s metadata and build requirements.
It is the canonical place to specify:

- Package dependencies
- Optional dependency groups
- Build backend configuration
- Versioning and metadata

It is required for modern Python packaging workflows and provides a standardized interface for tools like PyPI, pip, and conda-forge to build the package from source. See [this guide for information on writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

## When to bump version?

Follow semantic versioning

```
PATCH (X.Y.Z → X.Y.Z+1)
Bugfixes only
MINOR (X.Y.Z → X.Y+1.0)
New features, no breaking changes
MAJOR (X.Y.Z → X+1.0.0)
Breaking API changes
```

Inclusion criteria for a release

Before tagging:
- [ ] Tests pass locally
- [ ] Docs build locally
- [ ] No unresolved deprecation notes
- [ ] Changelog updated
- [ ] Version incremented in pyproject.toml
- [ ] Update docs/local/release/release.rst. 
- [ ] Add a folder in docs/local/release/ver_ABC for the old version of the package,
put in the `.tar.gz file`. of the old version.
- [ ] Update versions in docs/source/conf.py.

If a release changes behavior, update relevant docstrings and tutorials.

## Releasing on PyPI

Once a new version is ready for release to PyPI, follow these steps:

### 1. Update the version in `pyproject.toml`
```toml
[project]
version = "X.Y.Z"
```

### 2. Commit this change and tag the release:
```bash
git add pyproject.toml
git commit -m "Release X.Y.Z"
git tag vX.Y.Z
git push
git push --tags
```

### 3. Build the package
Use the `build` package
```bash
pip install build
python -m build
```

This generates
```bash
dist/
  pythtb-X.Y.Z.tar.gz
  pythtb-X.Y.Z-py3-none-any.whl
```

### 4. Upload to PyPI
Install `twine` if needed
```bash
pip install twine
```

Upload
```bash
twine upload dist/*
```

or

```bash
******************* BE CAREFUL WITH THIS *************************

  git checkout main
  git pull
  rm -rf dist
  python setup.py sdist
  twine upload dist/* -r pypi
  
******************************************************************
```

This uploads to PyPI. Verify the release appears at [https://pypi.org/project/pythtb/](https://pypi.org/project/pythtb/). This procedure could be automated with GitHub Actions.

## Releasing on conda-forge

PythTB is maintained on conda-forge via a __feedstock__ repository. When a new PyPI version is released, the feedstock must be updated.
See [conda-forge documentation](https://conda-forge.org/docs/maintainer/updating_pkgs/#updating-the-maintainer-list) for more details.

The conda-forge bot will 

- Detect the new version release
- Propose a version bump
- Update the hash and metadata

If the bot doesn’t automatically trigger, you can manually bump the version in recipe/meta.yaml.

### 1. Navigate to feedstock
Navigate to the feedstock at [https://github.com/conda-forge/pythtb-feedstock](https://github.com/conda-forge/pythtb-feedstock)

### 2. Fork feedstock
Create your own fork of the feedstock, clone it to your computer.

### 3. Implement changes in a new branch
Create a new branch and implement your changes locally. Commit your changes.

#### Updating recipes
- Always use a fork of the feedstock while updating the recipe.
- When a package's version is not changed, but other metadata or parts of the recipe are changed, increase the build number by 1.
- While shipping a new version of your package, reset the build number to 0.

#### Rerendering feedstocks
Rerendering is conda-forge's way to update the files common to all feedstocks (e.g., README, [CI](https://conda-forge.org/docs/glossary/#ci) configuration, pinned dependencies). Rerendering can be done in two ways:

- Using the webservice to run conda-smithy on the cloud by adding the comment ``@conda-forge-admin, please rerender``
- Run conda-smithy locally on your machine

### 4. Create a PR back to the original feedstock
Push your changes and create a pull request on GitHub.


## New Version

### Check-list

A new release has been prepared and is up to date with the 'dev' branch. Some things to check before proceeding:

- [ ] All tests are passing

- [ ] Documentation has been updated throughout the code

- [ ] All examples run

After these have been checked off, we create a new branch 

```bash
  git checkout -b release/X.Y.Z
```

Update __version__ variable and date in the header of pythtb.py.
Also, update the year in the line below, starting with "Copyright"

Update version string in setup.py


Update docs/local/release/release.rst by specifying what is new in the
package.

  
Add folder in website/local/release/ver_ABC for the old version of the package,
put in .tar.gz file. of the old version (it is enough to do this for the old version only).

New version will be added automatically by the "go" script in website folder.
To get old source file go to this website: https://pypi.org/project/pythtb/1.OLD.OLD/#files
and download the source. Then do,

```
  git add ver_1.OLD.OLD
  git add ver_1.OLD.OLD/pythtb-1.OLD.OLD.tar.gz
```

Update website/source/install.rst with the new version. It is enough to
change string "1.6.1" with "1.6.2" or similar.

```
  git add install.rst
```

Update website/source/conf.py. Version number appears at two places, just
update the string.

There is NO need to update src/CHANGES. We may want to remove that
file in the future.

4h. Do a quick grep on the old version number just to make sure there
isn't something new that needs to be done.

5. We should test the website to make sure it is rendered correctly. David
should do this test on release/1.x.x branch

   git checkout release/1.x.x

6. Sinisa will wait for David's confirmation that website is running well
on his end before proceeding to the next step.  David can push to release/1.x.x
if he has any additional corrections.

7. When we are happy with polishing release/1.x.x, we should merge
it into develop and master (checking for and resolving conflicts if develop
has been changed).

8. Now Sinisa will update the code on the PyPI server.

8a. Make sure that you link ../../private/.pypirc_MOVE_TO_HOME to home folder as ~/.pypirc  !
This file contains a passwords so I'm not keeping it on github.

8b. Here are instructions for testing purposes only:

  git checkout master
  git pull
  rm -rf dist
  python setup.py sdist
  twine upload dist/* -r testpypi

8c. If you wish to test the package do this,

  git checkout master
  git pull
  pip install -i https://test.pypi.org/simple/ pythtb==1.8.0

8d. When you are sure that this works you can officially upload it to pypi like this.
Note that this code below should not be executed lightly! These lines will make
some changes in the pypi servers and after that it is hard to tweak things.
Therefore make sure you do all the tests you need to do first with the 
"testpypi" version!  Once you are happy with how things look like execute
lines below:

******************* BE CAREFUL WITH THIS *************************

  git checkout master
  git pull
  rm -rf dist
  python setup.py sdist
  twine upload dist/* -r pypi
  
******************************************************************

9. David should now do the final update of the website from the "master" branch of git.  David,
please add here more information if needed.

   git checkout master
   git pull
   ...

10. Sinisa will now make sure that conda-forge is up to date.
Please follow instructions here: https://conda-forge.org/docs/maintainer/adding_pkgs/
Once you do, please list here what had to be done to update conda-forge.

11. Update this file if needed.




