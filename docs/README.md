## Documentation Structure

The documentation is organized into the following sections:


## Building the Documentation

Sphinx is used to build the documentation.

From the project root:

```bash
cd docs/
```

Then run either 

__Option 1: Standard HTML__

```bash
make html
```

The generated site will be located at `docs/_build/html/`.

Alternatively:

```bash
sphinx-build -b html . _build/html
```

__Option 2: Directory HTML__

```bash
make dirhtml
```

This produces a directory structure in `_build/dirhtml/` that mirrors the source structure, which can be useful for hosting on certain web servers.

__Making Clean Builds__

Use 

```bash
make clean html
```

to clean previous builds before building.

## Viewing the Documentation Locally

A utility script named `see` is provided to open the built documentation in your default web browser.
From the docs root (`docs/`), run:

```bash
./see
```

Optionally, you can specify the build type (`html` or `dirhtml`):

```bash
./see --html # or
./see --dirhtml
```

This will open the `index.html` file from the specified build in your default web browser hosted on a local server.

## Contributing to the Documentation

Contributions to the documentation are welcome! Please follow the guidelines in the GitHub wiki, and read the CONTRIBUTING.md file in the repository root for more information on how to contribute.

### Editing Docs

To edit the documentation, navigate to the `docs/source/` directory and modify the relevant `.md` or `.rst` files. After making changes, rebuild the documentation using one of the methods described above to see your updates.

### Adding Examples

To add new examples, create a new Jupyter notebook in the `docs/source/examples/` directory. Ensure that the notebook is well-documented and includes explanations of the code. After adding the notebook, rebuild the documentation to include it in the examples section.
