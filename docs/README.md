
## Documentation Structure

The documentation is organized into the following sections:


## Building the Documentation
- **Sphinx** is used to build the documentation. To build the docs locally, navigate to the `docs/` directory and run:
    ```bash
    make html
    ```
    The generated HTML files will be located in the `_build/html/` directory.

- You can use:
    ```bash
    sphinx-build -b html . _build/html
    ```
- Or, you can use:
    ```bash
    make dirhtml
    ```
    This will create a directory structure in `_build/dirhtml/` that mirrors the source structure, which can be useful for hosting on certain web servers.

Some bash scripts are provided to help with building and deploying the documentation. 
- `serve` - Builds and serves the built documentation to localhost for easy viewing.
- `see` - Opens the built documentation in the default web browser.
