# Documentation Guide

## Read the Docs hosting
_Read the Docs_ is a free hosting service that builds and serves the documentation with Sphinx.
It:

- Watches the remote repository
- On every push, checks out the repo, installs dependencies, and runs `sphinx-build`

No manual deployment step is needed.

## Sphinx
Sphinx is a static documentation generator. You write the webpages in reStructuredText (.rst) or Myst Markdown (.md), and Sphinx converts them into HTML. It can also inspect Python modules (autodoc) and convert the docstrings into HTML. Sphinx also supports pre-built themes for .css styling. We use the [PyData theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html).

__Features:__

- Parses .rst / .md documents
- Builds HTML
- Supports cross-references, math, and theming
- Uses autodoc to pull docstrings directly from source code
- Uses [autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html) to generate stub pages for modules/classes/functions 

The result is a static site served to users. 

## Updating the documentation

A typical workflow may be something like,

1. Edit content
- .rst / .md pages in /docs or,
- docstrings in the code-base or,
- tutorial Jupyter notebook

2. Viewing changes
- Build locally using

```bash
cd docs
make html
```

3. commit and push changes to the remote repo
4. RTD automatically updates and publishes the website

Below are guidelines for each part of the workflow.

## Building locally

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


## Writing .rst/.md source files

Each page in `docs/source/` represents a different page of the website. Using the `myst-nb` Sphinx extension, we can write pages in either reStructuredText (.rst) or Myst Markdown (.md). `index.md` is the homepage, and it links to all of the other .md/.rst pages to be included on the site. For example, at the bottom of `index.md` we have

````md
```{eval-rst}
.. toctree::å
:maxdepth: 1
:hidden:

About <about>
install
API <api>
Tutorials <tutorials>
Development <development>
release
CHANGELOG
formalism
resources
citation
```
````

Inside the `{eval-rst}` directive (surrounded by triple backticks), we write .rst syntax. Here, we use the `toctree` directive to specify which pages to include in the documentation. These will appear in the sidebar navigation and the top navigation bar.
What is written in the toctree is the name of the file (omitting the extension), the name that is rendered in the navigation is by default set to the top level header of that file. We can also give the name directly using `Name <file>`. See the [Sphinx documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer) for instructions on writing .rst.

## Writing Docstrings

The code API is documented using docstrings in the source code. We use NumPy-styled docstrings. Sphinx parses this using `numpydoc` + `autodoc` (see [Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) and [numpydoc extension](https://numpydoc.readthedocs.io/en/latest/usage.html#sphinx-extension)).

To update the API documentation, modify the docstrings in the relevant Python files in the `pythtb/` directory. After making changes, rebuild the documentation to reflect the updated API docs. The API reference page is located at `docs/source/usage.md`. This will use Sphinx's `autodoc` feature to pull in the updated docstrings and generate the .rst files in `docs/source/generated/`. Sphinx is very picky with the syntax of the docstrings. Below is an example describing each part of the docstring: 

### Basic Structure
```python

def new_function(
    param1: np.ndarray, 
    param2: bool = False
    ):
    r"""
    Brief summary in one line.

    Extended explanation, if needed. Keep line length reasonable and
    avoid overly technical notation unless necessary.

    .. versionadded: X.Y.Z
       This is a description about the new things in the function.

    .. versionchanged: X.Y.Z
       This is a description of what changed in the function.

    .. versionremoved: X.Y.Z
       This behavior or parameter no longer exists.
    
    .. deprecated: X.Y.Z
       This thing is no longer supported and will be removed.

    Parameters
    ----------
    param1 : numpy.ndarray of shape (..., 3)
        An array of this, used for that.
 
        .. versionchanged: X.Y.Z
           This is renamed, this parameter behaves differently.

    param2 : bool, optional
        If True, does this. Otherwise, that. Default is False.

    Returns
    -------
    output : ndarray
        Array of this elements defined in that way. Make sure to pay 
        attention to this side case. Expected shapes are 
        
        - ``(x, y, *z)`` if this scenario, double backticks render as code literal
        - ``(w, x, y, *z)`` if that scenario

   output2 : int, optional
        This is another possible output, only returned when this happens.

    See Also
    --------
    module_method : This is a description. The module_method is defined in this script.
    class.method  : class is defined in the package somewhere, method is its class method.
    class : class itself somewhere in package

    Notes
    -----
    - You can include math LaTeX 

      .. math:: 
         \mathbf{x} \cdot \mathbf{y} = \mathcal{C}
     
      or use it inline like :math:`\mathbf{v}`.

    - This is a ``code_literal``, this is a `hyperlink <www.something.com>`_ 
    - This is a reference to a :meth:`class_method` or a :class:`class_somewhere_else`

   Examples
   --------
   This is an example, you do this in code
   
   >>> def method(
   ...     param
   ...  )
   >>> x = method(param=[0, 4])
   >>> x.shape
   (5,)

   Then do this thing,
   
   >>> for i in x:
   ...    x[i] -= 1
   >>> x
   [-1, 0, 1, 2, 3] # ouput line

   """

```

Not all of these things are necessary in every docstring, but "Parameters" and "Returns" are always expected. In the docstring, there can be .rst-type syntax as well that Sphinx will render. For example 

__Sphinx roles for linking__
- `:mod:`: modules
- `:class:` : classes
- `:meth:` : method
- `:func:` : functions
- `:attr:` : attributes

__Math__
Using `` :math:`$x+y$` ``, or 
```python
.. math::
    x + y
```

Sphinx is sensitive to the syntax details and the documentation will not render correctly if the expected syntax is broken. Some of this syntax is .rst (e.g., the double dots and math blocks); see [Sphinx's description of reStructured Text](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer). Notice some things about the documentation in general

- The indentation level.
- The extra lines around the math block and the `..versionadded` etc. 
- Backticks surrounding .rst `` :meth:`function` ``
- The raw string prefix behind the opening triple quotes `r"""` . This will treat backslashes literally instead as the start of an escape sequence.


### General style

For guidance on the style of numpy docstrings, see the [numpydoc reference](https://numpydoc.readthedocs.io/en/latest/format.html#overview) or [this example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy). Generally, we expect,

- The order of the sections should be consistent throughout the documentation, and follow the order written above.
- Imperative tone (“Compute X”, not “Computes X”)
- Be concise but informative
- Use See Also links rather than repeating content
- Add reference knowledge where helpful

## Writing .ipynb tutorial notebooks

To add new tutorials, create a new Jupyter notebook in the `docs/source/tutorials/` directory. Ensure that the notebook is well-documented and includes explanations of the code. After adding the notebook, rebuild the documentation to include it in the tutorials section. Upon building, Sphinx with the `myst-nb` extension will convert the notebook into a static HTML page and link it appropriately in the tutorials section of the documentation. See [MyST-nb documenation](https://myst-nb.readthedocs.io/en/latest/authoring/jupyter-notebooks.html) for syntax details.

We have added a custom template for auto-generating a converted .py script from the .ipynb file, which is stored in the `build/dirhtml/tutorials_py/` directory when building with `dirhtml`. Each notebook will have a link at the top to download the .ipynb and .py script versions.
We use Binder to host live, interactive versions of the tutorial notebooks. The configuration of Binder is in the root directory under the `.binder/` folder. Each tutorial notebook has a "Launch Binder" button at the top that links to the live version hosted on Binder. See the [Binder documentation](https://mybinder.readthedocs.io/en/latest/) for more details on how this works.

## Troubleshooting

- If you remove a public class or method, there will still be the residual .rst files in docs/source/generated, and you may get an error like

  ```bash
  WARNING: autodoc: failed to import property 'TBModel.removed_property' from module 'pythtb'; the following exception was raised:
  ```

  Just delete the entire `/generated` directory or the obsolete .rst files. After deletion, the next build will regenerate the `docs/source/generated/` folder with only the current API documentation.