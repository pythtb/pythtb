# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import pythtb
import logging
import plotly.io as pio
pio.renderers.default = 'sphinx_gallery'
logging.getLogger("pythtb").setLevel(logging.WARNING)

project = u'PythTB'
copyright = '2025, PythTB team'
author = 'PythTB team'
version = pythtb.__version__

# preamble for latex formulas
# pngmath_latex_preamble = r"\usepackage{cmbright}"
# pngmath_dvipng_args = ['-gamma 1.5', '-D 110']
# pngmath_use_preview = True

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'no-show-inheritance': True,
}
autodoc_typehints = "description"

# link to numpy and python
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  
    'sphinx.ext.viewcode',
    "sphinx.ext.autosummary",
    # 'myst_parser',        # <-- enable Markdown
    "myst_nb",
    'sphinx.ext.doctest',
   # 'sphinx.ext.imgmath',
    'matplotlib.sphinxext.plot_directive',
    # 'sphinx_thebe',
    'sphinx.ext.mathjax',
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinxcontrib.programoutput",
    "sphinx_design",
    "sphinx_gallery"
    # "numpydoc"
]

myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "dollarmath",
    "amsmath",
    "substitution",
    "colon_fence",
    "attrs_inline"
]

nb_execution_mode = "cache"        # instead of "auto"
nb_execution_timeout = 600     # seconds per notebook
nb_execution_cache_path = ".jupyter_cache"  # keep cache OUTSIDE _build so 'clean' doesn't erase it
nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/vnd.plotly.v1+json",   # <- key line
        "text/html",
        "image/svg+xml",
        "image/png",
        "text/markdown",
        "text/plain",
    )
}

thebe_config = {
    "binderhub_url": "https://mybinder.org",
    "binder_options": {
        "repo": "pythtb/pythtb",
        "ref": "dev"
    },
    "kernelName": "python3",
    # Activate on containers with class "thebe"; pair with an output container
    # CSS selector for code cells
    "selector": ".thebe",
    "selector_output": ".output",
    "codemirror-config": {"theme": "abcdef"},
}

copybutton_only_copy_prompt_lines = False
copybutton_remove_prompts = True

# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3', None),
#     'numpy': ('https://numpy.org/doc/stable/', None)
# }

# tell Sphinx to treat .md files as sources
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

# for matplotlib plots
plot_formats=[('png',140),('pdf',140)]
pygments_style = "sphinx"
pygments_dark_style = "monokai"  # for dark theme compatibility

# for autodoc to work on PythTB package
sys.path.append("../src")

html_theme = 'pydata_sphinx_theme' #'sphinx_book_theme' #'classic' pydata_sphinx_theme
html_title = f"{project} Docs"
templates_path = ['_templates']
html_static_path = ['_static']
html_js_files = [
    ("custom-icons.js", {"defer": "defer"})
]
html_js_files += [
    ("plotly-2.34.0.min.js", {"defer": "defer"}),  # update filename to what you vendored
]
html_extra_path = ['misc', 'simple_fig', 'examples_py']
html_css_files = ["custom.css"]
html_copy_source = True
html_show_sourcelink = False
html_sourcelink_suffix = ""
exclude_patterns = ['generated/*.md', 'examples_rst/*', 'examples_py/*']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_context = {
    "github_user": "pythtb",
    "github_repo": "pythtb",
    "github_version": "main",
    "doc_path": "docs",
}

html_sidebars = {
    "index": [],
    "install": [],
    "getstarted": [],
    "CHANGELOG": [],
    "formalism": [],
    "resources": [],
    "citation": [],
}


html_theme_options = {
    "logo": {
        "image_light": "_static/pythtb_logo2_dark.svg",
        "image_dark": "_static/pythtb_logo2_dark.svg",
    },
    "github_url": "https://github.com/pythtb/pythtb",
    "collapse_navigation": False,
#     "navigation_depth": 4,
    "article_header_end": ["nb-download"],
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
#     "show_nav_level": 2,
    # "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "navbar-icon-links"
    ],
    "navbar_persistent": [],
    # "switcher": {
    #     "version_match": version,
    #     "json_url": "_static/switcher.json",
    # },
    "show_version_warning_banner": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pythtb/pythtb",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pythtb/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    # "use_thebe": True
}

# html_theme_options["use_thebe"] = True  # e/nables Thebe for notebook
# html_js_files = [
    # "https://unpkg.com/thebe@latest/lib/index.js"
# ] # for executing code

# documentation.
# html_theme_options = {
#     # "repository_url": "https://github.com/sinisacoh/pythtb",
#     # "use_repository_button": True,
#     # "use_issues_button": True,
#     # "use_edit_page_button": True,
#     # "path_to_docs": "docs/source",
#     # "repository_branch": "main",
#     # "use_download_button": True,
#     # "home_page_in_toc": True,
#     "show_toc_level": 2,
# }
# html_sidebars = {
#           '**':    ['globaltoc.html', 'localtoc.html', 'searchbox.html'],
#           'index': ['globaltoc.html', 'searchbox.html'],
#        }
# remove "show source" from website

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# preamble for latex formulas
pngmath_latex_preamble=r"\usepackage{cmbright}"
pngmath_dvipng_args=['-gamma 1.5', '-D 110']
pngmath_use_preview=True

# Output file base name for HTML help builder.
htmlhelp_basename = 'PythTBdoc'

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'PythTB.tex', u'PythTB Documentation',
   u'Trey Cole, Sinisa Coh and David Vanderbilt', 'manual'),
]

man_pages = [
    ('index', 'pythtb', u'PythTB Documentation',
     [u'Trey Cole, Sinisa Coh and David Vanderbilt'], 1)
]

texinfo_documents = [
  ('index', 'PythTB', u'PythTB Documentation',
   u'Trey Cole, Sinisa Coh and David Vanderbilt', 
   'PythTB', 'Python software package implementation of tight-binding approximation',
   'Miscellaneous'),
]

# for autodoc so that things are ordered as in source
autodoc_member_order = 'bysource' 


# In order to skip some functions in documentation
def setup(app):
    app.connect('autodoc-skip-member', _maybe_skip_member)
    app.connect("builder-inited", _export_ipynb_to_py)

def _export_ipynb_to_py(app):
    import os, nbformat
    from nbconvert import ScriptExporter

    srcdir = app.srcdir
    nb_root = os.path.join(srcdir, "examples_ipynb")   # adjust if yours differs
    out_root = os.path.join(srcdir, "_static", "nb-scripts")
    exporter = ScriptExporter()

    for root, _, files in os.walk(nb_root):
        for name in files:
            if not name.endswith(".ipynb"):
                continue
            in_path = os.path.join(root, name)
            rel = os.path.relpath(root, nb_root)
            os.makedirs(os.path.join(out_root, rel), exist_ok=True)

            nb = nbformat.read(in_path, as_version=4)
            body, _ = exporter.from_notebook_node(nb)
            out_path = os.path.join(out_root, rel, name[:-6] + ".py")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(body)

def _maybe_skip_member(app, what, name, obj, skip, options):
    if name in ["tbmodel","add_hop","set_sites","no_2pi"]:
        return True
    else:
        return skip
