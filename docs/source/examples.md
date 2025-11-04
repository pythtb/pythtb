# Examples
[![Run examples on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythtb/pythtb/dev?urlpath=lab/tree/docs/source/examples/)

This section contains a collection of examples demonstrating how to use PythTB
to build and analyze tight-binding models. Each example is provided as a Jupyter
notebook (.ipynb) file, which includes both the code and explanatory text.

To open a live version of the content, click the **launch Binder** button above.
This will open a JupyterLab environment in your web browser where you can
interactively run and modify the example notebooks without needing to install
anything on your local machine. You can also launch individual tutorials on Binder by clicking on the rocket icon that appears in the upper-left corner of each tutorial. To download a local copy 
of the .ipynb or converted .py files, you can use the download icon in the 
upper-left corner of each tutorial.

If you are unfamiliar with Python or are not sure whether Python and
the necessary modules are installed on your system, see our
{doc}`python introduction <resources>`
and {doc}`installation instructions <install>`.

```{note}
There is a useful [`collection of PythTB sample programs`](https://minisites.cambridgecore.org/berryphases/ptb_samples.html)
that were developed in connection with David Vanderbilt's book [`Berry Phases in Electronic Structure Theory`](https://www.cambridge.org/9781107157651)(Cambridge University Press, 2018).
```
   
```{tip}
See **New to v2.0** for examples demonstrating features added in PythTB v2.0.
```

```{toctree}
:maxdepth: 1
:caption: New to v2.0

examples/lattice
examples/mesh
examples/tb_model_v2
examples/wfarray_v2
examples/param_model
examples/haldane_wannier
examples/reduced_wannier
examples/quantum_geom_tens
examples/local_chern
examples/axion_fkm
examples/nn_shells
examples/visualize_3d
```

```{toctree}
:maxdepth: 1
:caption: Building the TBModel

examples/0dim
examples/checkerboard
examples/graphene
examples/haldane
examples/buckled_layer
examples/trestle
examples/supercell
```

```{toctree}
:maxdepth: 1
:caption: Topology and quantum geometry

examples/finite_ssh
examples/three_site_thouless
examples/graphene_cone
examples/haldane_bp
examples/haldane_hwf
examples/kane_mele
examples/fkm_model
examples/boron_nitride
examples/slab_hwf
```

```{toctree}
:maxdepth: 1
:caption: Wannier90 integration

examples/w90
```

```{toctree}
:maxdepth: 1
:caption: Visualization

examples/visualize
examples/haldane_edge
```