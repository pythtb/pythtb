# Tutorials
    
[![Run tutorials on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pythtb/pythtb/dev?urlpath=lab/tree/docs/source/tutorials/)

This section contains a collection of tutorials demonstrating how to use PythTB to build and analyze tight-binding models. Each tutorial is provided as a Jupyter notebook (.ipynb) file, which includes both the code and explanatory text.

To open a live version of the content, click the **launch Binder** button above. This will open a JupyterLab environment in your web browser where you can interactively run and modify the tutorial notebooks without needing to install anything on your local machine. You can also launch individual tutorials on Binder by clicking on the rocket icon that appears in the upper-left corner of each tutorial. 
To download a local copy of the .ipynb or converted .py files, you can use the download icon in the upper-left corner of each tutorial.

If you are unfamiliar with Python or are not sure whether Python and the necessary modules are installed on your system, see our {doc}`python introduction <resources>` and {doc}`installation instructions <install>`.

```{note}
There is a useful [`collection of PythTB sample programs`](https://minisites.cambridgecore.org/berryphases/ptb_samples.html)
that were developed in connection with David Vanderbilt's book [`Berry Phases in Electronic Structure Theory`](https://www.cambridge.org/9781107157651)(Cambridge University Press, 2018).
```
   
```{admonition} v2.0 Upgrade Notice
:class: attention
For tips on upgrading to v2.0, see the {doc}`release notes <release/2.0.0-notes>`. For a full list of changes, see the {doc}`changelog <CHANGELOG>`. Explore the new features with the **New to v2.0 tutorials**.
```

```{toctree}
:maxdepth: 1
:caption: New to v2.0

tutorials/lattice
tutorials/mesh
tutorials/tb_model_v2
tutorials/wfarray_v2
tutorials/param_model
tutorials/haldane_wannier
tutorials/reduced_wannier
tutorials/quantum_geom_tens
tutorials/local_chern
tutorials/axion_fkm
tutorials/nn_shells
tutorials/visualize_3d
```

```{toctree}
:maxdepth: 1
:caption: Building the TBModel

tutorials/0dim
tutorials/checkerboard
tutorials/graphene
tutorials/haldane
tutorials/buckled_layer
tutorials/trestle
tutorials/supercell
```

```{toctree}
:maxdepth: 1
:caption: Topology and quantum geometry

tutorials/finite_ssh
tutorials/three_site_thouless
tutorials/graphene_cone
tutorials/haldane_bp
tutorials/haldane_hwf
tutorials/kane_mele
tutorials/fkm_model
tutorials/boron_nitride
tutorials/slab_hwf
```

```{toctree}
:maxdepth: 1
:caption: Wannier90 integration

tutorials/w90
```

```{toctree}
:maxdepth: 1
:caption: Visualization

tutorials/visualize
tutorials/haldane_edge
```