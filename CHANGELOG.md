# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [Unreleased]

## [2.0.0] - 2025-11-??

### Fixed
- Fixed bug in `TBModel._shift_to_home()` where only the last orbital was shifted. This affected the `to_home` flag in `change_nonperiodic_vector()` and `make_supercell()`.

### Improved
- Vectorized code throughout using NumPy for substantial speed improvements
  - `TBModel` initialization, Hamiltonian construction, and diagonalization orders of magnitude faster for large models
  - `W90.model()` construction from Wannier90 data significantly faster, allowing practical use with large first-principles models
- Type hints added throughout the codebase for improved developer experience and IDE support
- Modernized Sphinx-based documentation website, copying over the previous tutorials, and adding some new ones to cover new features
- `TBModel.visualize()`: Enhanced 2D Visualization

### Changed

- Restructured from single `pythtb.py` file to organized `pythtb/` package with separate purpose-specific modules:
  - `tbmodel.py`: Tight-binding model class and methods
  - `wfarray.py`: Wavefunction array class for storing and manipulating quantum states
  - `w90.py`: Wannier90 interface
  - etc.
- Migrated from `setup.py` to modern `pyproject.toml` configuration as per PEP 518

**Breaking Changes**

- Updated class and method names to follow PEP 8 conventions:
  - `tb_model` -> `TBModel`
  - `wf_array` -> `WFArray`
  - `w90` -> `W90`
- `TBModel` Initialization Changes
  - Replaced `dim_r`, `dim_k`, `lat`, `orb`, and `per` parameters with a single `Lattice` instance
  - Replaced `nspin` parameter with `spinful` boolean flag
- `TBModel.solve_ham()` (replaces `solve_one()` and `solve_all()`)
  - Changed eigenvalue/eigenvector indexing for vectorized workflows
  - Eigenvalues: shape `(nk, nstate)` (matrix elements last for NumPy compatibility)
  - Eigenvectors for spinless (`nspin=1`): shape `(nk, nstate, nstate)`
  - Eigenvectors for spinful (`nspin=2`): shape `(nk, nstate, norb, 2)`
- `TBModel.position_expectation()` parameter renaming
  - Renamed parameter `evec` to `evecs` for clarity
  - Renamed parameter `dir` to `pos_dir` to avoid conflict with built-in Python function `dir()`
- `TBModel.position_matrix()` parameter renaming
  - Renamed parameter `evec` to `evecs` for clarity
  - Renamed parameter `dir` to `pos_dir` to avoid conflict with built-in Python function `dir()`
- `TBModel.position_hwf()` parameter renaming
  - Renamed parameter `evec` to `evecs` for clarity
  - Renamed parameter `dir` to `pos_dir` to avoid conflict with built-in Python function `dir()`

- `WFArray` Initialization Changes:
  - Replaced `mesh_arr` parameter with a `Mesh` instance
  - Replaced `model` parameter with a `Lattice` instance
  - Renamed `nsta_arr` parameter to integer `nstates` for clarity
- `WFArray.berry_phase()` parameter renaming 
  - `dir` renamed to `axis_idx` to avoid conflict with Python built-in `dir()`
  - `occ` renamed to `state_idx` to emphasize that band indices need not be occupied
  - Removed `"all"` option; use `None` (default) to include all states
- `WFArray.berry_flux()` parameter renaming
  - `dirs` renamed to `plane` now only accepts 2-element tuples defining planes
  - `occ` renamed to `state_idx` to emphasize that band indices need not be occupied
- `WFArray.position_matrix()` parameter renaming
  - `dir` renamed to `pos_dir` to avoid conflict with built-in Python function `dir()`
  - `occ` renamed to `state_idx` to emphasize that states need not be occupied
  - Removed `"all"` option; use `None` (default) to include all states
- `WFArray.position_expectation()` parameter renaming
  - `dir` renamed to `pos_dir` to avoid conflict with built-in Python function `dir()`
  - `occ` renamed to `state_idx` to emphasize that states need not be occupied
  - Removed `"all"` option; use `None` (default) to include all states
- `WFArray.position_hwf()` parameter renames:
  - `dir` renamed to `pos_dir` to avoid conflict with built-in Python function `dir()`
  - `occ` renamed to `state_idx` to emphasize that states need not be occupied
  - Removed `"all"` option; use `None` (default) to include all states
- `WFArray.choose_states()` parameter rename:
  - `subset` renamed to `state_idxs` for clarity and consistency
- `WFArray.empty_like()` parameter rename:
  - `nsta_arr` renamed to `nstates` for clarity and consistency

- `W90.w90_bands_consistency()` (deprecated and renamed to `bands_w90()`)
  - Returned energy array shape changed from `(band, kpts)` to `(kpts, band)`
  - Now consistent with eigenvalue shape from `TBModel.solve_ham()`
  - Aligns with NumPy convention of putting k-points in first axis

### Added
- Published `pythtb` package to [conda-forge](https://anaconda.org/conda-forge/pythtb) for easy installation via `conda install -c conda-forge pythtb`
- Optional TensorFlow backend for linear algebra acceleration on compatible hardware (GPUs/TPUs) in `TBModel` and `WFArray`
  - Enable by passing `use_tensorflow=True` on some methods
- Comprehensive unit tests added using `pytest` to cover core functionality
- New examples and tutorials added to documentation website covering new features and workflows

**New classes**
- `Lattice` class that handles lattice geometry and reciprocal operations.  
  - Encapsulates lattice manipulation methods previously embedded in `TBModel`  
  - Used by `TBModel` and `WFArray`
- `Mesh` class that defines structured grids in k-space or parameter space.  
  - Supports arbitrary dimensions and mixed $(k, \lambda)$ meshes  
  - Includes `Axis` helper class for labeled mesh axes  
  - Used by `WFArray` for consistent data mapping
- `Wannier` class for constructing and analyzing Wannier functions from `WFArray` 
  - Projection onto trial orbitals  
  - Iterative maximal localization/disentanglement  
  - Visualization of centers, decay profiles, and spreads

**New modules**
- `pythtb.models`: collection of common tight-binding modelsthat are importable using, e.g.,
- `pythtb.io.w90`: Wannier90 file parsing utilities
  - `read_hr()`, `read_centres()` etc. for standalone Wannier90 file parsing
  - Supports loading a full Wannier90 dataset for downstream `W90` -> `TBModel` processing
- `pythtb.io.qe`: Quantum ESPRESSO file parsing utilities
  - `read_bands_qe()` for reading `prefix_bands.out` `bands.x` output files

**New features to pre-existing classes**
- `TBModel` methods:
  - `TBModel.__str__`: Allows printing a `TBModel` instance using `print(TBModel)`, which prints `TBModel.info()`
  - `TBModel.info()`: Replaces `display()` for printing model summary
  - `TBModel.get_lat_vecs()`: Replaces `get_lat()` for clarity
  - `TBModel.get_orb_vecs()`: Replaces `get_orb()` for clarity
    - Added boolean flag `cartesian` to return orbital vectors in Cartesian coordinates (default `False`)
  - `TBModel.solve_ham()`: Diagonalizes Hamiltonian and returns eigenvalues and optionally eigenvectors
    - Replaces `solve_one()` and `solve_all()` with a unified, vectorized diagonalization method
  - Symbolic `TBModel.set_onsite` and `TBModel.set_hop` : both accept strings and callables for setting onsite energies and hoppings allowing for parameter-dependent terms
  - `TBModel.with_parameters()` returns model at specific parameter values for parameterized models 
  - `TBModel.set_parameters()` resolves parameterized terms with scalar values
  - `TBModel.set_shell_hops()`: Bulk setting of n'th nearest-neighbor hoppings for faster model construction
  - `TBModel.hamiltonian()` constructs Hamiltonians for finite and periodic systems
  - `TBModel.velocity()` computes velocity operator $dH/dk$ in orbital basis
  - `TBModel.quantum_geometric_tensor()` quantum geometric tensor using Kubo formula
  - `TBModel.quantum_metric()` quantum metric tensor from quantum geometric tensor
  - `TBModel.berry_curvature()` berry curvature from quantum geometric tensor
  - `TBModel.chern_number()` computes Chern number using Berry curvature
  - `TBModel.axion_angle()` computes axion angle using 4-curvature integration
  - `TBModel.local_chern_marker()` Bianco-Resta formula for real-space Chern marker
  - `TBModel.get_recip_lat()` returns reciprocal lattice vectors
  - `TBModel.make_finite()`: Convenience function for chaining `cut_piece()` along different directions
  - `TBModel.plot_bands()` built-in band structure plotting utility
  - `TBModel.visualize3d()` interactive 3D visualization for 3D models

- `TBModel` attributes
  - `TBModel.assume_position_operator_diagonal`: attribute setter to control diagonal approximation for position operator
    - Replaces deprecated `ignore_position_operator_offdiagonal()` method
  - `TBModel.lattice`: read-only property returning associated `Lattice` instance
  - `TBModel.nspin`: read-only property returning spinful/spinless status
  - `TBModel.periodic_dirs`: read-only property returning list of periodic directions
  - `TBModel.norb`: read-only property returning number of orbitals
  - `TBModel.nstate`: read-only property returning number of states
  - `TBModel.dim_r`: read-only property returning real-space dimension
  - `TBModel.dim_k`: read-only property returning k-space dimension
  - `TBModel.onsite`: read-only property returning on-site energies as NumPy array
  - `TBModel.hoppings`: read-only property returning hoppings as list of dictionaries
  - `TBModel.spinful`: read-only property returning spinful/spinless status

- `WFArray` methods:
  - `WFArray.overlap_matrix()`: Computes overlap matrix of the states in the `WFArray` with their nearest neighbors on a k-`Mesh`.
  - `WFArray.links()`: Computes the unitary part of the overlap between states and their nearest neighbors in each mesh direction
  - `WFArray.berry_connection()`: Computes Berry connection from the links between nearest neighbor states in the mesh
  - `WFArray.wilson_loop()`: Static method that computes the Wilson loop unitary matrix for a loop of states
  - `WFArray.berry_curvature()`: Computes dimensionful Berry curvature by divinding Berry flux by mesh cell area/volume
  - `WFArray.chern_number()`: Returns the Chern number for a given plane in the parameter mesh
  - `WFArray.solve_model()`: Populates `WFArray` with energy eigenstates from a given `TBModel` along the `Mesh`
    - Replaces deprecated `solve_on_grid()` and `solve_on_one_point()` methods
  - `WFArray.projectors()`: Returns band projectors and optionally their complements as NumPy arrays
  - `WFArray.states()`: Returns states as a NumPy array, optionally the full Bloch states including phase factors
  - `WFArray.get_k_shell()`: Generates vectors connecting nearest neighboring k-points in the mesh. 
  - `WFArray.get_shell_weights()`: Returns the finite-difference weights for a given shell of k-neighbors.
  - `WFArray.roll_states_with_pbc()`: Rolls states along a given mesh axis with periodic boundary conditions.
  - `WFArray.copy()`: Creates a deep copy of the `WFArray` instance
  - Added parameter `non_abelian` to `WFArray.berry_flux()` to compute non-Abelian Berry flux for a manifold of states

- `W90` methods:
  - `W90.bands_w90()`: Replaces `w90_bands_consistency()` for clarity
    - Returns energy array with shape `(kpts, band)` consistent with `TBModel.solve_ham()`
  - `W90.bands_qe()`: Reads band structure from Quantum ESPRESSO `prefix_bands.out` output files

### Deprecated

The following methods are deprecated but still functional with backward compatibility wrappers.
These will be removed in a future release.

- `tb_model`, and `w90` class names remain available as aliases for new classes
- `TBModel.display()` (deprecated) renamed to `TBModel.info()` to prevent confusion with visualization 
  - Use `TBModel.info()` or `print(my_model)` instead
- `TBModel.get_lat()` (deprecated) renamed to `TBModel.get_lat_vecs()` for clarity
- `TBModel.get_orb()` (deprecated) renamed to `TBModel.get_orb_vecs()` for clarity
- `TBModel.set_onsite()` and `TBModel.set_hop()` modes:
  - `reset` mode is deprecated; use `set` instead
- `TBModel.solve_one()` (deprecated) replaced by `TBModel.solve_ham()`
- `TBModel.solve_all()` (deprecated) replaced by `TBModel.solve_ham()`
- `WFArray.impose_pbc()` (deprecated). Periodic boundary conditions are handled automatically by `Mesh`. Will raise `NotImplementedError` if called. 
- `WFArray.impose_loop()` (deprecated). Periodic boundary conditions are handled automatically by `Mesh`. Will raise `NotImplementedError` if called. 
- `WFArray.solve_on_grid()` (deprecated). Use `WFArray.solve_model()` instead. Will raise `NotImplementedError` if called. 
- `WFArray.solve_on_one_point()` (deprecated). Use `WFArray.solve_model()` instead. Will raise `NotImplementedError` if called.
- `W90.w90_bands_consistency()` (deprecated) renamed to `W90.bands_w90()` for clarity

### Removed 

- Following [SPEC-0](https://scientific-python.org/specs/spec-0000/) (Scientific Python Ecosystem Coordination)
  - Dropped support for Python < 3.12 
  - Dropped support for NumPy < 2.0
  - Dropped support for Matplotlib < 3.9

**Expired deprecations**

- Removed parameter `to_home_supress_warning` in `TBModel.change_nonperiodic_vector()` and `TBModel.make_supercell()` (deprecated since v1.8.0). Default behavior is to only shift orbitals along periodic directions, with a warning sent to the logger if an orbital is outside the home unit cell in a non-periodic direction.

The following functionality has been removed. Users should update their code accordingly.

- Removed `TBModel.reduce_dim()`. This was used fix a particular k-component. `TBModel` is not intended to handle such constraints directly. This should be handled externally. 
- Removed `TBModel.ignore_position_operator_offdiagonal()`. Functionality replaced by `TBModel.assume_position_operator_diagonal` attribute setter. 
- Removed flag `individual_phases` in `WFArray.berry_flux()`. The returned Berry fluxes are always computed with individual phases now. 

## [1.8.0] - 2022-09-20

### Changed
- Updated class `wf_array` to make it easier to store states
  which are not Bloch-like eigenstates.
- Fixed various small issues.

### Added
-  Added new functionality to `wf_array`
    - `solve_on_one_point`
    - `choose_states` 
    - `empty_like`
- Added function change_nonperiodic_vector and changed the way
  `to_home` parameter works.


### Removed
- Removed some functions that were kept for backwards compatibility
    - `berry_curv`
    - `k_path`
    - `tbmodel`
    - `set_sites`
    - `add_hop`.
  
## [1.7.2] - 2017-08-01

### Changed
- Display function now prints hopping distances

### Added
- Added support for deleting orbitals


## [1.7.1] - 2016-12-22

### Added
- Added support for python 3.x in addition to 2.x

## [1.7.0] - 2016-06-07

### Changed
- Cleaned up period boundary condition in the `wf_array` class

### Fixed
- Fixed bug with reduce_dim.  Some hopping terms were not correctly cast as onsite terms.
- Fixed bug in `impose_pbc` when `dim_k` is less than `dim_r`.

### Added
- Added interface with Wannier90 package
- Added support for making bandstructure plots along multi-segment
  paths in the Brillouin zone
- Added support for hybrid Wannier functions.
- Berry curvature in dimensions higher than 2.


## [1.6.2] - 2013-02-25

### Added
- Added support for spinors.
- Added make_supercell method with which one can make arbitrary
  super-cells of the model and also generate slabs with arbitrary
  orientation.
 
## [1.6.1] - 2012-11-15

For the most part, the code should be backward-compatible with version 1.5.
### Changed
- Renamed the code package (previously PyTB) to avoid confusion with
  other acronyms.
- Streamlined the code to be more consistent in naming conventions.
- Made some improvements and extensions to the calculation of Berry
  phases and curvatures.
- Changed the way in which the `impose_pbc` function is used.
- `tb_mode`, `set_onsite`, `set_hop` are named differently but have aliases to names from version 1.5

### Added
- Built a proper python distribution including documentation and an
  improved website.
- Added a more powerful method of setting onsite and hopping parameters.
- Added some additional examples.


### Removed
- Removed `add_wf` function from `wf_array` object and replaced it
  with `[]` operator, and 


## [1.5] - 2012-06-



