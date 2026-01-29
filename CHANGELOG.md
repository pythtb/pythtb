# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [2.0.1] - 2026-01-29

### Fixed
- Fixed bug in `Wannier.project` where the projection procedure improperly reshaped spinful wavefunctions leading to an error when projecting onto trial orbitals.
- Fixed bug in `models.ssh` where the intercell hopping was not set between neighboring unit cells as intended.

## [2.0.0] - 2025-11-11

### Fixed
- Fixed bug in `TBModel._shift_to_home()` where only the last orbital was shifted. This affected the `to_home` flag in `change_nonperiodic_vector()` and `make_supercell()`.

### Improved
- Vectorized code throughout using NumPy for substantial speed improvements
  - `TBModel` initialization, Hamiltonian construction, and diagonalization orders of magnitude faster for large models
  - `WFArray` state storage and manipulation significantly accelerated for large k-meshes
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
- `WFArray` Initialization Changes:
  - Replaced `mesh_arr` parameter with a `Mesh` instance
  - Replaced `model` parameter with a `Lattice` instance
  - Renamed `nsta_arr` parameter to integer `nstates` for clarity
- `TBModel.solve_ham()` (replaces `solve_one()` and `solve_all()`)
  - Changed eigenvalue/eigenvector indexing for vectorized workflows
  - Eigenvalues: shape `(nk, ..., nstate)` (matrix elements last for NumPy compatibility)
  - Eigenvectors for spinless (`nspin=1`): shape `(nk, ..., nstate, nstate)`
  - Eigenvectors for spinful (`nspin=2`): shape `(nk, ..., nstate, norb, 2)`
- `W90.w90_bands_consistency()` (deprecated and renamed to `bands_w90()`)
  - Returned energy array shape changed from `(band, kpts)` to `(kpts, band)`
  - Now consistent with eigenvalue shape from `TBModel.solve_ham()`
  - Aligns with NumPy convention of putting k-points in first axis

**Renaming**
- `TBModel.position_expectation()` parameter renaming
  - Renamed parameter `evec` to `evecs` for clarity
  - Renamed parameter `dir` to `pos_dir` to avoid conflict with built-in Python function `dir()`
- `TBModel.position_matrix()` parameter renaming
  - Renamed parameter `evec` to `evecs` for clarity
  - Renamed parameter `dir` to `pos_dir` to avoid conflict with built-in Python function `dir()`
- `TBModel.position_hwf()` parameter renaming
  - Renamed parameter `evec` to `evecs` for clarity
  - Renamed parameter `dir` to `pos_dir` to avoid conflict with built-in Python function `dir()`
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

### Added
- Published `pythtb` package to [conda-forge](https://anaconda.org/conda-forge/pythtb) for easy installation via `conda install -c conda-forge pythtb`
- Optional TensorFlow backend for linear algebra acceleration on compatible hardware (GPUs/TPUs) in `TBModel` and `WFArray`
  - Enable by passing `use_tensorflow=True` on compatible methods
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

**New functionality in `TBModel`**
- Added support for specifying symbolic strings and callables for parameter-dependent onsite energies and hoppings in parameterized tight-binding models in `TBModel.set_onsite()` and `TBModel.set_hop()`
  - This enables passing kwargs specifying parameter values as scalars or arrays to methods like `TBModel.hamiltonian()` or any other method that uses the tight-binding Hamiltonian (e.g., `TBModel.velocity()`, `TBModel.berry_curvature()`, etc.)

**New methods in `TBModel`**
- Added `TBModel.__str__`: Allows printing a `TBModel` instance using `print(TBModel)`, which prints `TBModel.info()`
- Added `TBModel.info()`: Replaces `display()` for printing model summary
- Added `TBModel.copy()`: Creates a deep copy of the `TBModel` instance
- Added `TBModel.get_lat_vecs()`: Replaces `get_lat()` for clarity
- Added `TBModel.get_orb_vecs()`: Replaces `get_orb()` for clarity
  - Added boolean flag `cartesian` to return orbital vectors in Cartesian coordinates (default `False`)
- Added `TBModel.get_recip_lat()`: Returns reciprocal lattice vectors
- Added `TBModel.add_orb()`: Adds a single orbital to the model
- Added `TBModel.make_finite()`: Convenience function for chaining `cut_piece()` along different directions
- Added `TBModel.solve_ham()`: Replaces `solve_one()` and `solve_all()` for a unified, vectorized diagonalization of Hamiltonians at multiple k-points and parameter values
  - Added `flatten_spin_axis` flag to return spinful eigenvectors in flattened shape for easier processing
  - Added `use_tensorflow` flag to enable optional TensorFlow backend acceleration
  - Added `params` kwargs to pass parameter values for parameterized models
- Added `TBModel.clear_hoppings()`: Removes all hopping terms from the model
- Added `TBModel.clear_onsite()`: Resets all on-site energies to zero
- Added `TBModel.with_parameters()`: Returns model at specific parameter values for parameterized models 
- Added `TBModel.set_parameters()`: Resolves parameterized terms with scalar values
- Added `TBModel.set_shell_hops()`: Bulk setting of n'th nearest-neighbor hoppings for faster model construction
- Added `TBModel.nn_bonds()`: Returns information about nearest-neighbor bonds in the model, useful for manually setting hoppings on shells
- Added `TBModel.hamiltonian()`: constructs Hamiltonians for finite and periodic systems
- Added `TBModel.velocity()`: computes velocity operator $dH/dk$ in orbital basis
- Added `TBModel.quantum_geometric_tensor()`: quantum geometric tensor using Kubo formula
- Added `TBModel.quantum_metric()`: quantum metric tensor from quantum geometric tensor
- Added `TBModel.berry_curvature()`: berry curvature from quantum geometric tensor
- Added `TBModel.chern_number()`: computes Chern number using Berry curvature
- Added `TBModel.axion_angle()`: computes axion angle using 4-curvature integration
- Added `TBModel.local_chern_marker()`: Bianco-Resta formula for real-space Chern marker
- Added `TBModel.plot_bands()`: built-in band structure plotting utility using `matplotlib`
- Added `TBModel.visualize3d()`: interactive 3D visualization for 3D models using `plotly`

**New attributes in `TBModel`**
- Added `TBModel.assume_position_operator_diagonal`: attribute setter to control diagonal approximation for position operator
  - Replaces deprecated `ignore_position_operator_offdiagonal()` method
- Added `TBModel.lattice`: read-only property returning associated `Lattice` instance
- Added `TBModel.nspin`: read-only property returning spinful/spinless status
- Added `TBModel.periodic_dirs`: read-only property returning list of periodic directions
- Added `TBModel.norb`: read-only property returning number of orbitals
- Added `TBModel.nstate`: read-only property returning number of states
- Added `TBModel.dim_r`: read-only property returning real-space dimension
- Added `TBModel.dim_k`: read-only property returning k-space dimension
- Added `TBModel.onsite`: read-only property returning on-site energies as NumPy array
- Added `TBModel.hoppings`: read-only property returning hoppings as list of dictionaries
- Added `TBModel.nhops`: read-only property returning number of hopping terms
- Added `TBModel.spinful`: read-only property returning spinful/spinless status
- Added `TBModel.parameters`: read-only property returning current parameter values for parameterized models
- Added `TBModel.from_w90`: read-only property returning whether model was constructed from Wannier90 data
- Added `TBModel.lat_vecs`: read-only property returning lattice vectors
- Added `TBModel.orb_vecs`: read-only property returning orbital position vectors in reduced coordinates
- Added `TBModel.cell_volume`: read-only property returning unit cell volume
- Added `TBModel.recip_lat_vecs`: read-only property returning reciprocal lattice vectors
- Added `TBModel.recip_volume`: read-only property returning reciprocal unit cell volume

**New methods in `WFArray`**
- Added `WFArray.set_states()`: Sets wavefunction data from NumPy array
- Added `WFArray.remove_states()`: Removes specified states from the `WFArray`
- Added `WFArray.overlap_matrix()`: Computes overlap matrix of the states in the `WFArray` with their nearest neighbors on a `Mesh`.
- Added `WFArray.links()`: Computes the unitary part of the overlap between states and their nearest neighbors in each mesh direction
- Added `WFArray.berry_connection()`: Computes Berry connection from the links between nearest neighbor states in the mesh
- Added `WFArray.wilson_loop()`: Static method that computes the Wilson loop unitary matrix for a loop of states
- Added `WFArray.berry_curvature()`: Computes dimensionful Berry curvature by divinding Berry flux by mesh cell area/volume
- Added `WFArray.chern_number()`: Returns the Chern number for a given plane in the parameter mesh
- Added `WFArray.solve_model()`: Populates `WFArray` with energy eigenstates from a given `TBModel` along the `Mesh`
  - Replaces deprecated `solve_on_grid()` and `solve_on_one_point()` methods
- Added `WFArray.projectors()`: Returns band projectors and optionally their complements as NumPy arrays
- Added `WFArray.states()`: Returns states as a NumPy array, optionally the full Bloch states including phase factors
- Added `WFArray.roll_states_with_pbc()`: Rolls states along a given mesh axis with periodic boundary conditions.
- Added `WFArray.copy()`: Creates a deep copy of the `WFArray` instance
- Added parameter `non_abelian` to `WFArray.berry_flux()` to compute non-Abelian Berry flux for a manifold of states

**New attributes in `WFArray`**
- Added `WFArray.model`: read-only property returning associated `TBModel` instance (if provided with `solve_model()`)
- Added `WFArray.mesh`: read-only property returning associated `Mesh` instance
- Added `WFArray.lattice`: read-only property returning associated `Lattice` instance
- Added `WFArray.filled`: read-only property returning whether states have been populated
- Added `WFArray.wfs`: read-only property returning wavefunction data as NumPy array
- Added `WFArray.energies`: read-only property returning energies as NumPy array (if populated via `solve_model()`)
- Added `WFArray.u_nk`: read-only property returning cell-periodic parts of Bloch states as NumPy array (if populated via `set_states()` or `solve_model()` and system is periodic)
- Added `WFArray.psi_nk`: read-only property returning full Bloch states including phase factors as NumPy array (if populated via `set_states()` or `solve_model()` and system is periodic)
- Added `WFArray.Mmn`: read-only property returning overlap matrices between states and their nearest neighbors on the mesh (if computed via `overlap_matrix()`)
- Added `WFArray.hamiltonian` : read-only property returning Hamiltonian matrices at each mesh point (if populated via `solve_model()`)
- Added `WFArray.shape`: read-only property returning shape of the stored wavefunction array
- Added `WFArray.nstates`: read-only property returning number of states stored
- Added `WFArray.spinful`: read-only property returning spinful/spinless status
- Added `WFArray.nspins`: read-only property returning number of spin components (1 or 2)
- Added `WFArray.norb`: read-only property returning number of orbitals in the underlying `Lattice`
- Added `WFArray.shape_mesh`: number of points along each mesh axis
- Added `WFArray.dim_k`: read-only property returning k-space dimension of the underlying `Lattice`
- Added `WFArray.dim_lambda`: read-only property returning number of swept parameters in the `Mesh`
- Added `WFArray.naxes`: read-only property returning total number of mesh axes
- Added `WFArray.k_points`: read-only property returning k-points stored in the `Meshe` in reduced coordinates as NumPy array
- Added `WFArray.param_points`: read-only property returning parameter points stored in the `Mesh` as NumPy array

**New methods in `W90`**
- Added `W90.bands_w90()`: Renaming of `w90_bands_consistency()` for consistency
  - Returns energy array with shape `(kpts, band)` consistent with `TBModel.solve_ham()`
- Added `W90.bands_qe()`: Reads band structure from Quantum ESPRESSO `prefix_bands.out` output files

### Deprecated

The following methods are deprecated but still functional with backward compatibility wrappers.
These will be removed in a future release.

- `tb_model` class name remains available as alias for `TBModel` 
- `TBModel.display()` (deprecated) renamed to `TBModel.info()` to prevent confusion with visualization. Use `TBModel.info()` or `print(my_model)` instead
- `TBModel.get_lat()` (deprecated) renamed to `TBModel.get_lat_vecs()` for clarity
- `TBModel.get_orb()` (deprecated) renamed to `TBModel.get_orb_vecs()` for clarity
- `TBModel.set_onsite()` and `TBModel.set_hop()` `"reset"` mode (deprecated): use `"set"` instead
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



