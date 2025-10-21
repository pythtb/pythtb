# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [2.0.0] - 2025--??-??

### Overview

Version 2.0.0 represents a major refactoring of PythTB with significant architectural changes, new features, and breaking changes. The package has been restructured from a single `pythtb.py` file into a modular package with multiple modules, improving code organization, maintainability, and extensibility.

### Added

#### Package Structure
- **Modular architecture**: Restructured from single `pythtb.py` file to organized `pythtb/` package with separate modules:
  - `tbmodel.py`: Tight-binding model class and methods
  - `wfarray.py`: Wavefunction array class for storing and manipulating quantum states
  - `w90.py`: Wannier90 interface
  - `lattice.py`: New lattice handling class
  - `mesh.py`: New mesh and axis classes for k-point grids
  - `wannier.py`: New Wannier function construction and localization
  - `utils.py`: Utility functions
  - `plotting.py`: Visualization utilities
  - `models/`: Predefined model examples

#### Documentation
- Type hints added throughout the codebase for improved developer experience and IDE support
- Modernized Sphinx-based documentation website, copying over the previous tutorials, and adding some new ones to cover new features.

#### Testing
- Comprehensive unit tests added using `pytest` to cover core functionality and ensure code reliability

#### New Core Classes

- **`Lattice`**  
  Handles lattice geometry and reciprocal operations.  
  - Encapsulates methods previously embedded in `TBModel`  
  - Provides `k_path()`, `k_uniform_mesh()`, `make_supercell()`  
  - Used by `TBModel` and `WFArray`

- **`Mesh`**  
  Defines structured grids in k-space or parameter space.  
  - Supports arbitrary dimensions and mixed (k, λ) meshes  
  - Includes `Axis` helper class for labeled mesh axes  
  - Used by `WFArray` for consistent data mapping

- **`Wannier`**  
  Framework for constructing and analyzing Wannier functions.  
  - SVD projection onto trial orbitals  
  - Iterative maximal localization/disentanglement  
  - Spread minimization, center computation, visualization

#### Models
- Added a folder of example models (`pythtb/models/`) that is importable using, e.g.,
  ```python
  from pythtb.models import haldane
  my_model = haldane(delta, t, t2)
  ```
- Available models include:
  - `haldane`: Haldane model on honeycomb lattice
  - `ssh`: Su-Schrieffer-Heeger (SSH) chain
  - `graphene`: Graphene tight-binding model
  - `kane_mele`: Kane-Mele model with spin-orbit coupling
  - `fu_kane_mele`: Fu-Kane-Mele 3D topological insulator
  - `checkerboard`: Checkerboard lattice model

#### `TBModel`

##### New Methods:
- `TBModel.__repr__`: Object representation now displays `rdim`, `kdim`, and `nspin`
- `TBModel.__str__`: Allows printing a `TBModel` instance using `print(TBModel)`, which calls `TBModel.report()`
- `TBModel.hamiltonian()`: Generates Hamiltonians for both single and multiple k-points
- `TBModel.solve_ham()`: Unified method that subsumes `solve_one()` and `solve_all()` with vectorized diagonalization
- `TBModel.velocity()`: Computes $dH/dk$ (velocity operator) in the orbital basis
- `TBModel.berry_curv()`: Computes Berry curvature from $dH/dk$ elements using the Kubo formula
  - Accepts occupied band indices
  - Assumes a global gap defining occupied and unoccupied bands
- `TBModel.chern()`: Returns Chern number for a given set of occupied bands using Berry curvature
- `TBModel.local_chern_marker()`: Bianco-Resta formula for real-space Chern marker
- `TBModel.visualize3d()`: For 3D tight-binding models, displays an interactive 3D figure using `plotly`
  - Shows orbitals, bonds, and model terms (onsite energies, hopping parameters)
  - Supports rotation, zooming, and interactive highlighting
- `TBModel.get_recip_lat()`: Returns reciprocal lattice vectors
- `TBModel.set_nn_hoppings()`: Bulk setting of nearest-neighbor hoppings for faster model construction
- **Read-only properties**: Core attributes (e.g., `dim_r`, `dim_k`, `nspin`, `spinful`, `per`, `norb`, `nstate`, `lat`, `orb`, `site_energies`, `hoppings`) are now accessible via properties, preventing unintended modification

##### Performance Improvements:
- Vectorized implementations using NumPy for substantial speed improvements in:
  - `hamiltonian()` and `velocity()` construction
  - `solve_ham()` diagonalization
  - `berry_curv()` computation

##### Enhanced Methods:
- `TBModel.get_orb()`: New boolean flag `cartesian` to return orbital vectors in Cartesian coordinates (default `False`)

#### `WFArray`

##### New Methods:
- `WFArray.chern_num()`: Returns the Chern number for a given plane in the parameter mesh
- `WFArray.wilson_loop()`: Computes the Wilson loop unitary matrix for a loop of states
- `WFArray.get_links()`: Computes the unitary part of the overlap between states and their nearest neighbors in each mesh direction
- `WFArray.solve_on_path()`: Populates a 1D `WFArray` with states diagonalized along a 1D k-path
- `WFArray.get_projectors()`: Returns band projectors and optionally their complement
- `WFArray.get_bloch_states()`: For states on a k-mesh, applies $e^{ik·r}$ phase factors and returns both cell-periodic $u_{nk}$ and Bloch states $\phi_{nk}$
- `WFArray.get_states()`: Returns `WFArray` states in NumPy array form
  - Optional flag to flatten spin axis for spinful states

##### Read-only properties: Added properties for core attributes to prevent unintended modifications

##### Performance Improvements:
- Vectorized implementations using NumPy for substantial speed improvements in:
  - `berry_flux()` computation
  - State manipulations and overlaps
  - Berry phase calculations

### Changed

#### Build System
- Migrated from `setup.py` to modern `pyproject.toml` configuration
- Updated packaging metadata and dependencies specification
- Improved development tooling support

#### Class and Method Naming
- Renamed public classes following PEP 8 conventions:
  - `tb_model` → `TBModel` (backward compatibility wrapper provided)
  - `wf_array` → `WFArray` 
  - `w90` → `W90`

#### `TBModel` Class Changes

**Initialization Changes** - `__init__()`
- **Breaking**: Replaced `lat`, `orb`, and `per` parameters with a single `Lattice` instance
  - Users must now create a `Lattice` object to define lattice geometry and periodicity
  - This decouples lattice handling from `TBModel`, allowing reuse of `Lattice` objects across multiple models
- **Breaking**: Replaced `dim_r` and `dim_k` parameters with automatic inference from the `Lattice` object
  - Users no longer need to specify these dimensions explicitly
- **Breaking**: Replaced `nspin` parameter with `spinful` boolean flag
  - `spinful=True` indicates spinful (2-component spinors); `False` indicates spinless (1-component)
  - Improves clarity
- To initialize, the user must now provide:
  - `lattice`: a `Lattice` instance defining the lattice geometry and periodicity
  - `spinful`: boolean indicating whether the model is spinful

**`solve_ham()` - Unified Diagonalization (replaces `solve_one()` and `solve_all()`)**
- Merged `solve_one()` and `solve_all()` into single optimized method
- Utilizes NumPy vectorization for significantly faster diagonalization
- Automatically handles single k-point or multiple k-points
- **Breaking**: Changed eigenvalue/eigenvector indexing for vectorized workflows:
  - Eigenvalues: shape `(nk, nstate)` (matrix elements last for NumPy compatibility)
  - Eigenvectors for spinless (`nspin=1`): shape `(nk, nstate, nstate)`
  - Eigenvectors for spinful (`nspin=2`): shape `(nk, nstate, norb, 2)`
  - For finite systems (no k-axis): `(nstate, ...)` with spin axes as before
- Renamed parameter: `eigvectors` → `return_eigvecs` for clarity

**`visualize()` - Enhanced 2D Visualization**
- Hopping vectors now depicted as curved arrows instead of two straight lines at an angle
- Lattice vectors shown as arrows with unit cell outlined by dotted lines
- Arrow transparency scales with hopping magnitude (shows relative strengths visually)
- For spinful models, uses maximum element of 2×2 hopping matrix for scaling

**`display()` (deprecated) → `report()`**
- Renamed `display` to `report()` to prevent confusion with visualization 
- Now also callable via `print(TBModel)` using `__str__` method
- Prints orbital vectors in both Cartesian and reduced coordinates

**Methods moved to `Lattice` class** (still accessible via `TBModel` for compatibility):
- `k_path()`: Generate k-point paths through Brillouin zone
- `k_uniform_mesh()`: Create uniform k-space meshes
- `make_supercell()`: Construct supercells of the model

**`position_expectation()` - Parameter renaming**
- Renamed parameter: `evec` → `evecs` for clarity

**`position_matrix()` - Parameter renaming**
- Renamed parameter: `evec` → `evecs` for clarity

**`hwf_centers()` - Parameter renaming**
- Renamed parameter: `evec` → `evecs` for clarity

#### `WFArray` Method Changes

**`__init__()` - Initialization Changes**
- **Breaking**: Replaced `mesh_arr` parameter with a `Mesh` instance
  - Users must now create a `Mesh` object to define k/parameter grids
- **Breaking**: Replaced `model` parameter with a `Lattice` instance
  - Users must now create a `Lattice` object to define lattice geometry
  - This decouples `WFArray` from `TBModel`, allowing storage of arbitrary states, even if they don't correspond to a specific tight-binding model. `TBModel` can still be used to populate the `WFArray` with its energy eigenstates via `solve_model()` or external diagonalization.
- Renamed `nsta_arr` parameter to integer `nstates` for clarity
- To initialize, the user must now provide:
  - `lattice`: a `Lattice` instance defining the lattice geometry
  - `mesh`: a `Mesh` instance defining the k/parameter grid
  - `nstates`: integer number of states per mesh point

**`berry_flux()` - Enhanced and Optimized**
- **Breaking**: Flag renames for clarity:
  - `occ` → `state_idx`: band indices need not be occupied 
  - `dirs` → `plane`: only accepts 2-element tuples defining planes
- **Breaking**: Removed `individual_phases` flag
  - Previously returned integrated Berry flux as function of remaining parameters
  - Now users must sum over appropriate axes if integration is desired, or call Chern number method
- **Breaking**: Default behavior change when `plane` is unspecified (or `None`):
  - Returns Berry flux with 2 additional axes for all plane combinations
  - E.g., `berry_flux()[0,1]` is Berry flux in the (0,1) plane
- Substantial speed improvements using NumPy vectorization

#### `W90` Method Changes

**`w90_bands_consistency()` renamed to `w90_bands()`**
- **Breaking**: Returned energy array shape changed from `(band, kpts)` to `(kpts, band)`
  - Now consistent with eigenvalue shape from `TBModel.solve_ham()`
  - Aligns with NumPy convention of putting k-points in first axis

### Fixed
- Fixed bug in `TBModel._shift_to_home()` where only the last orbital was shifted. This affected the `to_home` flag in `change_nonperiodic_vector()` and `make_supercell()`.

### Deprecated

The following methods are deprecated but still functional with backward compatibility wrappers:

#### `TBModel` Methods
- `display()`: Use `TBModel.report()` or `print(my_model)` instead
  - New naming is more intuitive, less likely to be confused with visualization
- `get_lat()`: Renamed to `get_lat_vecs()` for clarity
- `get_orb()`: Renamed to `get_orb_vecs()` for clarity
- `reset` flag in `set_onsite()`: Use `set` instead
  - Only `set` and `add` modes retained
  - `reset` functionality merged into `set` for simplicity
- `reset` flag in `set_hop()`: Use `set` instead
  - Only `set` and `add` modes retained
  - `reset` functionality merged into `set` for simplicity
- `solve_one()`: Use `TBModel.solve_ham()` instead
  - `solve_ham()` automatically handles single k-points
- `solve_all()`: Use `TBModel.solve_ham()` instead
  - `solve_ham()` provides vectorized, faster diagonalization
- `display()`: Use `TBModel.report()` or `print(my_model)` instead
  - New naming is more intuitive

#### `TBModel` Parameters  
- `reset` flag in `set_onsite()`: Use `set` instead
  - Only `set` and `add` modes retained
  - `reset` functionality merged into `set` for simplicity

#### Backward Compatibility
- Old class names (`tb_model`, `wf_array`, `w90`) remain available as aliases
  - Allows existing code to work without modification
  - Users encouraged to migrate to new PEP 8 compliant name

### Removed 

#### TBModel Methods
- `TBModel.reduce_dim()`: This would fix a particular k component. However, `TBModel` is not intended to handle such constraints directly. This should be handeled externally or by using a cutom `Mesh`. 
- Flag `to_home_supress_warning` in `change_nonperiodic_vector()` and `make_supercell()`: previously deprecated in v1.8.0, now fully removed. Default behavior is to only shift orbitals along periodic directions, with a warning sent to the logger if an orbital is outside the home unit cell in a non-periodic direction.

#### Python Version Support
- **Breaking**: Dropped support for Python <3.10 
  - Following [SPEC-0](https://scientific-python.org/specs/spec-0000/) (Scientific Python Ecosystem Coordination)
  - Allows use of modern Python features (structural pattern matching, improved type hints, etc.)

#### Build System
- Removed `setup.py` in favor of `pyproject.toml`
  - Modern, declarative build configuration
  - Better tool integration and dependency management

#### API Cleanup
- Removed `WFArray.berry_flux()` flag `individual_phases` (see Changed section)
- Removed `TBModel` initialization flags `dim_r`, `dim_k`, `lat`, `orb` and `per`:
  - These are all inferred from the `Lattice` object passed during initialization
- Removed `WFArray` initialization flags `model`, `mesh_arr`, 


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

## [1.7.0] - 2916-06-07

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



