# Changelog

All notable changes to this project will be documented in this file.  

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [2.0.0] - Unreleased

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

#### New Classes

- **`Lattice`**: New class for handling lattice structures
  - Provides methods for lattice operations previously embedded in `TBModel`
  - Includes `k_path()`, `k_uniform_mesh()`, and `make_supercell()` methods moved from `TBModel`
  
- **`Mesh`**: New class for handling k-point and parameter meshes
  - Provides structured grid generation for k-space sampling
  - Includes `Axis` helper class for defining mesh dimensions
  
- **`Wannier`**: New class for constructing and analyzing Wannier functions
  - Single-shot projection via SVD alignment to trial orbitals
  - Iterative Wannier function localization via maximal localization procedure
  - Quadratic spread evaluation and band disentanglement

#### Developer Notes
For detailed technical explanations, see the developer documentation [DEVNOTES.md](https://github.com/sinisacoh/pythtb/blob/v2/dev/DEVNOTES.md).

#### Models
- Added a [folder of example models](https://github.com/sinisacoh/pythtb/blob/v2/pythtb/models) that is importable using, e.g.,
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
  
#### Examples
- [visualize_3d.py](https://github.com/sinisacoh/pythtb/blob/v2/examples/visualize/visualize_3d.py): Demonstrates 3D visualization feature for `TBModel`
- [ssh.py](https://github.com/sinisacoh/pythtb/blob/v2/examples/ssh/ssh.py): Constructs the SSH model and plots band structure with slider to change intracell hopping
- Examples now organized categorically into folders by dimensionality and topic 

#### `TBModel`

**New Methods:**
- `TBModel.__repr__`: Object representation now displays `rdim`, `kdim`, and `nspin`
- `TBModel.__str__`: Allows printing a `TBModel` instance using `print(TBModel)`, which calls `TBModel.report()`
- `TBModel.hamiltonian()`: Generates Hamiltonians for both single and multiple k-points
- `TBModel.solve_ham()`: Unified method that subsumes `solve_one()` and `solve_all()` with vectorized diagonalization
- `TBModel.get_velocity()`: Computes dH/dk (velocity operator) in the orbital basis
- `TBModel.berry_curv()`: Computes Berry curvature from dH/dk elements using the Kubo formula
  - Accepts occupied band indices
  - Assumes a global gap defining occupied and unoccupied bands
- `TBModel.chern()`: Returns Chern number for a given set of occupied bands using Berry curvature
- `TBModel.visualize3d()`: For 3D tight-binding models, displays an interactive 3D figure using `plotly`
  - Shows orbitals, bonds, and model terms (onsite energies, hopping parameters)
  - Supports rotation, zooming, and interactive highlighting
- `TBModel.get_recip_lat()`: Returns reciprocal lattice vectors
- `TBModel.set_nn_hoppings()`: Bulk setting of nearest-neighbor hoppings for faster model construction
- **Read-only properties**: Core attributes (e.g., `dim_r`, `dim_k`, `nspin`, `per`, `norb`, `nstate`, `lat`, `orb`, `site_energies`, `hoppings`) are now accessible via properties, preventing unintended modification

**Enhanced Methods:**
- `TBModel.get_orb()`: New boolean flag `cartesian` to return orbital vectors in Cartesian coordinates (default `False`)
- `TBModel.visualize()`: Improved visualization
  - Hopping vectors depicted as curved arrows instead of straight lines
  - Lattice vectors shown as arrows with unit cell delineated by dotted lines
  - Arrow transparency scales with hopping magnitude

#### `WFArray`

**New Methods:**
- `WFArray.chern_num()`: Returns the Chern number for a given plane in the parameter mesh
- `WFArray.wilson_loop()`: Computes the Wilson loop unitary matrix for a loop of states
- `WFArray.get_links()`: Computes the unitary part of the overlap between states and their nearest neighbors in each mesh direction
- `WFArray.solve_on_path()`: Populates a 1D `WFArray` with states diagonalized along a 1D k-path
- `WFArray.get_projectors()`: Returns band projectors and optionally their complement
- `WFArray.get_bloch_states()`: For states on a k-mesh, applies e^(ik·r) phase factors and returns both cell-periodic u_{nk} and Bloch states φ_{nk}
- `WFArray.get_states()`: Returns `WFArray` states in NumPy array form
  - Optional flag to flatten spin axis for spinful states
- **Read-only properties**: Added properties for core attributes to prevent unintended modifications

**Performance Improvements:**
- Vectorized implementations using NumPy for substantial speed improvements in:
  - `berry_flux()` computation
  - State manipulations and overlaps
  - Berry phase calculations

### Changed

#### Package Architecture
- **Modularization**: Refactored from single 5000+ line `pythtb.py` file to organized module structure
  - Core classes separated into individual modules for better maintainability
  - Related functionality grouped together (e.g., plotting, utilities)
  - Public API maintained through `pythtb/__init__.py`

#### Class and Method Naming
- Renamed public classes following PEP 8 conventions (see [DEVLOG](https://github.com/sinisacoh/pythtb/blob/v2/dev/DEVLOG.md) for details):
  - `tb_model` → `TBModel` (backward compatibility wrapper provided)
  - `wf_array` → `WFArray` 
  - `w90` → `W90`
- Examples reorganized into categorical folders by dimensionality and physics topic

#### Moved Functionality
- **From `TBModel` to `Lattice`**: 
  - `k_path()`: K-point path generation
  - `k_uniform_mesh()`: Uniform k-space mesh generation
  - `make_supercell()`: Supercell construction
  These methods remain accessible through `TBModel` for backward compatibility but are now primarily `Lattice` methods.

#### Build System
- Migrated from `setup.py` to modern `pyproject.toml` configuration
- Updated packaging metadata and dependencies specification
- Improved development tooling support

#### `TBModel` Method Changes

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

**`display()` → `report()` (deprecated)**
- Renamed to `report()` for consistency
- Now also callable via `print(TBModel)` using `__str__` method
- Prints orbital vectors in both Cartesian and reduced coordinates

**`change_nonperiodic_vector()` and `make_supercell()`**
- Flag `to_home_supress_warning` renamed to `to_home_warning` for clarity
- **Breaking**: Boolean meaning reversed - `to_home_warning=True` means warning WILL be displayed

**Methods moved to `Lattice` class** (still accessible via `TBModel` for compatibility):
- `k_path()`: Generate k-point paths through Brillouin zone
- `k_uniform_mesh()`: Create uniform k-space meshes
- `make_supercell()`: Construct supercells of the model

#### `WFArray` Method Changes

**`berry_flux()` - Enhanced and Optimized**
- **Breaking**: Flag renames for clarity:
  - `occ` → `state_idx`
  - `dirs` → `plane`
- **Breaking**: Removed `individual_phases` flag
  - Previously returned integrated Berry flux as function of remaining parameters
  - Now users must sum over appropriate axes if integration is desired
  - Berry flux now has axes for all parameter directions
- **Breaking**: Default behavior change when `plane` is unspecified (or `None`):
  - Returns Berry flux with 2 additional axes for all plane combinations
  - E.g., `berry_flux()[0,1]` is Berry flux in the (0,1) plane
- Substantial speed improvements using NumPy vectorization

#### `W90` Method Changes

**`w90_bands_consistency()`**
- **Breaking**: Returned energy array shape changed from `(band, kpts)` to `(kpts, band)`
  - Now consistent with eigenvalue shape from `TBModel.solve_ham()`
  - Aligns with NumPy convention of putting k-points in first axis
 
### Removed 

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
- Removed `TBModel` initialization flags `dim_r` and `dim_k`:
  - `dim_r` now automatically inferred from shape of lattice vectors
  - `dim_k` now automatically inferred from number of periodic directions in `per`
  - Reduces redundancy and potential for user error

### Deprecated

The following methods are deprecated but still functional with backward compatibility wrappers:

#### `TBModel` Methods
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
  - Users encouraged to migrate to new PEP 8 compliant names


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
