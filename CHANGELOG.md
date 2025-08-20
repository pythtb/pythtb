# Changelog

All notable changes to this project will be documented in this file.  

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [2.0.0] - Unreleased

### Added

#### Developer Notes
For a detailed explanation of the changes, see the developer documentation [DEVNOTES.md](https://github.com/sinisacoh/pythtb/blob/v2/dev/DEVNOTES.md).

#### Models
- Added a [folder of example models](https://github.com/sinisacoh/pythtb/blob/v2/pythtb/models) that is importable using, e.g.,
  ```
  from pythtb.models import haldane
  my_model = haldane(delta, t, t2)
  ```
  
#### Examples
- [visualize_3d.py](https://github.com/sinisacoh/pythtb/blob/v2/examples/visualize/visualize_3d.py): demonstrates the 3D visualization feature for `TBModel`
- [ssh.py](https://github.com/sinisacoh/pythtb/blob/v2/examples/ssh/ssh.py): Constructs the ssh model and plots the band structure with a slider to change the intracell hopping. 

#### `TBModel`
- `TBModel.__repr__`: Object representation now displays `rdim`, `kdim`, and `nspin`. 
- `TBModel.__str__`: Allows printing a `TBModel` instance using 
    ```python
    print(TBModel)
    ```
    This calls `TBModel.report()` (formerly `TBModel.display()`) and prints model information.
- `TBModel.get_orb` boolean flag `cartesian`
    - Returns orbital vectors in Cartesian coordinates if `True`, `False` by default.
- `TBModel.hamiltonian`
    - Generates Hamiltonians for both single and multiple k-points.
- `TBModel.solve_ham`:
    - Unified method subsumes previous methods `tb_model.solve_one` and `tb_model.solve_all`
    - Handles single-k-point input automatically and reproduces `solve_one`.
- `TBModel.get_velocity`
    - Computes $dH/dk$ (velocity operator) in the orbital basis.
- `TBModel.berry_curv`
    - Computes Berry curvature from $dH/dk$ elements using the Kubo formula. 
    - Accepts occupied band indices. 
    - Assumes there is a global gap that defines the occupied and unoccupied bands. 
- `TBModel.chern`
    - Returns Chern number for a given set of occupied bands using the Berry curvature from above. 
    - Assumes there is a global gap that defines the occupied and unoccupied bands. 
-  `TBModel.visualize3d`
    - For 3-dimensional tight-binding models, displays a 3d figure of the tight-binding orbitals using `plotly`. 
    - Also prints a legend with the model terms (onsite energies, orbital positions). 
    - The figure can be rotated and zoomed in. When highlighting an orbital or bond, the onsite or hopping terms appear. 
- `TBModel.get_recip_lat`
    - Returns reciprocal lattice vectors.
- Read-only retrieval of core TBModel attributes (e.g., `dim_r`, `dim_k`, `nspin`, `per`, `norb`, `nstate`, `lat`, `orb`, `site_energies`, and `hoppings`) using e.g. `my_model.dim_r` preventing unintended modification of internal model parameters.

#### `WFArray`
- `WFArray.chern_num`
    - Returns the chern number for a given plane in the parameter mesh
- `WFArray.wilson_loop`
    - Computes the Wilson loop unitary matrix for a loop of states
- `WFArray.get_links`
    - Computes the unitary part of the overlap between the mesh of states and their nearest neighbors in each direction
- `WFArray.solve_on_path`
    - Populates a 1D `WFArray` with states diagonalized on a 1D k-path
- `WFArray.get_projectors`
    - Returns the band projectors and optionally their compliment 
- `WFArray.get_bloch_states`
    - When the states populated on a k-mesh, this function applies the $e^{i \mathbf{k} \cdot \mathbf{r}}$ phase factors and returns both the cell-periodic $u_{n\mathbf{k}}$ and the Bloch states $\phi_{n\mathbf{k}}$.
- `WFArray.get_states`
    - Returns the `WFArray` states in NumPy form. 
    - Has an optional flag to flatten the spin axis in cases where the states are spinful
- Properties
    - Added a series of read-only properties for certain attributes

### Changed

- Renamed public classes (see [DEVLOG](https://github.com/sinisacoh/pythtb/blob/v2/dev/DEVLOG.md) for more details)
    - `tb_model` -> `TBModel`
    - `wf_array` -> `WFArray`
    - `w90` -> `W90`
- Examples are now grouped categorically into folders

#### `TBModel`
- `TBModel.solve_all()` and `TBModel.solve_one()` (deprecated, merged into faster `TBModel.solve_ham()`)
    - These methods have been unified into a faster `TBModel.solve_ham()` that utilizes NumPy vectorization for faster diagonalization.
    - The returned eigenvalues and eigenvectors are indexed differently for vectorized workflows (matrix elements go last for NumPy linear algebra operations)
        - Eigenvalues now indexed as `(Nk, n_state)` .
        - Eigenvectors shaped for spinful and spinless cases (see docstring for full details).
            - `n_spin`= 1: (Nk, n_state, n_state) 
            - `n_spin`= 2: (Nk, n_state, n_orb, 2)
            - If finite (no k axis): (n_state, ...) and spin axes are as before
    - Flag in `TBModel.solve_ham()`: `eigvectors` -> `return_eigvecs` for clarity.
- `TBModel.visualize()`
    - Hopping vectors depicted as curved arrows instead of two straight lines at an angle. 
    - Lattice vectors shown as arrows, unit cell delineated with dotted lines. 
    - Arrow transparency scales with hopping magnitude (max element in 2x2 matrix if spinful) to give an idea of the strength of the hopping terms in the model.
- `TBModel.display()` (deprecated, now `TBModel.report()` and `print(TBModel)`):
    - Prints orbital vectors in both Cartesian and reduced units.
- `TBModel.change_nonperiodic_vector()` flag `to_home_supress_warning` has been renamed to `to_home_warning` for brevity and clarity.
    - WARNING: The boolean meaning of the flag also changes, `to_home_warning == True` means a warning _will_ be displayed.
- `TBModel.make_supercell()` flag `to_home_supress_warning` has been renamed to `to_home_warning` for brevity and clarity.
    - WARNING: The boolean meaning of the flag also changes, `to_home_warning == True` means a warning _will_ be displayed.

#### `WFArray`
- `WFArray.berry_flux`
    - Flag renamed: `occ` -> `state_idx`
    - Flag renamed: `dirs` -> `plane`
    - Flag removed: `individual_phases`
        - This flag previously returned the integrated Berry flux in the plane as a function of the remaining parameters. For clarity, it is now up to the user to sum over the proper axes if they want to integrate the Berry flux. The Berry flux will have axes for all parameter directions. 
    - Default behavior change: when `dirs` is unspecified (or `None`) the returned Berry flux will have 2 additional axes (first and second) over all combinations of planes (e.g. berry_flux()[0,1] is the Berry flux in the 0,1 plane)
    - Substantial speed improvements using NumPy vectorization

#### `W90`
- `W90.w90_bands_consistency`
    - Returned energies now have shape (kpts, band) instead of (band, kpts). This matches the shape of the returned eigenvalues in `TBModel.solve_ham`.
 
### Removed 
- Support for Python <3.10 ([SPEC-0](https://scientific-python.org/specs/spec-0000/))
- `setup.py`: migration to `pyproject.toml`.
- `WFArray.berry_flux` flag `individual_phases`

### Deprecated
- `tb_model.solve_one`: Use `TBModel.solve_ham` instead
- `tb_model.solve_all`: Use `TBModel.solve_ham` instead
- `tb_model.display`: Use `TBModel.report` instead
- `reset` flag for `TBModel.set_onsite`: Use `set` instead.
    - Only `set` and `add` parameters retained; `reset` is now merged into `set`.


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



