# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.0] - 2022-09-20

### Added
- New functionality to `wf_array` class: `solve_on_one_point`, `choose_states`, and `empty_like` methods
- Function `change_nonperiodic_vector` for changing non-periodic lattice vectors
- Enhanced testing suite with tests for all examples
- Silicon files for Wannier90 example
- Support for Read the Docs documentation
- Conda downloads badge and installation information to README
- Examples to website for readthedocs

### Changed
- Updated class `wf_array` to make it easier to store states which are not Bloch-like eigenstates
- Changed the way "to_home" parameter works
- Revised README with installation details and badges
- Enhanced README with installation and dependencies info
- Renamed README.txt to README.md
- Updated license year
- Updated website/publist script
- Changes to documentation for Zenodo repository

### Removed
- Functions kept for backwards compatibility: `berry_curv`, `k_path`, `tbmodel`, `set_sites`, `add_hop`
- Old test files (replaced with new test suite)
- INSTALL file (information moved to README.md)

### Fixed
- Various small issues
- Tests for v1.8
- Typo in reference to PDF
- Updated gitignore for website builds and outputs, fix link in examples

## [1.7.2] - 2017-08-01

### Added
- Support for deleting orbitals
- Display function now prints hopping distances

## [1.7.1] - 2016-12-22

### Added
- Support for Python 3.x in addition to 2.x

## [1.7.0] - 2016-06-07

### Added
- Interface with Wannier90 package
- Support for making bandstructure plots along multi-segment paths in the Brillouin zone
- Support for hybrid Wannier functions
- Berry curvature in dimensions higher than 2

### Changed
- Cleaned up period boundary condition in the wf_array class

### Fixed
- Bug with `reduce_dim` - some hopping terms were not correctly casted as onsite terms
- Bug in `impose_pbc` when dim_k is less than dim_r

## [1.6.2] - 2013-02-25

### Added
- Support for spinors
- `make_supercell` method to create arbitrary super-cells of the model and generate slabs with arbitrary orientation

## [1.6.1] - 2012-11-15

### Changed
- Renamed the code package (previously PyTB) to avoid confusion with other acronyms
- Built a proper python distribution including documentation and an improved website
- Streamlined the code to be more consistent in naming conventions
- Made some improvements and extensions to the calculation of Berry phases and curvatures
- Added a more powerful method of setting onsite and hopping parameters
- Replaced `add_wf` function from `wf_array` object with `[]` operator
- Changed the way `impose_pbc` function is used
- Added some additional examples

### Notes
- For the most part, the code should be backward-compatible with version 1.5
- `tb_model`, `set_onsite`, `set_hop` are named differently but have aliases to names from version 1.5

## [1.5] - 2012-06-04

Initial public release.

[1.8.0]: https://github.com/pythtb/pythtb/releases/tag/v1.8.0
[1.7.2]: https://github.com/pythtb/pythtb/compare/v1.7.1...v1.7.2
[1.7.1]: https://github.com/pythtb/pythtb/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/pythtb/pythtb/compare/v1.6.2...v1.7.0
[1.6.2]: https://github.com/pythtb/pythtb/compare/v1.6.1...v1.6.2
[1.6.1]: https://github.com/pythtb/pythtb/compare/v1.5...v1.6.1
[1.5]: https://github.com/pythtb/pythtb/releases/tag/v1.5
