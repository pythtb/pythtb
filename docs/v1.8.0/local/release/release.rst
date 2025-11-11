Version 1.8.0 (current)
^^^^^^^^^^^^^^^^^^^^^^^

20 September 2022: :download:`pythtb-1.8.0.tar.gz <misc/pythtb-1.8.0.tar.gz>`

* Updated class wf_array to make it easier to store states
  which are not Bloch-like eigenstates.

* Added new functionality to wf_array (solve_on_one_point,
  choose_states, empty_like)

* Added function change_nonperiodic_vector and changed the way
  "to_home" parameter works.

* Fixed various small issues.

* Removed some functions that were kept for backwards compatibility
  (berry_curv, k_path, tbmodel, set_sites, add_hop).
  
Version 1.7.2
^^^^^^^^^^^^^

1 August 2017: :download:`pythtb-1.7.2.tar.gz <misc/pythtb-1.7.2.tar.gz>`

* Added support for deleting orbitals

* Display function now prints hopping distances

Version 1.7.1 
^^^^^^^^^^^^^

22 December 2016: :download:`pythtb-1.7.1.tar.gz <misc/pythtb-1.7.1.tar.gz>`

* Added support for python 3.x in addition to 2.x

Version 1.7.0
^^^^^^^^^^^^^

7 June 2016: :download:`pythtb-1.7.0.tar.gz <misc/pythtb-1.7.0.tar.gz>`

* Added interface with Wannier90 package

* Added support for making bandstructure plots along multi-segment
  paths in the Brillouin zone

* Added support for hybrid Wannier functions.

* Cleaned up period boundary condition in the wf_array class

* Berry curvature in dimensions higher than 2.

* Fixed bug with reduce_dim.  Some hopping terms were not correctly
  casted as onsite terms.

* Fixed bug in impose_pbc when dim_k is less than dim_r.

Version 1.6.2
^^^^^^^^^^^^^

25 February 2013: :download:`pythtb-1.6.2.tar.gz <misc/pythtb-1.6.2.tar.gz>`

* Added support for spinors.

* Added make_supercell method with which one can make arbitrary
  super-cells of the model and also generate slabs with arbitrary
  orientation.

Version 1.6.1
^^^^^^^^^^^^^

15 November 2012: :download:`pythtb-1.6.1.tar.gz <misc/pythtb-1.6.1.tar.gz>`

* Renamed the code package (previously PyTB) to avoid confusion with
  other acronyms.

* Built a proper python distribution including documentation and an
  improved website.

* Streamlined the code to be more consistent in naming conventions.

* Made some improvements and extensions to the calculation of Berry
  phases and curvatures.

* Added a more powerful method of setting onsite and hopping parameters.

* Removed *add_wf* function from *wf_array* object and replaced it
  with *[]* operator, and changed the way in which the *impose_pbc*
  function is used.

* Added some additional examples.

* For the most part, the code should be backward-compatible with version
  1.5. (*tb_model*, *set_onsite*, *set_hop* are named differently but
  have aliases to names from version 1.5).

Version 1.5
^^^^^^^^^^^

4 June 2012: :download:`pytb-1.5.tar.gz <misc/pytb-1.5.tar.gz>`
