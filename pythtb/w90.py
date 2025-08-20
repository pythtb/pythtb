import numpy as np
from .utils import _cart_to_red, _red_to_cart
from .tb_model import TBModel

__all__ = ["W90"]

class W90:
    r"""Interface to Wannier90
    
    This class of the PythTB package imports tight-binding model
    parameters from an output of a `Wannier90 <http://www.wannier.org>`_ code.
    Upon instantiation, this class will read in the entire Wannier90 output.
    To create :class:`pythtb.TBModel` object user needs to call
    :func:`pythtb.w90.model`.

    The `Wannier90 <http://www.wannier.org>`_ code is a
    post-processing tool that takes as an input electron wavefunctions
    and energies computed from first-principles using any of the
    following codes: Quantum-Espresso (PWscf), AbInit, SIESTA, FLEUR,
    Wien2k, VASP. As an output Wannier90 will create files that
    contain parameters for a tight-binding model that exactly
    reproduces the first-principles calculated electron band
    structure.

    The interface from Wannier90 to PythTB will use only the following
    files created by Wannier90:

    - *prefix*.win
    - *prefix*\_hr.dat
    - *prefix*\_centres.xyz
    - *prefix*\_band.kpt (optional)
    - *prefix*\_band.dat (optional)

    The first file (*prefix*.win) is an input file to Wannier90 itself. This
    file is needed so that PythTB can read in the unit cell vectors.

    To correctly create the second and the third file (*prefix*\_hr.dat and
    *prefix*\_centres.dat) one needs to include the following flags in the win
    file::

       write_hr = True
       write_xyz = True
       translate_home_cell = False

    These lines ensure that *prefix*\_hr.dat and *prefix*\_centres.dat
    are written and that the centers of the Wannier functions written
    in the *prefix*\_centres.dat file are not translated to the home
    cell. The *prefix*\_hr.dat file contains the onsite and hopping
    terms.

    The final two files (*prefix*\_band.kpt and *prefix*\_band.dat)
    are optional. Please see documentation of function
    :func:`pythtb.w90.w90_bands_consistency` for more detail.

    Parameters
    ----------
    path : str
        Relative path to the folder that contains Wannier90
        files. These are *prefix*.win, *prefix*\_hr.dat,
        *prefix*\_centres.dat and optionally *prefix*\_band.kpt and
        *prefix*\_band.dat.

    prefix : str
        This is the prefix used by `Wannier90` code.
        Typically the input to the `Wannier90` code is name *prefix*.win.

    See Also
    --------
    :ref:`w90-quick-nb`
    :ref:`w90-nb`

    Notes
    -----
    Units used throught this interface with Wannier90 are
    electron-volts (eV) and Angstroms.

    .. warning::
        So far we have only tested Wannier90 version 2.0.1.

    .. warning:: 
        The user needs to make sure that the Wannier functions
        computed using Wannier90 code are well localized. Otherwise the
        tight-binding model may not accurately interpolate the band
        structure. To ensure that the Wannier functions are well
        localized it is often enough to check that the total spread at
        the beginning of the minimization procedure (first total spread
        printed in .wout file) is not more than 20% larger than the
        total spread at the end of the minimization procedure. If those
        spreads differ by much more than 20% user needs to specify
        better initial projection functions.

    .. warning::
        The interpolation is only exact within the frozen energy window
        of the disentanglement procedure.

    .. warning:: 
        So far PythTB assumes that the position operator is
        diagonal in the tight-binding basis. This is discussed in the
        :download:`notes on tight-binding formalism
        </misc/pythtb-formalism.pdf>` in Eq. 2.7.,
        :math:`\langle\phi_{{\bf R} i} \vert {\bf r} \vert \phi_{{\bf
        R}' j} \rangle = ({\bf R} + {\bf t}_j) \delta_{{\bf R} {\bf R}'}
        \delta_{ij}`. However, this relation does not hold for Wannier
        functions! Therefore, if you use tight-binding model derived
        from this class in computing Berry-like objects that involve
        position operator such as Berry phase or Berry flux, you would
        not get the same result as if you computed those objects
        directly from the first-principles code! Nevertheless, this
        approximation does not affect other properties such as band
        structure dispersion.


    Examples
    --------
    Read Wannier90 from folder called *example_a*
    This assumes that that folder contains files "silicon.win" (and so on)

    >>> silicon = w90("example_a", "silicon")
    """

    def __init__(self, path, prefix):
        # store path and prefix
        self.path = path
        self.prefix = prefix

        # read in lattice_vectors
        f = open(self.path + "/" + self.prefix + ".win", "r")
        ln = f.readlines()
        f.close()
        # get lattice vector
        self.lat = np.zeros((3, 3), dtype=float)
        found = False
        for i in range(len(ln)):
            sp = ln[i].split()
            if len(sp) >= 2:
                if sp[0].lower() == "begin" and sp[1].lower() == "unit_cell_cart":
                    # get units right
                    if ln[i + 1].strip().lower() == "bohr":
                        pref = 0.5291772108
                        skip = 1
                    elif ln[i + 1].strip().lower() in ["ang", "angstrom"]:
                        pref = 1.0
                        skip = 1
                    else:
                        pref = 1.0
                        skip = 0
                    # now get vectors
                    for j in range(3):
                        sp = ln[i + skip + 1 + j].split()
                        for k in range(3):
                            self.lat[j, k] = float(sp[k]) * pref
                    found = True
                    break
        if not found:
            raise Exception("Unable to find unit_cell_cart block in the .win file.")

        # read in hamiltonian matrix, in eV
        f = open(self.path + "/" + self.prefix + "_hr.dat", "r")
        ln = f.readlines()
        f.close()
        
        # get number of wannier functions
        self.num_wan = int(ln[1])
        # get number of Wigner-Seitz points
        num_ws = int(ln[2])
        # get degenereacies of Wigner-Seitz points
        deg_ws = []
        for j in range(3, len(ln)):
            sp = ln[j].split()
            for s in sp:
                deg_ws.append(int(s))
            if len(deg_ws) == num_ws:
                last_j = j
                break
            if len(deg_ws) > num_ws:
                raise Exception("Too many degeneracies for WS points!")
        deg_ws = np.array(deg_ws, dtype=int)
        # now read in matrix elements
        # Convention used in w90 is to write out:
        # R1, R2, R3, i, j, ham_r(i,j,R)
        # where ham_r(i,j,R) corresponds to matrix element < i | H | j+R >
        self.ham_r = {}  # format is ham_r[(R1,R2,R3)]["h"][i,j] for < i | H | j+R >
        ind_R = 0  # which R vector in line is this?
        for j in range(last_j + 1, len(ln)):
            sp = ln[j].split()
            # get reduced lattice vector components
            ham_R1 = int(sp[0])
            ham_R2 = int(sp[1])
            ham_R3 = int(sp[2])
            # get Wannier indices
            ham_i = int(sp[3]) - 1
            ham_j = int(sp[4]) - 1
            # get matrix element
            ham_val = float(sp[5]) + 1.0j * float(sp[6])
            # store stuff, for each R store hamiltonian and degeneracy
            ham_key = (ham_R1, ham_R2, ham_R3)
            if ham_key not in self.ham_r:
                self.ham_r[ham_key] = {
                    "h": np.zeros((self.num_wan, self.num_wan), dtype=complex),
                    "deg": deg_ws[ind_R],
                }
                ind_R += 1
            self.ham_r[ham_key]["h"][ham_i, ham_j] = ham_val

        # check if for every non-zero R there is also -R
        for R in self.ham_r:
            if not (R[0] == 0 and R[1] == 0 and R[2] == 0):
                found_pair = False
                for P in self.ham_r:
                    if not (R[0] == 0 and R[1] == 0 and R[2] == 0):
                        # check if they are opposite
                        if R[0] == -P[0] and R[1] == -P[1] and R[2] == -P[2]:
                            if found_pair:
                                raise Exception("Found duplicate negative R!")
                            found_pair = True
                if not found_pair:
                    raise Exception("Did not find negative R for R = " + R + "!")

        # read in wannier centers
        f = open(self.path + "/" + self.prefix + "_centres.xyz", "r")
        ln = f.readlines()
        f.close()
        # Wannier centers in Cartesian, Angstroms
        xyz_cen = []
        for i in range(2, 2 + self.num_wan):
            sp = ln[i].split()
            if sp[0] == "X":
                tmp = []
                for j in range(3):
                    tmp.append(float(sp[j + 1]))
                xyz_cen.append(tmp)
            else:
                raise Exception("Inconsistency in the centres file.")
        self.xyz_cen = np.array(xyz_cen, dtype=float)
        # get orbital positions in reduced coordinates
        self.red_cen = _cart_to_red(
            (self.lat[0], self.lat[1], self.lat[2]), self.xyz_cen
        )

    def model(
        self,
        zero_energy=0.0,
        min_hopping_norm=None,
        max_distance=None,
        ignorable_imaginary_part=None,
        ) -> TBModel:
        r"""Get TBModel associated with this Wannier90 calculation.

        This function returns :class:`pythtb.TBModel` object that can
        be used to interpolate the band structure at arbitrary
        k-point, analyze the wavefunction character, etc.

        The tight-binding basis orbitals in the returned object are
        maximally localized Wannier functions as computed by
        Wannier90. Locations of the orbitals in the returned
        :class:`pythtb.TBModel` object are the centers of
        the Wannier functions computed by Wannier90.

        Parameters
        ----------

        zero_energy : float
            Sets the zero of the energy in the band structure. 
            This value is typically set to the Fermi level
            computed by the density-functional code (or to the top of the valence band). 
            Units are electron-volts.

        min_hopping_norm : float
            Hopping terms read from Wannier90 with complex norm less than
            *min_hopping_norm* will not be included in the returned
            tight-binding model. This parameters is specified in
            electron-volts. By default all terms regardless of their
            norm are included.

        max_distance : float
            Hopping terms from site *i* to site *j+R* will be ignored if
            the distance from orbital *i* to *j+R* is larger than
            *max_distance*. This parameter is given in Angstroms.
            By default all terms regardless of the distance are included.

        ignorable_imaginary_part : float
            The hopping term will be assumed to be exactly real if the
            absolute value of the imaginary part as computed by Wannier90
            is less than *ignorable_imaginary_part*. By default imaginary
            terms are not ignored. Units are again eV.

        Returns
        -------
        tb : :class:`pythtb.TBModel`
            The :class:`pythtb.TBModel` that can be used to
            interpolate Wannier90 band structure to an arbitrary k-point as well
            as to analyze the character of the wavefunctions. 

        Notes
        -----
        The character of the maximally localized Wannier functions is
        not exactly the same as that specified by the initial
        projections. The orbital character of the Wannier functions can be 
        inferred either from the *projections* block in the *prefix*.win or 
        from the *prefix*.nnkp file.

        One way to ensure that the Wannier functions are as close to
        the initial projections as possible is to first choose a good set
        of initial projections (for these initial and final spread should
        not differ more than 20%) and then perform another Wannier90 run
        setting *num_iter=0* in the *prefix*.win file.

        The tight-binding model returned by this function is only as good as
        the input from Wannier90. In particular, the choice of initial
        projections can have a significant impact on the quality of the
        resulting Wannier functions. It is recommended to experiment with
        different sets of initial projections and to carefully analyze the
        resulting Wannier functions to ensure that they are physically
        meaningful.

        The number of spin components is always set to 1, even if the
        underlying DFT calculation includes spin.  Please refer to the
        *projections* block or the *prefix*.nnkp file to see which
        orbitals correspond to which spin.

        Examples
        --------
        Get `TBModel` with all hopping parameters

        >>> my_model = silicon.model()

        Simplified model that contains only hopping terms above 0.01 eV

        >>> my_model_simple = silicon.model(min_hopping_norm=0.01)
        >>> my_model_simple.display()

        """

        # make the model object
        tb = TBModel(3, 3, self.lat, self.red_cen)

        # remember that this model was computed from w90
        tb._assume_position_operator_diagonal = False

        # -------------------------
        # Onsites (vectorized)
        # -------------------------
        # Divide by degeneracy only once and assert onsite is (numerically) real
        hr0 = self.ham_r[(0, 0, 0)]
        deg0 = float(hr0["deg"])  # scalar
        onsite = (hr0["h"].diagonal() / deg0).real
        # sanity check: imaginary part should be tiny
        if np.max(np.abs(np.imag(np.diag(hr0["h"]) / deg0))) > 1.0e-9:
            raise Exception("Onsite terms should be real!")
        tb.set_onsite(onsite - zero_energy)

        # -------------------------
        # Hoppings (vectorized per R)
        # -------------------------
        # Precompute for speed
        xyz_cen = self.xyz_cen  # (num_wan, 3), Cartesian Angstroms
        num_wan = self.num_wan
        lat_tuple = (self.lat[0], self.lat[1], self.lat[2])

        # Helper to decide if we should process an R (to avoid double counting)
        def _use_R(R):
            r1, r2, r3 = R
            if r1 != 0:
                return r1 > 0
            if r2 != 0:
                return r2 > 0
            return r3 > 0

        for R, blk in self.ham_r.items():
            Hr = blk["h"]
            deg = float(blk["deg"])  # scalar

            # Onsite block already handled; keep only off-diagonal pairs here.
            if R == (0, 0, 0):
                avoid_diagonal = True
                use_this_R = True
            else:
                avoid_diagonal = False
                use_this_R = _use_R(R)

            if not use_this_R:
                continue

            # Convert reduced R to Cartesian vector once per R
            vecR = _red_to_cart(lat_tuple, [R])[0]  # (3,)

            # Build full i,j grids in one shot
            # delta_ijR = -vec_i + vec_j + vecR
            #   vec_i shape (num_wan, 1, 3); vec_j shape (1, num_wan, 3)
            delta = (-xyz_cen[:, None, :] + xyz_cen[None, :, :] + vecR[None, None, :])
            dist = np.linalg.norm(delta, axis=2)  # (num_wan, num_wan)

            # Start from all pairs; mask away those we don't keep
            keep = np.ones((num_wan, num_wan), dtype=bool)

            # Prevent double counting within the same R-block
            if avoid_diagonal:
                # original logic: skip j <= i when R == (0,0,0)
                iu = np.triu_indices(num_wan, k=1)
                keep[:] = False
                keep[iu] = True
            else:
                # we allow i,j all; symmetry handled by _use_R half-space rule
                # (still skip exact onsite i==j when R!=0 to match original behavior)
                np.fill_diagonal(keep, False)


            # Apply max_distance cutoff (if provided)
            if max_distance is not None:
                keep &= (dist <= max_distance)

            # Pull matrix elements divided by degeneracy once
            Ham = Hr / deg  # (num_wan, num_wan)

            # Apply min_hopping_norm filter
            if min_hopping_norm is not None:
                keep &= (np.abs(Ham) >= min_hopping_norm)

            # If nothing to keep for this R, continue early
            if not np.any(keep):
                continue

            # Optionally zero-out tiny imaginary parts before insertion
            if ignorable_imaginary_part is not None:
                small_imag = np.abs(Ham.imag) < ignorable_imaginary_part
                Ham = Ham.real + 0.0j + 1j * Ham.imag
                Ham[small_imag] = Ham.real[small_imag] + 0.0j

            # Finally, emit kept hoppings
            ii, jj = np.nonzero(keep)
            for i, j in zip(ii, jj):
                tb.set_hop(Ham[i, j], i, j, list(R))

        return tb
    
    def dist_hop(self):
        r"""Get distances and hopping terms of Hamiltonian in Wannier basis.

        This function returns all hopping terms (from orbital *i* to
        *j+R*) as well as the distances between the *i* and *j+R*
        orbitals. For well localized Wannier functions hopping term
        should decay exponentially with distance.

        Returns
        -------
        dist : np.ndarray
            Distances between Wannier function centers (*i* and *j+R*) in Angstroms.

        ham : np.ndarray
            Corresponding hopping terms in eV.

        Notes
        -----
        This function can be used to help determine the *min_hopping_norm*
        and *max_distance* parameters in the :func:`pythtb.w90.model` function
        call.

        Examples
        --------
        Get distances and hopping terms

        >>> (dist, ham) = silicon.dist_hop()

        Plot logarithm of the hopping term as a function of distance

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.scatter(dist, np.log(np.abs(ham)))
        >>> fig.savefig("localization.pdf")

        """

        ret_ham = []
        ret_dist = []
        for R in self.ham_r:
            # treat diagonal terms differently
            if R[0] == 0 and R[1] == 0 and R[2] == 0:
                avoid_diagonal = True
            else:
                avoid_diagonal = False

            # get R vector
            vecR = _red_to_cart((self.lat[0], self.lat[1], self.lat[2]), [R])[0]
            for i in range(self.num_wan):
                vec_i = self.xyz_cen[i]
                for j in range(self.num_wan):
                    vec_j = self.xyz_cen[j]
                    # diagonal terms
                    if not (avoid_diagonal and i == j):

                        # divide the matrix element from w90 with the degeneracy
                        ret_ham.append(
                            self.ham_r[R]["h"][i, j] / float(self.ham_r[R]["deg"])
                        )

                        # get distance between orbitals
                        ret_dist.append(
                            np.sqrt(
                                np.dot(-vec_i + vec_j + vecR, -vec_i + vec_j + vecR)
                            )
                        )

        return (np.array(ret_dist), np.array(ret_ham))

    def shells(self, num_digits=2):
        r"""Get all shells of distances between Wannier function centers.

        This is one of the diagnostic tools that can be used to help
        in determining *max_distance* parameter in
        :func:`pythtb.w90.model` function call.

        Parameters
        ----------
        num_digits : int
            Distances will be rounded up to these many digits. Default value is 2.

        Returns
        -------
        shells : list
            All distances between all Wannier function centers (*i* and *j+R*) in Angstroms.

        Examples
        --------
        Print all shells

        >>> print(silicon.shells())
        """

        shells = []
        for R in self.ham_r:
            # get R vector
            vecR = _red_to_cart((self.lat[0], self.lat[1], self.lat[2]), [R])[0]
            for i in range(self.num_wan):
                vec_i = self.xyz_cen[i]
                for j in range(self.num_wan):
                    vec_j = self.xyz_cen[j]
                    # get distance between orbitals
                    dist_ijR = np.sqrt(
                        np.dot(-vec_i + vec_j + vecR, -vec_i + vec_j + vecR)
                    )
                    # round it up
                    shells.append(round(dist_ijR, num_digits))

        # remove duplicates and sort
        shells = np.sort(list(set(shells)))

        return shells

    def w90_bands_consistency(self):
        r"""Read interpolated band structure from Wannier90 output files.

        .. versionchanged:: 2.0.0
            Returned energies now have axes `(kpts, band)` instead of `(band, kpts)`.

        This function reads in band structure as interpolated by
        Wannier90. Please note that this is not the same as the band
        structure calculated by the underlying DFT code. The two will
        agree only on the coarse set of k-points that were used in
        Wannier90 generation.

        The code assumes that the following files were generated by
        Wannier90,

          - *prefix*\_band.kpt
          - *prefix*\_band.dat

        These files will be generated only if the *prefix*.win file
        contains the *kpoint_path* block.

        Returns
        -------

        kpts : array
            k-points in reduced coordinates used in the
            interpolation in Wannier90 code. The format of *kpts* is
            the same as the one used by the input to
            :func:`pythtb.TBModel.solve_all`.

        ene : array
            Energies interpolated by Wannier90 in eV. Format is ``ene[kpt,band]``.

        Notes
        -----
        The purpose of this function is to compare the interpolation
        in Wannier90 with that in PythTB. If no terms were ignored in
        the call to :func:`pythtb.w90.model` then the two should
        be exactly the same (up to numerical precision). Otherwise
        one should expect deviations. However, if one carefully
        chooses the cutoff parameters in :func:`pythtb.w90.model`
        it is likely that one could reproduce the full band-structure
        with only few dominant hopping terms. Please note that this
        tests only the eigenenergies, not eigenvalues (wavefunctions).

        Examples
        --------
        Get band structure from `Wannier90`

        >>> (w90_kpt, w90_evals) = silicon.w90_bands_consistency()

        Get simplified model

        >>> my_model_simple = silicon.model(min_hopping_norm=0.01)

        Solve simplified model on the same k-path as in `Wannier90`

        >>> evals = my_model.solve_ham(w90_kpt)

        Now plot the comparison of the two
        
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> for i in range(evals.shape[0]):
        >>>     ax.plot(range(evals.shape[1]), evals[i], "r-", zorder=-50)
        >>> for i in range(w90_evals.shape[0]):
        >>>     ax.plot(range(w90_evals.shape[1]), w90_evals[i], "k-", zorder=-100)
        >>> fig.savefig("comparison.pdf")

        """

        # read in kpoints in reduced coordinates
        kpts = np.loadtxt(self.path + "/" + self.prefix + "_band.kpt", skiprows=1)
        # ignore weights
        kpts = kpts[:, :3]

        # read in energies
        ene = np.loadtxt(self.path + "/" + self.prefix + "_band.dat")
        # ignore kpath distance
        ene = ene[:, 1]
        # correct shape
        ene = ene.reshape((self.num_wan, kpts.shape[0])).T

        return (kpts, ene)
