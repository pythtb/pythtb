import re
from pathlib import Path

import numpy as np

from .tbmodel import TBModel
from .lattice import Lattice
from .utils import _cart_to_red, _red_to_cart, deprecated, kpath_distance

__all__ = ["W90"]

BOHRTOANG = 0.52917721092  # Bohr radius in Angstroms
_KPOINT_LABEL_PATTERN = re.compile(r"^(?P<base>[^\d]+?)(?P<suffix>\d+)?$", re.UNICODE)

class W90:
    r"""Interface to Wannier90 

    This class imports tight-binding model parameters from an output 
    of a `Wannier90 <http://www.wannier.org>`_ code.
    Upon instantiation, this class will read in the Wannier90 output
    files from the specified folder. To create :class:`pythtb.TBModel`
    object use the :meth:`model` function.

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
    :meth:`w90_bands` for more detail.

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
        self.folder = Path(path).expanduser()
        if not self.folder.exists():
            raise FileNotFoundError(f"Wannier90 folder not found: {self.folder}")
        self.path = str(self.folder)
        self.prefix = prefix

        win_lines = self._load_win_lines()
        self._win_lines = win_lines
        self.lat = self._parse_unit_cell_block(win_lines)
       
        # read in hamiltonian matrix, in eV
        self.num_wan, self.ham_r = self._load_hr_fast()

        # check if for every non-zero R there is also -R
        self._validate_hr_symmetry()

        # read in wannier centers
        self.xyz_cen, self.red_cen = self._load_centers()

        self.lattice = Lattice(self.lat, self.red_cen, periodic_dirs=[0,1,2])

        # caches (filled lazily)
        self._vecR_cache = {}
        self._dist_cache = {}

    def _load_win_lines(self):
        win_path = self.folder / f"{self.prefix}.win"
        try:
            with win_path.open("r") as fh:
                return fh.readlines()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Unable to locate Wannier90 input file {win_path}") from exc
        
    def _load_hr_fast(self):
        hr_path = self.folder / f"{self.prefix}_hr.dat"
        with hr_path.open("r") as fh:
            fh.readline()  # skip first Wannier90 comment/header
            num_wan = int(fh.readline())  # read number of Wannier functions
            num_ws = int(fh.readline())  # read number of Wigner-Seitz cells

            # Read degeneracies of Wigner-Seitz cells
            deg_ws = []
            while len(deg_ws) < num_ws:
                line = fh.readline()
                if not line:
                    raise RuntimeError("Unexpected EOF while reading Wigner–Seitz degeneracies.")
                deg_ws.extend(int(val) for val in line.split())
            # Truncate to expected count
            deg_ws = np.asarray(deg_ws[:num_ws], dtype=int)

            # Load remaining numeric table (R vectors, indices, real/imaginary parts)
            data = np.loadtxt(fh)

        # Check if data is empty
        if data.size == 0:
            return num_wan, {}

        # Promote single row to shape (1, 7)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] != 7:
            raise RuntimeError("Wannier90 _hr.dat must have seven columns per row.")

        R_vecs = data[:, :3].astype(np.int64)  # Triplets (R1, R2, R3)
        i_idx = data[:, 3].astype(np.int64) - 1  # Wannier function index i
        j_idx = data[:, 4].astype(np.int64) - 1  # Wannier function index j
        hop_vals = data[:, 5] + 1j * data[:, 6]  # Hopping values

        # Find unique R vectors and their indices (unique_R), remember first line each
        # appears (first_idx), and how each row maps back (inverse)
        unique_R, first_idx, inverse = np.unique(
            R_vecs, axis=0, return_inverse=True, return_index=True
        )
        order = np.argsort(first_idx)  # sort shells by first occurrence
        unique_R = unique_R[order]  # reorder unique R vectors accordingly

        if deg_ws.size < unique_R.shape[0]:
            raise RuntimeError("Not enough degeneracy entries for the R shells present.")
        deg_ws = deg_ws[: unique_R.shape[0]]  # use only degeneracies actually needed (zeros get dropped)

        remap = np.empty_like(order) 
        remap[order] = np.arange(order.size) # permutation that maps sorted order back to original
        inverse = remap[inverse] # remap per-row shell into new ordering

        # One (num_wan x num_wan) matrix per unique R vector
        blocks = np.zeros((unique_R.shape[0], num_wan, num_wan), dtype=complex)
        # Scatter-add every hopping into proper (R, i, j) block
        np.add.at(blocks, (inverse, i_idx, j_idx), hop_vals)

        ham_r = {
            tuple(int(v) for v in R_vec): {"h": blocks[idx], "deg": int(deg_ws[idx])}
            for idx, R_vec in enumerate(unique_R)
        }
        return num_wan, ham_r
        
    def _extract_win_block(self, win_lines, name):
        begin = f"begin {name}".lower()
        end = f"end {name}".lower()
        inside = False
        block = []
        for raw in win_lines:
            stripped = raw.strip()
            lower = stripped.lower()
            if not inside and lower.startswith(begin):
                inside = True
                continue
            if inside:
                if lower.startswith(end):
                    break
                block.append(raw.rstrip("\n"))
        return block
    
    def _parse_unit_cell_block(self, win_lines):
        block = self._extract_win_block(win_lines, "unit_cell_cart")
        if not block:
            raise Exception("Unable to find unit_cell_cart block in the .win file.")

        scale = 1.0
        first = block[0].strip().lower()
        if first in {"bohr", "ang", "angstrom"}:
            if first == "bohr":
                scale = BOHRTOANG
            block = block[1:]
        if len(block) < 3:
            raise ValueError("unit_cell_cart block must contain three lattice vectors.")

        lat = np.zeros((3, 3), dtype=float)
        for row_idx in range(3):
            parts = block[row_idx].split()
            if len(parts) < 3:
                raise ValueError("Each unit_cell_cart row must have three components.")
            lat[row_idx] = [float(parts[col]) * scale for col in range(3)]
        return lat
    
    def _validate_hr_symmetry(self):
        R_set = set(self.ham_r.keys())
        for R in R_set:
            if R != (0, 0, 0) and (-R[0], -R[1], -R[2]) not in R_set:
                raise Exception(f"Did not find negative R for R = {R}!")

    @staticmethod
    def _wrap01(x: np.ndarray) -> np.ndarray:
        out = np.mod(x, 1.0)
        # snap 1.0 → 0.0 to avoid 2π glitches
        out[np.isclose(out, 1.0, atol=1e-12)] = 0.0
        return out
            
    def _load_centers(self):
        centres_path = self.folder / f"{self.prefix}_centres.xyz"
        try:
            with centres_path.open("r") as fh:
                lines = fh.readlines()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Unable to locate Wannier center file {centres_path}") from exc

        coords = []
        start = 2
        for idx in range(self.num_wan):
            try:
                fields = lines[start + idx].split()
            except IndexError as exc:
                raise Exception("Centres file shorter than expected.") from exc
            if not fields or fields[0] != "X":
                raise Exception("Inconsistency in the centres file.")
            coords.append([float(value) for value in fields[1:4]])

        xyz_cen = np.asarray(coords, dtype=float)
        red = _cart_to_red((self.lat[0], self.lat[1], self.lat[2]), xyz_cen)
        return xyz_cen, red
      
    def _kpoint_path_nodes(self, *, latex: bool = True):
        """
        Return the reduced-coordinate nodes declared in the ``kpoint_path`` block.

        Parameters
        ----------
        latex : bool, optional
            When True (default) convert labels into LaTeX-friendly strings,
            e.g. ``"G" -> r"$\\Gamma$"``.

        Returns
        -------
        coords : numpy.ndarray
            Array with shape ``(n_nodes, 3)`` containing the reduced coordinates.
        labels : list[str]
            Labels for each node, optionally formatted for LaTeX rendering.
        """
        block = self._extract_win_block(self._win_lines, "kpoint_path")
        if not block:
            raise ValueError("No kpoint_path block present in the .win file.")

        entries: list[tuple[str, np.ndarray]] = []
        for line in block:
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) % 4:
                raise ValueError(
                    "Each kpoint_path entry must be a label followed by three coordinates."
                )
            for offset in range(0, len(tokens), 4):
                label = tokens[offset]
                try:
                    coords = np.array(
                        [float(tokens[offset + 1]), float(tokens[offset + 2]), float(tokens[offset + 3])],
                        dtype=float,
                    )
                except ValueError as exc:
                    raise ValueError(f"Failed to parse coordinates for k-point '{label}'.") from exc
                entries.append((label, coords))

        if not entries:
            raise ValueError("kpoint_path block is empty.")

        labels: list[str] = []
        coords_list: list[np.ndarray] = []
        prev_label = None
        prev_coords = None
        for label, coords in entries:
            if prev_label is not None and label == prev_label and np.allclose(coords, prev_coords):
                # Skip duplicated node introduced by segment chaining (e.g. P1->P2, P2->P3)
                continue
            labels.append(label)
            coords_list.append(coords)
            prev_label = label
            prev_coords = coords

        coords_arr = np.vstack(coords_list)
        if latex:
            labels = [self._format_k_label(lbl) for lbl in labels]

        return coords_arr, labels

    def _format_k_label(self, label: str) -> str:
        special = {
            "g": r"\Gamma",
            "gamma": r"\Gamma",
            "Γ": r"\Gamma",
            "delta": r"\Delta",
            "Δ": r"\Delta",
            "theta": r"\Theta",
            "Θ": r"\Theta",
            "lambda": r"\Lambda",
            "λ": r"\Lambda",
            "xi": r"\Xi",
            "ξ": r"\Xi",
            "pi": r"\Pi",
            "π": r"\Pi",
            "sigma": r"\Sigma",
            "σ": r"\Sigma",
            "upsilon": r"\Upsilon",
            "υ": r"\Upsilon",
            "phi": r"\Phi",
            "ϕ": r"\Phi",
            "psi": r"\Psi",
            "ψ": r"\Psi",
            "omega": r"\Omega",
            "ω": r"\Omega",
        }

        raw = label.strip()
        if not raw:
            return "$$"

        match = _KPOINT_LABEL_PATTERN.match(raw)
        if match:
            base = match.group("base").strip()
            suffix = match.group("suffix")

            key = base.lower()
            latex_base = special.get(key)
            if latex_base is None:
                latex_base = special.get(base)

            if latex_base is None:
                if len(base) == 1 and base.isalpha():
                    latex_base = base
                else:
                    latex_base = rf"\mathrm{{{base}}}"
        else:
            latex_base = rf"\mathrm{{{raw}}}"
            suffix = None

        if match:
            suffix = match.group("suffix")
        else:
            suffix = None

        if suffix:
            return rf"${latex_base}_{{{suffix}}}$"
        return rf"${latex_base}$"

    def _get_vecR(self, R):
        """Cartesian vector for reduced lattice vector R, cached."""
        if not hasattr(self, "_vecR_cache"):
            self._vecR_cache = {}
        if R in self._vecR_cache:
            return self._vecR_cache[R]
        vecR = _red_to_cart((self.lat[0], self.lat[1], self.lat[2]), [R])[0]
        self._vecR_cache[R] = vecR
        return vecR


    def _get_dist_matrix(self, R):
        """Distance for reduced lattice vector R, cached."""
        if not hasattr(self, "_dist_cache"):
            self._dist_cache = {}
        if R in self._dist_cache:
            return self._dist_cache[R]
        vecR = self._get_vecR(R)
        delta = (-self.xyz_cen[:, None, :] + self.xyz_cen[None, :, :]) + vecR[None, None, :]
        dist = np.linalg.norm(delta, axis=2)  # (num_wan, num_wan)
        self._dist_cache[R] = dist
        return dist

    def _precompute_distances(self):
        """Precompute distance matrices for all reduced lattice vectors."""
        if not hasattr(self, "_dist_cache"):
            self._dist_cache = {}
        for R in self.ham_r.keys():
            if R not in self._dist_cache:
                self._get_dist_matrix(R)

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
        tb = TBModel(self.lattice)

        # remember that this model was computed from w90
        tb._from_w90 = True
        tb._assume_position_operator_diagonal = False

        # -------------------------
        # Onsites (vectorized)
        # -------------------------
        # Divide by degeneracy only once and assert onsite is (numerically) real
        hr0 = self.ham_r[(0, 0, 0)]
        deg0 = float(hr0["deg"])  # scalar
        onsite = (hr0["h"].diagonal() / deg0).real
        # sanity check: imaginary part should be tiny
        if np.max(np.abs(np.imag(np.diag(hr0["h"]) / deg0))) > 1e-9:
            raise Exception("Onsite terms should be real!")
        tb.set_onsite(onsite - zero_energy)

        # -------------------------
        # Hoppings (vectorized per R)
        # -------------------------
        # Precompute for speed
        # xyz_cen = self.xyz_cen  # (num_wan, 3), Cartesian Angstroms
        num_wan = self.num_wan
        # lat_tuple = (self.lat[0], self.lat[1], self.lat[2])

        # Helper to decide if we should process an R (to avoid double counting)
        def _use_R(R):
            r1, r2, r3 = R
            if r1 != 0:
                return r1 > 0
            if r2 != 0:
                return r2 > 0
            return r3 > 0
        
        if max_distance is not None and not self._dist_cache:
            self._precompute_distances()

        for R, blk in self.ham_r.items():
            Hr = blk["h"]
            deg = float(blk["deg"])  # scalar

            # Onsite block already handled; keep only off-diagonal pairs here.
            if R == (0, 0, 0):
                use_this_R = True
            else:
                use_this_R = _use_R(R)

            if not use_this_R:
                continue

            # Divide by degeneracy once per block
            Ham = Hr / deg  # (num_wan, num_wan)

            # Start from allowed entries and avoid double counting
            if R == (0, 0, 0):
                keep = np.zeros((num_wan, num_wan), dtype=bool)
                iu = np.triu_indices(num_wan, k=1)
                keep[iu] = True

            else:
                keep = np.ones((num_wan, num_wan), dtype=bool)
                # np.fill_diagonal(keep, False)  
                np.fill_diagonal(keep, True)  

            # Distance cutoff (use cached distances; compute lazily if needed)
            if max_distance is not None:
                dist = self._get_dist_matrix(R)
                keep &= (dist <= max_distance)
                if not np.any(keep):
                    continue

            # Apply min_hopping_norm filter
            if min_hopping_norm is not None:
                keep &= (np.abs(Ham) >= min_hopping_norm)
                if not np.any(keep):
                    continue

            # Optionally zero-out tiny imaginary parts before insertion
            if ignorable_imaginary_part is not None:
                sel = keep & (np.abs(Ham.imag) < ignorable_imaginary_part)
                if np.any(sel):
                    Ham = Ham.copy()
                    Ham.imag[sel] = 0.0

            # Emit kept hoppings in bulk to minimize Python overhead
            ii, jj = np.nonzero(keep)
            if ii.size:
                amps = Ham[ii, jj]
                R_arr = np.repeat(np.array(R)[None, :], ii.size, axis=0)
                tb._append_hops(amps, ii, jj, R_arr)
                # for i, j, a in zip(ii.tolist(), jj.tolist(), amps.tolist()):
                #     tb.set_hop(a, i, j, list(R))

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
        num_wan = self.num_wan

        for R, blk in self.ham_r.items():
            Ham = blk["h"] / float(blk["deg"])
            dist = self._get_dist_matrix(R)  # (num_wan, num_wan)
            keep = np.ones((num_wan, num_wan), dtype=bool)

            if R == (0, 0, 0):
                np.fill_diagonal(keep, False)  # avoid diagonal terms

            ret_ham.append(Ham[keep])
            ret_dist.append(dist[keep])

        return (np.concatenate(ret_dist), np.concatenate(ret_ham))


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
        for R in self.ham_r.keys():
            dist = self._get_dist_matrix(R)
            shells.extend(np.round(dist.ravel(), num_digits).tolist())

        # remove duplicates and sort
        shells = np.sort(list(set(shells)))

        return shells
    
    @deprecated("use .solve_ham() instead (since v2.0).", category=FutureWarning)
    def w90_bands_consistency(self):
        """
        .. deprecated:: 2.0.0
            Use .w90_bands() instead.
        """
        return self.w90_bands()

    def w90_bands(
            self, 
            return_k_cart: bool = False,
            return_k_dist: bool = False, 
            return_k_nodes: bool = False,
        ):
        r"""Read interpolated band structure from Wannier90 output files.

        .. versionadded:: 2.0.0
            Added ``return_k_dist`` to optionally return cumulative path distances.

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

        Parameters
        ----------
        return_k_cart : bool, optional
            If True, also return k-points in Cartesian coordinates.
        return_k_dist : bool, optional
            If True, also return the cumulative k-path distance (in 1/Å).
        return_k_nodes : bool, optional
            If True, also return the k-point nodes and labels used in the path.
            Format is ``(k_nodes, k_labels)`` where ``k_nodes`` is an array
            of reduced coordinates and ``k_labels`` is a list of strings.

        Returns
        -------
        kpts : array
            k-points in reduced coordinates used in the
            interpolation in Wannier90 code. The expected format is
            the same as the input to
            :func:`pythtb.TBModel.solve_ham`.
        ene : array
            Energies interpolated by Wannier90 in eV. Format is ``ene[kpt,band]``.
        k_dist : array, optional
            Cumulative distances along the path (1/Å) as reported by Wannier90.
            Returned when ``return_k_dist=True``. Useful for plotting band structures.
        k_cart : array, optional
            k-points in Cartesian coordinates (1/Å).
            Returned when ``return_k_cart=True``.
        k_nodes : tuple[array, list[str]], optional
            Tuple ``(k_nodes, k_labels)`` containing the reduced coordinates
            of the k-point nodes and their labels.
            Returned when ``return_k_nodes=True``.

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

        >>> w90_kpt, w90_evals, w90_k_dist, w90_k_nodes, w90_k_labels = silicon.w90_bands(
        ... return_k_dist=True, return_k_nodes=True)

        Get simplified model

        >>> my_model_simple = silicon.model(min_hopping_norm=0.01)

        Solve simplified model on the same k-path as in `Wannier90`

        >>> evals = my_model.solve_ham(w90_kpt)

        Now plot the comparison of the two
        
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> for i in range(evals.shape[1]):
        >>>     ax.plot(range(evals.shape[1]), evals[i], "r-", zorder=-50)
        >>> for i in range(w90_evals.shape[0]):
        >>>     ax.plot(range(w90_evals.shape[1]), w90_evals[i], "k-", zorder=-100)
        >>> fig.savefig("comparison.pdf")
        """
        kpts_path = self.folder / f"{self.prefix}_band.kpt"
        ene_path = self.folder / f"{self.prefix}_band.dat"

        # read in kpoints in reduced coordinates
        kpts = np.loadtxt(kpts_path, skiprows=1)[:, :3]
        # read in energies
        ene_raw = np.loadtxt(ene_path)
        ene = ene_raw[:, 1].reshape((self.num_wan, kpts.shape[0])).T

        B = self.lattice.recip_lat_vecs
        k_dist = kpath_distance(kpts, b1=B[0], b2=B[1], b3=B[2])

        results = (kpts, ene)
        if return_k_dist:
            results += (k_dist,)
        if return_k_cart:
            k_cart = kpts @ B
            results += (k_cart,)
        if return_k_nodes:
            k_nodes, k_labels = self._kpoint_path_nodes(latex=True)
            results += (k_nodes, k_labels)
        return results

    def qe_bands(self, return_k_cart=False, return_meta=False, return_kdist=False):
        """
        Read band structure from Quantum ESPRESSO output files.

        Parameters
        ----------
        return_k_cart : bool, optional
            If True, also return k-points in Cartesian coordinates.
        return_meta : bool, optional
            If True, return header metadata (nbnd, nks) when available.
        return_kdist : bool, optional
            If True, also return cumulative k-path distances (1/Å).

        Returns
        -------
        tuple
            Returns ``(k_frac, energies[, k_dist][, k_cart][, meta])`` depending
            on the requested flags. When ``return_kdist`` is True the
            cumulative distance is computed from the lattice reciprocal vectors.
        """
        bands_path = self.folder / f"{self.prefix}_bands.dat"
        # Try to grab nbnd/nks from header
        meta = {}
        m = re.search(r"nbnd\s*=\s*(\d+).+nks\s*=\s*(\d+)", open(bands_path).read(5000), re.I|re.S)
        if m:
            meta["nbnd"] = int(m.group(1))
            meta["nks"]  = int(m.group(2))

        def is_k_marker(s: str) -> bool:
            # k-marker line has exactly three floats
            try:
                vals = [float(x) for x in s.split()]
                return len(vals) == 3
            except ValueError:
                return False

        klist, energies_rows, ebuf = [], [], []
        with bands_path.open("r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if is_k_marker(s):
                    # starting a new k-point: flush previous energies if any
                    if ebuf:
                        energies_rows.append(ebuf)
                        ebuf = []
                    kx, ky, kz = (float(x) for x in s.split())
                    klist.append([kx, ky, kz])
                else:
                    # energy values (possibly many per line)
                    try:
                        vals = [float(x) for x in s.split()]
                    except ValueError:
                        continue
                    ebuf.extend(vals)

        # flush last k
        if ebuf:
            energies_rows.append(ebuf)

        # Convert
        E_raw = np.array(energies_rows, dtype=float) 
        k_cart = np.array(klist, dtype=float)
        # k_cart in units of 2pi/alat
        alat = np.linalg.norm(self.lattice.lat_vecs[0])
        k_cart *= (2 * np.pi / alat)

        # Infer dimensions if header absent / inconsistent
        nks = meta.get("nks", E_raw.shape[0])
        nbnd = meta.get("nbnd", int(max(len(row) for row in E_raw)))

        # Normalize to (nks, nbnd)
        E = np.full((nks, nbnd), np.nan, dtype=float)
        for i in range(min(nks, len(E_raw))):
            row = E_raw[i]
            E[i, :min(nbnd, len(row))] = row[:nbnd]

        B = self.lattice.recip_lat_vecs
        k_frac = k_cart @ np.linalg.inv(B)

        k_dist = None
        if return_kdist:
            k_dist = kpath_distance(k_frac, b1=B[0], b2=B[1], b3=B[2])

        result = [k_frac, E]
        if return_kdist:
            result.append(k_dist)
        if return_k_cart:
            result.append(k_cart)
        if return_meta:
            result.append(meta)

        return tuple(result)
