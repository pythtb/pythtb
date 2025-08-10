import numpy as np
from math import factorial
from itertools import permutations

__all__ = [
    "levi_civita",
    "finite_diff_coeffs",
    "is_Hermitian",
    "pauli_decompose",
    "get_trial_wfs",
    "get_periodic_H",
]


def get_periodic_H(model, H_flat, k_vals):
    orb_vecs = model.get_orb_vecs()
    orb_vec_diff = orb_vecs[:, None, :] - orb_vecs[None, :, :]
    # orb_phase = np.exp(1j * 2 * np.pi * np.einsum('ijm, ...m->...ij', orb_vec_diff, k_vals))
    orb_phase = np.exp(1j * 2 * np.pi * np.matmul(orb_vec_diff, k_vals.T)).transpose(
        2, 0, 1
    )
    H_per_flat = H_flat * orb_phase
    return H_per_flat


def get_trial_wfs(tf_list, norb, nspin=1):
    """
    Args:
        tf_list: list[int | list[tuple]]
            list of tuples defining the orbital and amplitude of the trial function
            on that orbital. Of the form [ [(orb, amp), ...], ...]. If spin is included,
            then the form is [ [(orb, spin, amp), ...], ...]

    Returns:
        tfs: np.ndarray
            Array of trial functions
    """

    # number of trial functions to define
    num_tf = len(tf_list)

    if nspin == 2:
        tfs = np.zeros([num_tf, norb, 2], dtype=complex)
        for j, tf in enumerate(tf_list):
            assert isinstance(
                tf, (list, np.ndarray)
            ), "Trial function must be a list of tuples"
            for orb, spin, amp in tf:
                tfs[j, orb, spin] = amp
            tfs[j] /= np.linalg.norm(tfs[j])

    elif nspin == 1:
        # initialize array containing tfs = "trial functions"
        tfs = np.zeros([num_tf, norb], dtype=complex)
        for j, tf in enumerate(tf_list):
            assert isinstance(
                tf, (list, np.ndarray)
            ), "Trial function must be a list of tuples"
            for site, amp in tf:
                tfs[j, site] = amp
            tfs[j] /= np.linalg.norm(tfs[j])

    return tfs


def detect_degeneracies(eigenvalues, tol=1e-8):
    """
    Detects degeneracies in a list of eigenvalues.

    Parameters:
        eigenvalues (array): List or array of eigenvalues (assumed sorted).
        tol (float): Tolerance for identifying degeneracy.

    Returns:
        degenerate_groups (list of lists): Indices of degenerate eigenvalues.
    """
    eigenvalues = np.array(eigenvalues)
    # sort eigenvalues if not already sorted
    if not np.all(np.diff(eigenvalues) >= 0):
        eigenvalues = np.sort(eigenvalues)
    degenerate_groups = []
    current_group = [0]

    for i in range(1, len(eigenvalues)):
        if abs(eigenvalues[i] - eigenvalues[i - 1]) < tol:
            current_group.append(i)
        else:
            if len(current_group) > 1:
                degenerate_groups.append(current_group)
            current_group = [i]

    if len(current_group) > 1:
        degenerate_groups.append(current_group)

    return degenerate_groups


def levi_civita(n, d):
    """
    Constructs the rank-n Levi-Civita tensor in dimension d.

    Parameters:
    n (int): Rank of the tensor (number of indices).
    d (int): Dimension (number of possible index values).

    Returns:
    np.ndarray: Levi-Civita tensor of shape (d, d, ..., d) with n dimensions.
    """
    shape = (d,) * n
    epsilon = np.zeros(shape, dtype=int)
    # Generate all possible permutations of n indices
    for perm in permutations(range(d), n):
        # Compute the sign of the permutation
        sign = np.linalg.det(np.eye(n)[list(perm)])
        epsilon[perm] = int(np.sign(sign))  # +1 for even, -1 for odd permutations

    return epsilon


def finite_diff_coeffs(order_eps, derivative_order=1, mode="central"):
    """
    Compute finite difference coefficients using the inverse of the Vandermonde matrix.

    Parameters:
        stencil_points (array-like): The relative positions of the stencil points (e.g., [-2, -1, 0, 1, 2]).
        derivative_order (int): Order of the derivative to approximate (default is first derivative).

    Returns:
        coeffs (numpy array): Finite difference coefficients for the given stencil.
    """
    if mode not in ["central", "forward", "backward"]:
        raise ValueError("Mode must be 'central', 'forward', or 'backward'.")

    num_points = derivative_order + order_eps

    if mode == "central":
        if num_points % 2 == 0:
            num_points += 1
        half_span = num_points // 2
        stencil = np.arange(-half_span, half_span + 1)

    elif mode == "forward":
        stencil = np.arange(0, num_points)

    elif mode == "backward":
        stencil = np.arange(-num_points + 1, 1)

    A = np.vander(stencil, increasing=True).T  # Vandermonde matrix
    b = np.zeros(num_points)
    b[derivative_order] = factorial(
        derivative_order
    )  # Right-hand side for the desired derivative

    coeffs = np.linalg.solve(A, b)  # Solve system Ax = b
    return coeffs, stencil

def fin_diff(U_k, mu, dk_mu, order_eps, mode='central'):
    coeffs, stencil = finite_diff_coeffs(order_eps=order_eps, mode=mode)

    fd_sum = np.zeros_like(U_k)

    for s, c in zip(stencil, coeffs):
        fd_sum += c * np.roll(U_k, shift=-s, axis=mu)

    v = fd_sum / (dk_mu)
    return v

def is_Hermitian(M):
    """
    Check if a matrix M is Hermitian.

    Parameters:
        M (array-like): A square matrix (as a numpy array or convertible to one).

    Returns:
        bool: True if M is Hermitian, False otherwise.
    """
    M = np.array(M, dtype=complex)
    if M.ndim == 0:
        return np.allclose(M, np.conj(M))
    # 1D: not Hermitian (by usual definition)
    if M.ndim == 1:
        return False
    # Otherwise: check M == M^\dagger
    return np.allclose(M, M.conj().swapaxes(-1, -2))


def pauli_decompose(M):
    """
    Decompose a 2x2 matrix M in terms of the Pauli matrices.

    That is, find coefficients a0, a1, a2, a3 such that:

        M = a0 * I + a1 * sigma_x + a2 * sigma_y + a3 * sigma_z

    Parameters:
        M (array-like): A 2x2 matrix (as a numpy array or convertible to one).
        precision (int): Number of significant digits for the coefficients.

    Returns:
        str: A string representing the decomposition, e.g.
             "1I + 0.3τₓ - 0.2τ_y + 0τ_z"

    Note: This function is applicable only when nspin = 2.
    """
    M = np.array(M, dtype=complex)
    if M.shape != (2, 2):
        raise ValueError("Matrix must be 2x2 for Pauli decomposition.")

    # Define the 2x2 identity and Pauli matrices.
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Compute coefficients using the Hilbert-Schmidt inner product.
    a0 = 0.5 * np.trace(M)
    a1 = 0.5 * np.trace(np.dot(M, sigma_x))
    a2 = 0.5 * np.trace(np.dot(M, sigma_y))
    a3 = 0.5 * np.trace(np.dot(M, sigma_z))

    return [a0, a1, a2, a3]


def no_2pi(x, clos):
    "Make x as close to clos by adding or removing 2pi"
    while abs(clos - x) > np.pi:
        if clos - x > np.pi:
            x += 2.0 * np.pi
        elif clos - x < -1.0 * np.pi:
            x -= 2.0 * np.pi
    return x


def _cart_to_red(a_vecs, cart):
    "Convert cartesian vectors cart to reduced coordinates of a1,a2,a3 vectors"
    # (a1, a2, a3) = tmp
    # matrix with lattice vectors
    # cnv = np.array([a1, a2, a3])
    # cnv = cnv.T  # transpose
    # # reduced coordinates
    # red = np.zeros_like(cart, dtype=float)
    # for i in range(0, len(cart)):
    #     red[i] = np.dot(cnv, cart[i])
    # return red
    cnv = np.linalg.inv(np.array(a_vecs).T)  # inverse
    return np.dot(cart, cnv.T)


def _red_to_cart(a_vecs, red):
    "Convert reduced to cartesian vectors."
    a1, a2, a3 = a_vecs

    basis = np.array([a1, a2, a3])
    cart = np.array(red) @ basis

    # # cartesian coordinates
    # cart2 = np.zeros_like(red, dtype=float)
    # for i in range(0, len(cart)):
    #     cart2[i, :] = a1 * red[i][0] + a2 * red[i][1] + a3 * red[i][2]
    # print(np.allclose(cart, cart2))  # should be True

    return cart


def _is_int(a):
    return np.issubdtype(type(a), np.integer)


class PositionOperatorApproximationError(Exception):
    """
    Raised when a calculation involving the position operator is attempted
    using a tight-binding model generated by Wannier90, which neglects off-diagonal
    position operator elements.
    """

    pass


def _offdiag_approximation_warning_and_stop():
    raise PositionOperatorApproximationError(
        """

----------------------------------------------------------------------

  It looks like you are trying to calculate Berry-like object that
  involves position operator.  However, you are using a tight-binding
  model that was generated from Wannier90.  This procedure introduces
  approximation as it ignores off-diagonal elements of the position
  operator in the Wannier basis.  This is discussed here in more
  detail:

    http://www.physics.rutgers.edu/pythtb/usage.html#pythtb.w90

  If you know what you are doing and wish to continue with the
  calculation despite this approximation, please call the following
  function on your TBModel object

    my_model.ignore_position_operator_offdiagonal()

----------------------------------------------------------------------

"""
    )


def compute_d4k_and_d2k(delta_k):
    """
    Computes the 4D volume element d^4k and the 2D plaquette areas d^2k for a given set of difference vectors in 4D space.

    Parameters:
    delta_k (numpy.ndarray): A 4x4 matrix where each row is a 4D difference vector.

    Returns:
    tuple: (d4k, plaquette_areas) where
        - d4k is the absolute determinant of delta_k (4D volume element).
        - plaquette_areas is a dictionary with keys (i, j) and values representing d^2k_{ij}.
    """
    # Compute d^4k as the determinant of the 4x4 difference matrix
    d4k = np.abs(np.linalg.det(delta_k))

    # Function to compute 2D plaquette area in 4D space
    def compute_plaquette_area(v1, v2):
        """Compute the 2D plaquette area spanned by two 4D vectors."""
        area_squared = 0.0
        # Sum over all unique (m, n) pairs where m < n
        for m in range(4):
            for n in range(m + 1, 4):
                area_squared += (v1[m] * v2[n] - v1[n] * v2[m]) ** 2
        return np.sqrt(area_squared)

    # Compute all unique plaquette areas
    plaquette_areas = {}
    for i in range(4):
        for j in range(i + 1, 4):
            plaquette_areas[(i, j)] = compute_plaquette_area(delta_k[i], delta_k[j])

    return d4k, plaquette_areas


# def vel_op_fin_diff(model, H_flat, k_vals, dk, order_eps=1, mode='central'):
#     """
#     Compute velocity operators using finite differences.

#     Parameters:
#         H_mesh: ndarray of shape (Nk, M, M)
#             The Hamiltonian on the parameter grid.
#         dk: list of float
#             Step sizes in each parameter direction.

#     Returns:
#         v_mu_fd: list of ndarray
#             Velocity operators for each parameter direction.
#     # """

#     # recip_lat_vecs = model.get_recip_lat_vecs()
#     # recip_basis = recip_lat_vecs/ np.linalg.norm(recip_lat_vecs, axis=1, keepdims=True)
#     # g = recip_basis @ recip_basis.T
#     # sqrt_mtrc = np.sqrt(np.linalg.det(g))
#     # g_inv = np.linalg.inv(g)

#     # dk = np.einsum("ij, j -> i", g_inv, dk)

#     # assume only k for now
#     dim_param = model._dim_k # Number of parameters (dimensions)
#     # assume equal number of mesh points along each dimension
#     nks = ( int(H_flat.shape[0]**(1/dim_param)),)*dim_param

#     # Switch to periodic gauge H(k) = H(k+G)
#     H_flat = get_periodic_H(model, H_flat, k_vals)
#     H_mesh = H_flat.reshape(*nks, model._norb, model._norb)
#     v_mu_fd = np.zeros((dim_param, *H_mesh.shape), dtype=complex)

#     # Compute Jacobian
#     recip_lat_vecs = model.get_recip_lat_vecs()
#     inv_recip_lat = np.linalg.inv(recip_lat_vecs)

#     for mu in range(dim_param):
#         coeffs, stencil = finite_diff_coeffs(order_eps=order_eps, mode=mode)

#         derivative_sum = np.zeros_like(H_mesh)

#         for s, c in zip(stencil, coeffs):
#             H_shifted = np.roll(H_mesh, shift=-s, axis=mu)
#             derivative_sum += c * H_shifted

#         v_mu_fd[mu] = derivative_sum / (dk[mu])

#         # Ensure Hermitian symmetry
#         v_mu_fd[mu] = 0.5 * (v_mu_fd[mu] + np.conj(v_mu_fd[mu].swapaxes(-1, -2)))

#     return v_mu_fd


def _wf_dpr(wf1, wf2):
    """calculate dot product between two wavefunctions.
    wf1 and wf2 are of the form [orbital,spin]"""
    return np.dot(wf1.flatten().conjugate(), wf2.flatten())


def _one_berry_loop(wf, berry_evals=False):
    """Do one Berry phase calculation (also returns a product of M
    matrices).  Always returns numbers between -pi and pi.  wf has
    format [kpnt,band,orbital,spin] and kpnt has to be one dimensional.
    Assumes that first and last k-point are the same. Therefore if
    there are n wavefunctions in total, will calculate phase along n-1
    links only!  If berry_evals is True then will compute phases for
    individual states, these corresponds to 1d hybrid Wannier
    function centers. Otherwise just return one number, Berry phase."""
    # number of occupied states
    nocc = wf.shape[1]
    # temporary matrices
    prd = np.identity(nocc, dtype=complex)
    ovr = np.zeros([nocc, nocc], dtype=complex)
    # go over all pairs of k-points, assuming that last point is overcounted!
    for i in range(wf.shape[0] - 1):
        # generate overlap matrix, go over all bands
        for j in range(nocc):
            for k in range(nocc):
                ovr[j, k] = _wf_dpr(wf[i, j, :], wf[i + 1, k, :])
        # only find Berry phase
        if not berry_evals:
            # multiply overlap matrices
            prd = np.dot(prd, ovr)
        # also find phases of individual eigenvalues
        else:
            # cleanup matrices with SVD then take product
            matU, sing, matV = np.linalg.svd(ovr)
            prd = np.dot(prd, np.dot(matU, matV))
    # calculate Berry phase
    if not berry_evals:
        det = np.linalg.det(prd)
        pha = (-1.0) * np.angle(det)
        return pha
    # calculate phases of all eigenvalues
    else:
        evals = np.linalg.eigvals(prd)
        eval_pha = (-1.0) * np.angle(evals)
        # sort these numbers as well
        eval_pha = np.sort(eval_pha)
        return eval_pha


def _one_flux_plane(wfs2d):
    "Compute fluxes on a two-dimensional plane of states."
    # size of the mesh
    nk0 = wfs2d.shape[0]
    nk1 = wfs2d.shape[1]
    # number of bands (will compute flux of all bands taken together)
    nbnd = wfs2d.shape[2]

    # here store flux through each plaquette of the mesh
    all_phases = np.zeros((nk0 - 1, nk1 - 1), dtype=float)

    # go over all plaquettes
    for i in range(nk0 - 1):
        for j in range(nk1 - 1):
            # generate a small loop made out of four pieces
            wf_use = []
            wf_use.append(wfs2d[i, j])
            wf_use.append(wfs2d[i + 1, j])
            wf_use.append(wfs2d[i + 1, j + 1])
            wf_use.append(wfs2d[i, j + 1])
            wf_use.append(wfs2d[i, j])
            wf_use = np.array(wf_use, dtype=complex)
            # calculate phase around one plaquette
            all_phases[i, j] = _one_berry_loop(wf_use)

    return all_phases
