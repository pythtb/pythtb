"""Small numerical and bookkeeping helpers shared across PythTB modules."""

import numpy as np
from math import factorial
from itertools import permutations
import functools
import warnings

__all__ = [
    "levi_civita",
    "finite_diff_coeffs",
    "finite_difference",
    "is_Hermitian",
    "pauli_decompose",
    "get_trial_wfs",
]


def import_tensorflow():
    """Import and return the TensorFlow module, with a helpful error if absent."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is not installed. Please install it or disable the "
            "TensorFlow option (use_tensorflow/tf_speedup) for this call."
        ) from exc
    return tf


# deprecation decorator
def deprecated(message: str, category=FutureWarning):
    """
    Decorator to mark a function as deprecated.
    Raises a FutureWarning with the given message when the function is called.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__qualname__} is deprecated and will be removed in a future release: {message}",
                category=category,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def copydoc(src):
    """Decorator copying ``src``'s docstring onto the decorated function."""

    def deco(dst):
        dst.__doc__ = src.__doc__
        return dst

    return deco


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
            assert isinstance(tf, (list, np.ndarray)), (
                "Trial function must be a list of tuples"
            )
            for orb, spin, amp in tf:
                tfs[j, orb, spin] = amp
            tfs[j] /= np.linalg.norm(tfs[j])

    elif nspin == 1:
        # initialize array containing tfs = "trial functions"
        tfs = np.zeros([num_tf, norb], dtype=complex)
        for j, tf in enumerate(tf_list):
            assert isinstance(tf, (list, np.ndarray)), (
                "Trial function must be a list of tuples"
            )
            for site, amp in tf:
                tfs[j, site] = amp
            tfs[j] /= np.linalg.norm(tfs[j])

    return tfs


def mat_exp(M):
    """Matrix exponential via eigendecomposition (batched over leading axes)."""
    eigvals, eigvecs = np.linalg.eig(M)
    U = eigvecs
    U_inv = np.linalg.inv(U)
    # Diagonal matrix of the exponentials of the eigenvalues
    exp_diagM = np.exp(eigvals)
    # Construct the matrix exponential
    expM = np.einsum(
        "...ij, ...jk -> ...ik",
        U,
        np.multiply(U_inv, exp_diagM[..., :, np.newaxis]),
    )
    return expM


def levi_civita(n, d):
    """
    Constructs the rank-n Levi-Civita tensor in dimension d.

    The Levi-Civita tensor is an antisymmetric tensor used in various
    areas of physics and mathematics, particularly in the context of
    cross products and determinants. It is defined such that its components
    are +1 for even permutations of indices, -1 for odd permutations,
    and 0 if any indices are repeated.

    Parameters
    ----------
    n : int
        Rank of the tensor (number of indices).
    d : int
        Dimension (number of possible index values).

    Returns
    -------
    np.ndarray
        Levi-Civita tensor of shape (d, d, ..., d) with n dimensions.
    """
    shape = (d,) * n
    epsilon = np.zeros(shape, dtype=int)
    # Generate all possible permutations of n indices
    for perm in permutations(range(d), n):
        sign = 1
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    sign *= -1
        epsilon[perm] = sign

    return epsilon


def kpath_distance(
    k_frac: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray
) -> np.ndarray:
    """
    Build 1D cumulative k-path distance (in 1/Å) from fractional k-points.

    Parameters
    ----------
    k_frac : (nks, 3)
        Fractional k-points (crystal coords).
    b1,b2,b3 : (3,) in 1/Å
        Reciprocal lattice basis vectors in Cartesian coords.

    Returns
    -------
    x : (nks,)
        Cumulative distance along the path.
    """
    B = np.vstack([b1, b2, b3]).T  # 3x3, columns are basis vectors
    k_cart = k_frac @ B.T  # (nks,3) Cartesian k
    dk = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    x = np.zeros(len(k_cart), dtype=float)
    x[1:] = np.cumsum(dk)
    return x


def finite_diff_coeffs(order, derivative_order=1, mode="central"):
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

    num_points = derivative_order + order

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


def finite_difference(
    M,
    axis: int,
    delta: float,
    order: int,
    *,
    mode: str = "central",
    periodic: bool = False,
):
    """
    Finite-difference derivative along a uniformly sampled axis.

    Parameters
    ----------
    M : np.ndarray
        Array containing the values to differentiate.
    axis : int
        Axis along which the derivative is taken.
    delta : float
        Sample spacing along the axis.
    order : int
        Order (number of stencil points) used in the finite-difference scheme.
    mode : {'central', 'forward', 'backward'}, optional
        Stencil type. ``"central"`` is used by default.
    periodic : bool, optional
        If ``True``, wrap the stencil across the boundaries (cyclic parameter).
        If ``False``, forward/backward stencils are used near the edges.

    Returns
    -------
    np.ndarray
        Array of the same shape (and promoted dtype) containing the derivative.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if delta == 0:
        raise ValueError("delta must be non-zero for finite differences.")

    # Move the differentiation axis to the front and promote to a real dtype
    arr = np.asarray(M)
    dtype = np.result_type(arr.dtype, np.float64)
    data = np.moveaxis(arr.astype(dtype, copy=False), axis, 0)
    n = data.shape[0]

    # Obtain the main stencil (length = window size) and the corresponding offset grid
    coeff_core, stencil_core = finite_diff_coeffs(order=order, mode=mode)
    window = len(coeff_core)

    # Check that we have enough samples; otherwise explain what the maximum feasible order is
    if periodic:
        if n < window:
            max_order = n - 1  # window size is order + 1 for these stencils
            raise ValueError(
                f"Periodic finite differences along axis {axis} need at least {window} samples "
                f"for order {order}, but only {n} were provided. "
                f"With {n} samples the largest admissible order is {max_order}."
            )
    else:
        if mode == "central":
            # Number of points required to cover the interior window plus the one-sided padding
            min_needed = 2 * window - 2  # interior window plus both one-sided pads
            if n < min_needed:
                max_order = (n + 2) // 2 - 1
                raise ValueError(
                    f"Central differences of order {order} require at least {min_needed} samples "
                    f"along axis {axis}; received {n}. "
                    f"With {n} samples the largest central order you can request is {max_order}."
                )
        else:
            # Forward/backward stencils only need the window itself
            min_needed = window
            if n < min_needed:
                max_order = n - 1
                raise ValueError(
                    f"{mode.capitalize()} differences of order {order} need at least {min_needed} samples "
                    f"along axis {axis}; received {n}. "
                    f"With {n} samples the largest admissible order for this mode is {max_order}."
                )

    # Accumulate the derivative in the re-ordered array; moveaxes will restore the layout later
    out = np.empty_like(data)

    def _apply(coeffs, values):
        """Convenience helper: contract stencil coefficients with trailing axes."""
        return np.tensordot(coeffs, values, axes=(0, -1))

    if periodic:
        # Apply the periodic stencil explicitly via np.roll so every sample sees the same window
        acc = np.zeros_like(data)
        for shift, coeff in zip(stencil_core, coeff_core):
            acc += coeff * np.roll(data, -shift, axis=0)
        out[...] = acc / delta
    else:
        if mode == "central":
            # Apply the central stencil on the interior (where it fits completely)
            half = window // 2
            windows = sliding_window_view(data, window_shape=window, axis=0)
            interior = _apply(coeff_core, windows) / delta
            out[half : n - half] = interior

            # Use one-sided forward/backward stencils near the two boundaries
            coeff_fwd, _ = finite_diff_coeffs(order=order, mode="forward")
            width_fwd = len(coeff_fwd)
            for i in range(width_fwd - 1):
                seg = data[i : i + width_fwd]
                out[i] = np.tensordot(coeff_fwd, seg, axes=(0, 0)) / delta

            coeff_bwd, _ = finite_diff_coeffs(order=order, mode="backward")
            width_bwd = len(coeff_bwd)
            for i in range(width_bwd - 1):
                seg = data[n - width_bwd - i : n - i]
                out[n - 1 - i] = np.tensordot(coeff_bwd, seg, axes=(0, 0)) / delta

        else:
            # (Pure) forward or backward mode: slide the requested window and apply the stencil directly
            windows = sliding_window_view(data, window_shape=window, axis=0)
            deriv = _apply(coeff_core, windows) / delta
            if mode == "forward":
                out[: deriv.shape[0]] = deriv
                out[deriv.shape[0] :] = deriv[-1]
            else:  # 'backward'
                out[-deriv.shape[0] :] = deriv
                out[: -deriv.shape[0]] = deriv[0]

    # Restore the original axis ordering before returning
    return np.moveaxis(out, 0, axis)


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


class PositionOperatorApproximationError(Exception):
    """
    Raised when a calculation involving the position operator is attempted
    using a tight-binding model generated by Wannier90, which neglects off-diagonal
    position operator elements.
    """

    pass


def _offdiag_approximation_warning_and_stop():
    """Raise :class:`PositionOperatorApproximationError` with guidance for W90 models."""
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
