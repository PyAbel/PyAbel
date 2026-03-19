# cython: language_level=3
# (disable unneeded checks and adjustments to increase performance)
# cython: boundscheck=False, cdivision=True, wraparound=False
# (disable unneeded features to reduce compiled size)
# cython: always_allow_keywords=False, auto_pickle=False, binding=False

import numpy as np
from libc.math cimport sqrt, log
from libc.stdlib cimport calloc, free
from cython.parallel import parallel, prange


cpdef _cabel_direct_integral(const double[:, ::1] g, const double[::1] r,
                             const int correction):
    """
    Calculation of the integral
               ∞
               ⌠     g(r)
        G(x) = ⎮ ──────────── dr
               ⎮   _________
               ⌡  √ r² − x²
               r
    used in the forward and inverse direct Abel transforms.

    Parameters
    ----------
    g : numpy 2D array
        array with function values, indexed by (row, column)
    r : numpy 1D array
        array with coordinates corresponding to columns
    correction : int
        if zero, the singularity at r = x is skipped; otherwise, the
        singularity is integrated using local linear approximation for g(r)

    Returns:
    --------
    G : numpy 2D array
        array of the same shape as g, with the integral evaluated for each row
        and each x value from the r array
    """
    cdef Py_ssize_t rows = g.shape[0], cols = g.shape[1]

    cdef double[:, ::1] y = np.empty((cols, cols))  # y = sqrt(r^2 - x^2)
    cdef double[:, ::1] h = np.empty((cols, cols))  # h = Δr / y
    cdef double[:, ::1] G = np.empty((rows, cols))  # output

    cdef Py_ssize_t i, j, k  # loop indices (row, out col = x, in col = r)
    cdef double s  # for running sum in integration

    with nogil:
        # precompute helper arrays
        for j in range(cols - 1):
            for k in range(j + 1, cols):
                y[j, k] = sqrt(r[k]**2 - r[j]**2)

            # left edge (forward difference)
            h[j, j+1] = (r[j+1] - r[j]) / y[j, j+1]
            # inner points (central difference)
            for k in range(j + 2, cols - 1):
                h[j, k] = (r[k+1] - r[k-1]) / y[j, k]
            # right edge (backward difference)
            h[j, cols-1] = (r[cols-1] - r[cols-2]) / y[j, cols-1]
        # (last output column should skip trapezoidal integration)
        h[cols-2, cols-1] = 0

        # Parallelized loop over rows (must use "s = s + ..." instead of
        # "s += ..." because Cython interprets "+=" as parallel reduction)
        for i in prange(rows):
            for j in range(cols):  # loop over output columns (r)
                # Trapezoidal-rule integration, skipping r = x
                s = 0
                for k in range(j + 1, cols):  # loop over input columns (r > x)
                    s = s + g[i, k] * h[j, k]
                s /= 2

                if j + 1 < cols and correction:
                    # Integration of the singularity at r = x (k = j), assuming
                    # linear behaviour of g(r) between r[j] and r[j+1]
                    if r[j] == 0:  # r = 0 ==> j = 0, g[i, 0] = 0
                        s = s + g[i, 1]
                    else:
                        s = s + ((g[i, j] * r[j+1] - g[i, j+1] * r[j]) *
                                 log((r[j+1] + y[j, j+1]) / r[j]) +
                                 (g[i, j+1] - g[i, j]) * y[j, j+1]) / \
                                (r[j+1] - r[j])

                G[i, j] = s

    return G.base


cpdef _cabel_hansenlaw_recursion(const double[:, ::1] phi,
                                 const double[:, ::1] B0,
                                 const double[:, ::1] B1,
                                 const double[:, ::1] drive):
    """
    Hansen–Law recursion.

    Parameters
    ----------
    phi : numpy 2D array
    B0 : numpy 2D array or None
    B1 : numpy 2D array
        recursion coefficients, all shapes are (cols-2, K), where K = 9 is the
        expansion order; B0 is None for hold_order=0.
    drive : numpy 2D array
        driving function with shape (rows, cols), same as the input image

    Returns
    -------
    aim : numpy 2D array
        output image with shape (rows, cols), same as input, but with first and
        last columns undefined
    """
    cdef Py_ssize_t K = phi.shape[1]
    cdef int use_B0 = B0 is not None
    cdef Py_ssize_t rows, cols  # image size (= drive size)
    rows, cols = drive.shape[:2]

    cdef double[:, ::1] aim = np.empty((rows, cols))  # Abel-transformed image

    cdef Py_ssize_t row, col, i, k  # loop indices
    cdef double* x  # state (per row)
    cdef double s  # for running sum

    with nogil, parallel(num_threads=4):
        x = <double*>calloc(K, sizeof(double))  # (thread-local)
        # Parallelized loop over rows (must use "s = s + ..." instead of
        # "s += ..." because Cython interprets "+=" as parallel reduction)
        for row in prange(rows):
            for k in range(K):
                x[k] = 0
            for col in range(cols - 2, 0, -1):
                i = cols - 2 - col
                s = 0
                for k in range(K):
                    x[k] = phi[i, k] * x[k] + B1[i, k] * drive[row, col]
                    if use_B0:
                        x[k] += B0[i, k] * drive[row, col+1]
                    s = s + x[k]
                aim[row, col] = s
        free(x)

    return aim.base
