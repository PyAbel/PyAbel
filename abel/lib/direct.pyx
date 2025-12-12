# cython: language_level=3
# (disable unneeded checks and adjustments to increase performance)
# cython: boundscheck=False, cdivision=True, wraparound=False
# (disable unneeded features to reduce compiled size)
# cython: always_allow_keywords=False, auto_pickle=False, binding=False

import numpy as np
from libc.math cimport sqrt, log
from cython.parallel import prange


# define the hyperbolic arccos function since some old compilers (MSVC 2008) don't have this
cdef inline double acosh(double xx) nogil:
    return log(xx + sqrt(xx**2 - 1))


cpdef _cabel_direct_integral(double [:, ::1] f, double [::1] r, int correction):
    """
    Calculation of the integral  used in Abel transform (both direct and inverse).
             ∞                  
            ⌠                  
            ⎮      f(r)        
            ⎮ ────────────── dr
            ⎮    ___________   
            ⎮   ╱  2   2     
            ⎮ ╲╱  y - r      
            ⌡                  
            y
    Returns:
    --------
    np.array: of the same shape as f with the integral evaluated at r

    """
    cdef int i,j,k
    cdef int N0, N1
    cdef double val, dr, s

    dr = r[1] - r[0]

    N0 = f.shape[0]
    N1 = f.shape[1]
    cdef double [:, ::1] out = np.zeros((N0, N1)),\
                        I_sqrt = np.zeros((N1, N1)),\
                        I_isqrt = np.zeros((N1, N1))

    # computing forward derivative of the data
    cdef double [:, ::1] f_r = np.zeros((N0, N1))
    f_r.base[:, :N1-1] = (f.base[:,1:] - f.base[:,:N1-1])/np.diff(r)[None, :]

    with nogil:
        # pre-calculate the array of r_i, r_j
        for j in range(N1):
            for k in range(j+1, N1):
                val = sqrt(r[k]**2 - r[j]**2)
                I_sqrt[j,k] = val
                I_isqrt[j,k] = 1./val

        for i in range(N0): # loop over rows (z)
            for j in range(N1):  # loop over (r) elements
        
                s = 0
            
                # simple left Reimann sum:
                # for k in range(j+1, N1): # inner loop over elements such as r < y
                #   s = s + f[i,k] * I_isqrt[j,k] * dr
            
                # Trapezoidal rule integration, skipping r == y
                for k in range(j+1, N1-1): # inner loop over elements such as r < y
                    s += (f[i,k] * I_isqrt[j,k] + f[i,k+1] * I_isqrt[j,k+1]) * 0.5 * dr
            
                # Deal with the edge-case. Designed to match the Direct-Python implementation using np.trapz
                if j == N1-2:
                    k = j
                    s = f[i,k+1] * I_isqrt[j,k+1] * 0.25 * dr   
       
                            
                if j < N1 - 1 and correction == 1:
                   # Integration of the cell with the singular value
                   # Assuming a piecewise linear behaviour of the data
                   # c0*acosh(r1/y) - c_r*y*acosh(r1/y) + c_r*sqrt(r1**2 - y**2)
                   
                   if j == 0 and r[0]<(1e-8*r[1]):
                       s = s + I_sqrt[j,j+1]*f_r[i,j] \
                               + 1*(f[i,j] - f_r[i,j]*r[j])
                   else:
                       s = s + I_sqrt[j,j+1]*f_r[i,j] \
                               + acosh(r[j+1]/r[j])*(f[i,j] - f_r[i,j]*r[j])

                out[i,j] = s

    return out.base


cpdef _cabel_direct_integral_new(const double[:, ::1] g, const double[::1] r,
                                 const int correction):
    """
    Calculation of the integral
               ∞
               ⌠     g(r)
        G(x) = ⎮ ──────────── dr
               ⎮   _________
               ⌡  √ r² − x²
               r
    used in the forward and inverse Abel transforms.

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
        for j in range(cols):
            for k in range(j + 1, cols):
                y[j, k] = sqrt(r[k]**2 - r[j]**2)

            # left edge (forward difference)
            if j + 1 < cols:
                h[j, j+1] = (r[j+1] - r[j]) / y[j, j+1]
            # inner points (central difference)
            for k in range(j + 2, cols - 1):
                h[j, k] = (r[k+1] - r[k-1]) / y[j, k]
            # right edge (backward difference)
            h[j, cols-1] = (r[cols-1] - r[cols-2]) / y[j, cols-1]
        # ?? edge-case, to match the Direct-Python implementation using np.trapz
        h[cols-2, cols-1] /= 2

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
