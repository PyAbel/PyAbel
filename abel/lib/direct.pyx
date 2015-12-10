# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
## cython: profile=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, acosh

cpdef _cabel_integrate(double [:, ::1] f, double [::1] r):
    """
    Naive calculation of the integral  used in Abel transform (both direct and inverse).
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

    # computing forward derivative
    f_r_arr = np.zeros((N0, N1))
    f_r_arr[:, :N1-1] =  (f.base[:,1:] - f.base[:,:N1-1])/dr
    cdef double [:, ::1] f_r = f_r_arr

    with nogil:
        # pre-calculate the array of r_i, r_j
        for i in range(N1):
            for j in range(0,i):
                val = sqrt(r[i]**2 - r[j]**2)
                I_sqrt[j,i] = val
                I_isqrt[j,i] = 1./val

        for i in range(N0):
            for j in range(N1):
                s = 0
                # Integrating with the Simpson rule the part of the
                # integral that is numerically stable
                s = s + f[i,j+1]*I_isqrt[j,j+1] + f[i,N1-1]*I_isqrt[j,N1-1]
                for k in range(j+2, N1-1):
                    val = f[i,k]*I_isqrt[j,k]
                    if (k-j+1) % 2 == 0:
                        s = s + 2*val
                    else:
                        s = s + 4*val
                s = s * dr / 3.

                # Integration of the cell with the singular value
                # Assuming a piecewise linear behaviour of the data
                # c0*acosh(r1/y) - c_r*y*acosh(r1/y) + c_r*sqrt(r1**2 - y**2)
                if j < N1 - 1:
                    s = s + I_sqrt[j,j+1]*f_r[i,j] \
                            + acosh(r[j+1]/r[j])*(f[i,j] - f_r[i,j]*r[j])

                out[i,j] = s

    return out.base

cpdef _cabel_integrate_naive(double [:, ::1] f, double [::1] r):
    """
    Compute the integral  used in Abel transform (both direct and inverse).
    Same as _cbel_integrate but without correction for the singularity
    (mostly for testing purposes)
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


    with nogil:
        # pre-calculate the array of r_i, r_j
        for i in range(N1):
            for j in range(0,i):
                val = sqrt(r[i]**2 - r[j]**2)
                I_sqrt[j,i] = val
                I_isqrt[j,i] = 1./val

        for i in range(N0):
            for j in range(N1):
                s = 0
                # Integrating with the Simpson rule the part of the
                # integral that is numerically stable
                s = s + f[i,j+1]*I_isqrt[j,j+1] + f[i,N1-1]*I_isqrt[j,N1-1]
                for k in range(j+2, N1-1):
                    val = f[i,k]*I_isqrt[j,k]
                    if (k-j+1) % 2 == 0:
                        s = s + 2*val
                    else:
                        s = s + 4*val
                s = s * dr / 3.

                out[i,j] = s

    return out.base




cpdef double trapz (double [::1] y, double dx, int Ny):
    cdef double val=0
    cdef int k
    for k in range(1, Ny-1):
        val = val + y[k]
    val = val + y[0]/2.
    val = val + y[Ny-1]/2.
    val = val*dx
    return val
