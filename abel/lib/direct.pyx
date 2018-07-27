# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log

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
