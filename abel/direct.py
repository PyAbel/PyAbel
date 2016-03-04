# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.integrate

from .tools.math import gradient

try:
    from .lib.direct import _cabel_direct_integral
    cython_ext = True
except (ImportError, UnicodeDecodeError):
    cython_ext = False


###########################################################################
# direct - calculation of forward and inverse Abel transforms by direct
# numerical integration
#
# Roman Yurchak - Laboratoire LULI, Ecole Polytechnique/CNRS/CEA, France
#
# 12.2015: Added a pure python implementation following a dissuasion
#                                                     with Dan Hickstein
# 11.2015: Moved to PyAbel, added more unit tests, reorganized code base
#    2012: First implementation in hedp.math.abel
###########################################################################


def simpson_rule_wrong(f, x=None, dx=None, axis=1, **args):
    """
    This function computes the Simpson rule
    https://en.wikipedia.org/wiki/Simpson%27s_rule#Python
    both in the cases of odd and even intervals which is not technically valid
    """
    if x is not None:
        raise NotImplementedError
    if axis != 1:
        raise NotImplementedError

    res = f[:, 0] + f[:, -1]
    res += 2*f[:, 1:-1:2].sum(axis=axis)
    res += 4*f[:, 2:-1:2].sum(axis=axis)
    return res*dx/3


def _construct_r_grid(n, dr=None, r=None):
    """ Internal function to construct a 1D spatial grid """
    if dr is None and r is None:
        # default value, we don't care about the scaling
        # since the mesh size was not provided
        dr = 1.0

    if dr is not None and r is not None:
        raise ValueError('Both r and dr input parameters cannot be specified \
                            at the same time')
    elif dr is None and r is not None:
        if r.ndim != 1 or r.shape[0] != n:
            raise ValueError('The input parameter r should be a 1D array'
                             'of shape = ({},), got shape = {}'.format(
                                                                n, r.shape))
        # not so sure about this, needs verification
        dr = np.gradient(r)

    else:
        if isinstance(dr, np.ndarray):
            raise NotImplementedError
        r = (np.arange(n) + 0.5)*dr
    return r, dr


def direct_transform(fr, dr=None, r=None, direction='inverse',
                     derivative=gradient,
                     int_func=simpson_rule_wrong,
                     correction=True, backend='C'):
    """
    This algorithm does a direct computation of the Abel transform

      * integration near the singular value is done analytically
      * integration further from the singular value with the Simpson
        rule.

    Parameters
    ----------

    fr : 1d or 2d numpy array
        input array to which direct/inversed Abel transform will be applied.
        For a 2d array, the first dimension is assumed to be the z axis and
        the second the r axis.
    dr : float
        spatial mesh resolution           (optional, default to 1.0)
    f : 1D ndarray
        the spatial mesh (optional)
    derivative : callable
        a function that can return the derivative of the fr array
        with respect to r
        (only used in the inverse Abel transform).
    inverse : boolean
        If True inverse Abel transform is applied,
        otherwise use a forward Abel transform.
    correction : boolean
        if False integration is performed with the Simpson rule,
        the pixel where the weighting function has a singular value is ignored
        if True in addition to the integration with the Simpson rule,
        integration near the singular value is done analytically,
        assuming a piecewise linear data.

    Returns
    -------
    out : 1d or 2d numpy array of the same shape as fr
        with either the direct or the inverse abel transform.
    """

    backend = backend.lower()
    if backend not in ['c', 'python']:
        raise ValueError
    f = np.atleast_2d(fr.copy())

    r, dr = _construct_r_grid(f.shape[1], dr=dr, r=r)

    if direction == "inverse":
        # a derivative function must be provided
        f = derivative(f)/dr
        # setting the derivative at the origin to 0
        # f[:,0] = 0
    if direction == "inverse":
        f *= - 1./np.pi
    else:
        f *= 2*r[None, :]

    if backend == 'c':
        if not cython_ext:
            print('Warning: Cython extensions were not built, \
                    the C backend is not available!')
            print('Falling back to a pure Python backend...')
            backend = 'python'
        elif not is_uniform_sampling(r):
            print('Warning: non uniform sampling is currently not \
                    supported by the C backend!')
            print('Falling back to a pure Python backend...')
            backend = 'python'

    f = np.asarray(f, order='C', dtype='float64')
    if backend == 'c':
        out = _cabel_direct_integral(f, r, int(correction))
    else:
        out = _pyabel_direct_integral(f, r, int(correction), int_func)

    if f.shape[0] == 1:
        return out[0]
    else:
        return out


def _pyabel_direct_integral(f, r, correction, int_func=simpson_rule_wrong):
    """
    Calculation of the integral  used in Abel transform
    (both direct and inverse).
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
    if correction not in [0, 1]:
        raise ValueError

    if is_uniform_sampling(r):
        int_opts = {'dx': abs(r[1] - r[0])}
    else:
        int_opts = {'x': r}

    N0 = f.shape[0]
    N1 = f.shape[1]
    out = np.zeros(f.shape)
    R, Y = np.meshgrid(r, r, indexing='ij')
    # the following 2 lines can be better written
    i_vect = np.arange(len(r), dtype=int)
    II, JJ = np.meshgrid(i_vect, i_vect, indexing='ij')
    mask = (II < JJ)

    I_sqrt = np.zeros(R.shape)
    I_sqrt[mask] = np.sqrt((Y**2 - R**2)[mask])

    I_isqrt = np.zeros(R.shape)
    I_isqrt[mask] = 1./I_sqrt[mask]

    for i, row in enumerate(f):  # loop over rows (z)
        P = row[None, :] * I_isqrt  # set up the integral
        out[i, :] = int_func(P, axis=1, **int_opts)  # take the integral
    """
    Compute the correction
    Pre-calculated analytical integration of the cell with the singular value
    Assuming a piecewise linear behaviour of the data
    c0*acosh(r1/y) - c_r*y*acosh(r1/y) + c_r*sqrt(r1**2 - y**2)
    """
    if correction == 1:
        # computing forward derivative of the data
        f_r = (f[:, 1:] - f[:, :N1-1])/np.diff(r)[None, :]

        for i, row in enumerate(f):  # loop over rows (z)
            out[i, :-1] += I_sqrt[II+1 == JJ]*f_r[i] \
                    + np.arccosh(r[1:]/r[:-1])*(row[:-1] - f_r[i]*r[:-1])

    return out


def is_uniform_sampling(r):
    """
    Returns True if the array is uniformly spaced to within 1e-13.
    Otherwise False.
    """
    dr = np.diff(r)
    ddr = np.diff(dr)
    return np.allclose(ddr, 0, atol=1e-13)


def _abel_sym():
    """
    Analytical integration of the cell near the singular value
    in the abel transform.
    The resulting formula is implemented in abel.lib.direct.abel_integrate
    """
    from sympy import symbols, simplify, integrate, sqrt
    from sympy.assumptions.assume import global_assumptions
    r, y, r0, r1, r2, z, dr, c0, c_r, c_rr, c_z, c_zz, c_rz = symbols(
            'r y r0 r1 r2 z dr c0 c_r c_rr c_z c_zz c_rz', positive=True)
    f0, f1, f2 = symbols('f0 f1 f2')
    global_assumptions.add(Q.is_true(r > y))
    global_assumptions.add(Q.is_true(r1 > y))
    global_assumptions.add(Q.is_true(r2 > y))
    global_assumptions.add(Q.is_true(r2 > r1))
    P = c0 + (r-y)*c_r  # + (r-r0)**2*c_rr
    K_d = 1/sqrt(r**2-y**2)
    res = integrate(P*K_d, (r, y, r1))
    sres = simplify(res)
    print(sres)


def reflect_array(x, axis=1, kind='even'):
    """
    Make a symmetrically reflected array with respect to the given axis
    """
    if axis == 0:
        x_sym = np.flipud(x)
    elif axis == 1:
        x_sym = np.fliplr(x)
    else:
        raise NotImplementedError

    if kind == 'even':
        fact = 1.0
    elif kind == 'odd':
        fact = -1.0
    else:
        raise NotImplementedError

    return np.concatenate((fact*x_sym, x), axis=axis)
