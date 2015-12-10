# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .math import gradient
from .lib.direct import _cabel_integrate, _cabel_integrate_naive

###########################################################################
# direct - calculation of forward and inverse Abel transforms by direct
# numerical integration
#
# Roman Yurchak - Laboratoire LULI, Ecole Polytechnique/CNRS/CEA, France
#
# 2012: First implementation in hedp.math.abel
# 2015: Moved to PyAbel, added more unit tests, reorganized code base
###########################################################################


def iabel_direct(fr, dr=None, r=None, derivative=gradient, naive=False):
    """
    Returns inverse Abel transform.

    """
    return _abel_transform_wrapper(fr, dr=dr, r=r, inverse=True,
            naive=naive)


def fabel_direct(fr, dr=None, r=None, naive=False):
    """
    Returns the direct Abel transform of a function
    sampled at discrete points.

    This algorithm does a direct computation of the Abel transform:
      * integration near the singular value is done analytically
      * integration further from the singular value with the Simpson
        rule.

    Parameters:
    fr:  1d or 2d numpy array
        input array to which direct/inversed Abel transform will be applied.
        For a 2d array, the first dimension is assumed to be the z axis and
        the second the r axis.
    dr: float
        space between samples

    Returns
    out: 1d or 2d numpy array of the same shape as fr
        with either the direct or the inverse abel transform.
    """
    return _abel_transform_wrapper(fr, dr=dr, r=r, inverse=False,
            naive=naive)


def _construct_r_grid(n, dr=None, r=None):
    """ Internal function to construct a 1D spatial grid """
    if dr is None and r is None:
        # default value, we don't care about the scaling since the mesh size was not provided
        dr = 1.0

    if dr is not None and r is not None:
        raise ValueError('Both r and dr input parameters cannot be specified at the same time')
    elif dr is None and r is not None:
        if r.ndim != 1 or r.shape[0] != n:
            raise ValueError('The input parameter r should be a 1D array'
                             'of shape = ({},), got shape = {}'.format(n, r.shape))
        # not so sure about this, needs verification
        dr = np.gradient(r) 

    else:
        if isinstance(dr, np.ndarray):
            raise NotImplementedError
        r = (np.arange(n) + 0.5)*dr
    return r, dr



def _abel_transform_wrapper(fr, dr=None, r=None, inverse=False,
                                derivative=gradient, naive=False):
    """
    Returns the forward or the inverse Abel transform of a function
    sampled using direct integration.

    This algorithm does a direct computation of the Abel transform:
      * integration near the singular value is done analytically
      * integration further from the singular value with the Simpson
        rule.

    Parameters:
    fr:  1d or 2d numpy array
        input array to which direct/inversed Abel transform will be applied.
        For a 2d array, the first dimension is assumed to be the z axis and
        the second the r axis.
    dr: float
        spatial mesh resolution           (optional, default to 1.0)
    f : 1D ndarray with the spatial mesh  (optional)
    derivative: a function that can return the derivative of the fr array with respect to r
    inverse: boolean
        If True inverse Abel transform is applied, otherwise use a forward Abel transform.
    naive: if True integration is performed with the Simpson rule,
           the pixel where the weighting function has a singular value is ignored
           if False, in addition to the integration with the Simpson rule, integration near
               the singular value is done analytically, assuming a piecewise linear data. 

    Returns
    out: 1d or 2d numpy array of the same shape as fr
        with either the direct or the inverse abel transform.
    """
    f = np.atleast_2d(fr.copy())

    r, dr = _construct_r_grid(f.shape[1], dr=dr, r=r)

    if inverse:
        # a derivative function must be provided
        f = derivative(f)/dr
        ## setting the derivative at the origin to 0
        #f[:,0] = 0



    if inverse:
        f *= - 1./np.pi
    else:
        f *= 2*r

    f = np.asarray(f, order='C', dtype='float64')

    if not naive:
        out = _cabel_integrate(f, r)
    else:
        out = _cabel_integrate_naive(f, r)


    if f.shape[0] == 1:
        return out[0]
    else:
        return out


def _abel_sym():
    """
    Analytical integration of the cell near the singular value in the abel transform
    The resulting formula is implemented in abel.lib.direct.abel_integrate
    """
    from sympy import symbols, simplify, integrate, sqrt
    from sympy.assumptions.assume import global_assumptions
    r, y,r0, r1,r2, z,dr, c0, c_r, c_rr,c_z, c_zz, c_rz = symbols(
            'r y r0 r1 r2 z dr c0 c_r c_rr c_z c_zz c_rz', positive=True)
    f0, f1, f2 = symbols('f0 f1 f2')
    global_assumptions.add(Q.is_true(r>y))
    global_assumptions.add(Q.is_true(r1>y))
    global_assumptions.add(Q.is_true(r2>y))
    global_assumptions.add(Q.is_true(r2>r1))
    P = c0 + (r-y)*c_r #+ (r-r0)**2*c_rr
    K_d = 1/sqrt(r**2-y**2)
    res = integrate(P*K_d, (r,y, r1))
    sres= simplify(res)
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
