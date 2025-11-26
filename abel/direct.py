from warnings import warn

import numpy as np
if hasattr(np, 'trapezoid'):  # numpy >= 2
    trapezoid = np.trapezoid
else:
    trapezoid = np.trapz

from . import _deprecate
from .tools.math import gradient, trapezoid_new
try:
    from .lib.direct import _cabel_direct_integral, _cabel_direct_integral_new
    cython_ext = True
except ImportError:
    cython_ext = False

###########################################################################
# direct - calculation of forward and inverse Abel transforms by direct
# numerical integration
#
# Roman Yurchak - Laboratoire LULI, Ecole Polytechnique/CNRS/CEA, France
# 07.2018: DH fixed the correction for the case where r[0] = 0
# 03.2018: DH changed the default grid from 0.5, 1.5 ... to 0, 1, 2.
# 01.2018: DH dhanged the integration method to trapz
# 12.2015: RY Added a pure python implementation
# 11.2015: RY moved to PyAbel, added more unit tests, reorganized code base
#    2012: RY first implementation in hedp.math.abel
###########################################################################


def _construct_r_grid(n, dr=None, r=None):
    """ Internal function to construct a 1D spatial grid """
    if dr is None and r is None:
        # default value, we don't care about the scaling
        # since the mesh size was not provided
        dr = 1.0

    if dr is not None and r is not None:
        raise ValueError('Both r and dr input parameters cannot be specified '
                         'at the same time')
    elif dr is None and r is not None:
        if r.ndim != 1 or r.shape[0] != n:
            raise ValueError('The input parameter r should be a 1D array '
                             f'of shape ({n},), got shape {r.shape}')
        # not so sure about this, needs verification -RY
        dr = np.gradient(r)

    else:
        if isinstance(dr, np.ndarray):
            raise NotImplementedError
        r = (np.arange(n))*dr
    return r, dr


def direct_transform(fr, dr=None, r=None, direction='inverse',
                     derivative=gradient, int_func=trapezoid,
                     correction=True, backend='C', **kwargs):
    """
    This algorithm performs a :doc:`direct computation
    <transform_methods/direct>` of the Abel transform
    integrals. When correction=False, the pixel at the lower bound of the
    integral (where y=r) is skipped, which causes a systematic error in the
    Abel transform. However, if correction=True is used, then an analytical
    transform transform is applied to this pixel, which makes the approximation
    that the function is linear across this pixel. With correction=True, the
    Direct method produces reasonable results.

    The Direct method is implemented in both Python and a compiled C version
    using Cython, which is much faster. The implementation can be selected using
    the backend argument. If the C-backend is not available, you must re-install
    PyAbel with Numpy, Cython, and a C-compiler already installed.

    By default, integration at all other pixels is performed using the
    trapezoidal rule.

    Parameters
    ----------

    fr : 1D or 2D numpy array
        input array to which direct/inverse Abel transform will be applied.
        For a 2D array, the first dimension is assumed to be the z axis and
        the second the r axis.
    dr : float
        spatial mesh resolution (optional, default to 1.0)
    r : 1D ndarray
        the spatial mesh (optional). Unusually, direct_transform should, in
        principle, be able to handle non-uniform data. However, this has not
        been regorously tested.
    direction : string
        Determines if a forward or inverse Abel transform will be applied.
        can be 'forward' or 'inverse'.
    derivative : callable
        a function that can return the derivative of the fr array
        with respect to r. (only used in the inverse Abel transform).
    int_func : function
        This function is used to complete the integration. It should resemble
        np.trapz/np.trapezoid, in that it must be callable using axis=, x=, and
        dx= keyword arguments.
    correction : boolean
        If False the pixel where the weighting function has a singular value
        (where r==y) is simply skipped, causing a systematic under-estimation
        of the Abel transform.
        If True, integration near the singular value is performed analytically,
        by assuming that the data is linear across that pixel. The accuracy
        of this approximation will depend on how the data is sampled.
    backend : string
        There are currently two implementations of the Direct transform,
        one in pure Python and one in Cython. The backend paremeter selects
        which method is used. The Cython code is converted to C and compiled,
        so this is faster.
        Can be 'C' or 'python' (case insensitive).
        'C' is the default, but 'python' will be used
        if the C-library is not available.

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
        f = derivative(f)/dr
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


def _pyabel_direct_integral(f, r, correction, int_func=trapezoid):
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

    out = np.zeros(f.shape)
    R, Y = np.meshgrid(r, r, indexing='ij')

    i_vect = np.arange(len(r), dtype=int)
    II, JJ = np.meshgrid(i_vect, i_vect, indexing='ij')
    mask = (II < JJ)

    I_sqrt = np.zeros(R.shape)
    I_sqrt[mask] = np.sqrt((Y**2 - R**2)[mask])

    I_isqrt = np.zeros(R.shape)
    I_isqrt[mask] = 1./I_sqrt[mask]

    # create a mask that just shows the first two points of the integral
    mask2 = ((II > JJ-2) & (II < JJ+1))

    for i, row in enumerate(f):  # loop over rows (z)
        P = row[None, :] * I_isqrt  # set up the integral
        out[i, :] = int_func(P, axis=1, **int_opts)  # take the integral

        # correct for the extra triangle at the start of the integral
        out[i, :] = out[i, :] - 0.5*int_func(P*mask2, axis=1, **int_opts)

    """
    Compute the correction. Here we apply an
    analytical integration of the cell with the singular value,
    assuming a piecewise linear behaviour of the data.
    The analytical abel transform for this trapezoid is:
    c0*acosh(r1/y) - c_r*y*acosh(r1/y) + c_r*sqrt(r1**2 - y**2)
    see: https://github.com/luli/hedp/blob/master/hedp/math/abel.py#L87-L104
    """
    if correction == 1:

        # precompute a few variables outside the loop:
        f_r = (f[:, 1:] - f[:, :-1])/np.diff(r)[None, :]
        isqrt = I_sqrt[II+1 == JJ]

        if r[0] < r[1]*1e-8:  # special case for r[0] = 0
            ratio = np.append(np.cosh(1), r[2:]/r[1:-1])
        else:
            ratio = r[1:]/r[:-1]

        acr = np.arccosh(ratio)

        for i, row in enumerate(f):  # loop over rows (z)
            out[i, :-1] += isqrt*f_r[i] + acr*(row[:-1] - f_r[i]*r[:-1])

    return out


def is_uniform_sampling(r):
    """
    Returns True if the array is uniformly spaced to within 1e-13.
    Otherwise False.
    """
    dr = np.diff(r)
    ddr = np.diff(dr)
    return np.allclose(ddr, 0, atol=1e-13)


def direct_transform_new(f, dr=None, r=None, direction='inverse',
                         derivative=None, integral=None, correction=True,
                         backend='C', **kwargs):
    """
    This algorithm performs a :doc:`direct computation
    <transform_methods/direct>` of the Abel transform integrals.

    The direct method is implemented both in Python and as a Cython extension,
    compiled through C to machine code and thus working much faster. The
    implementation can be selected using the **backend** argument. (See the
    installation details in README if the C backend is not available.)

    Parameters
    ----------
    f : numpy 1D or 2D darray
        input array to which the Abel transform will be applied. For a 2D
        array, the first dimension (rows) is assumed to be the :math:`z` axis,
        and the second (columns) the :math:`r` axis.
    dr : float, optional
        mesh step for uniformly sampled data (default is 1)
    r : numpy 1D array, optional
        possibly non-uniform mesh along the :math:`r` axis (default is uniform
        with **dr** step size). Must start at ``r[0] = 0`` and be strictly
        increasing.
    direction : str, optional
        ``'forward'`` or ``'inverse'`` (default): determines which Abel
        transform will be applied
    derivative : callable, optional
        function that will be called as ``derivative(f, r)`` and should return
        the derivative of ``f`` with respect to ``r`` (by default, the
        derivative is computed as :func:`numpy.gradient(f, r, axis=-1)
        <numpy.gradient>`). Only used in the inverse Abel transform.
    integral : callable, optional
        function that will be called like ``integral(f, r)`` and should return
        ``f`` integrated over ``r`` row by row. Only used by the Python backend
        (:func:`abel.tools.math.trapezoid_new` by default); the C backend
        always uses the trapezoidal rule.
    correction : bool, optional
        If ``False``, the pixel where the Abel integrand has a singularity is
        simply skipped, causing a systematic underestimation of the Abel
        transform.
        If ``True`` (default), integration near the singularity is performed
        analytically, by assuming that the data is linear across that pixel.
    backend : str, optional
        select the implementation (case-insensitive):

        ``'C'``:
            compiled Cython extension. Is faster and used by default, with a
            fallback to ``'Python'`` if the extension is not available.
        ``'Python'``:
            Python, using NumPy. Slower but allows custom **integral** and is
            always available.

        Both implementations produce identical results (within numerical
        errors).

    Returns
    -------
    out : numpy 1D or 2D array
        the forward or inverse Abel transform of **f**, with the same shape
    """
    g = np.atleast_2d(f.copy())

    if dr is not None and r is not None:
        raise ValueError('Specifying both dr and r is meaningless.')
    if r is None:  # use dr
        r = np.arange(g.shape[1], dtype=float)
        if dr is not None:
            r *= dr
    else:  # use r
        if np.ndim(r) != 1:
            raise ValueError(f'r must be a 1D array (got {r!r}).')
        if r.shape != (g.shape[1],):
            raise ValueError(f'The length of r ({r.shape[0]}) does not match '
                             f'the numer of columns in f ({g.shape[1]}).')
        if r[0] != 0:
            _deprecate('r[0] must be zero. If you imply that f is flat near '
                       'the axis, use r[0] = 0 and f[:, 0] = f[:, 1].')

    if direction == 'inverse':
        if derivative is None:
            g = np.gradient(g, r, axis=-1)
        else:
            try:
                g = derivative(g, r)
            except TypeError:
                _deprecate('Passing one-argument derivative function to '
                           'abel.direct.direct_transform() is deprecated.')
                g = derivative(g) / (r[1] - r[0])  # assuming uniform
        g /= -np.pi
    else:  # 'forward'
        g *= 2 * r[None, :]

    backend = backend.lower()

    if backend == 'c' and not cython_ext:
        warn('Cython extensions were not built, the C backend is not '
             'available! Falling back to the Python backend...',
             RuntimeWarning, stacklevel=2)
        backend = 'python'

    if backend == 'c':
        if integral is not None:
            warn('C backend ignores the integral argument; to use it, '
                 'specify backed="Python"',
                 RuntimeWarning, stacklevel=2)
        g = np.asarray(g, order='C', dtype=float)
        out = _cabel_direct_integral_new(g, r, int(correction))
    elif backend == 'python':
        out = _pyabel_direct_integral_new(g, r, correction,
                                          integral or trapezoid_new)
    else:
        raise ValueError(f'backend must be "C" or "Python" (got {backend!r})')

    if out.shape[0] == 1:
        return out[0]
    return out


def _pyabel_direct_integral_new(g, r, correction, integral):
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
        array with corresponding coordinates, indexed by columns
    correction : bool
        if ``False``, the singularity at :math:`r = x` is skipped; otherwise,
        the singularity is integrated using local linear approximation for
        :math:`g(r)`
    integral : callable
        function for numerical integration

    Returns:
    --------
    G : numpy 2D array
        array of the same shape as g, with the integral evaluated for each row
        and each x value from the r array
    """
    cols = g.shape[1]

    x = r[:, None]
    mask = r > x
    # y = sqrt(r^2 - x^2)
    y = np.zeros((cols, cols), dtype=float)
    y[mask] = np.sqrt((r**2 - x**2)[mask])
    # y^{-1} = 1 / y
    y_1 = np.zeros_like(y)
    y_1[mask] = 1 / y[mask]

    out = np.empty_like(g)

    # Integration for r > x (skipping the singularity)
    for j in range(cols - 2):
        out[:, j] = integral(g[:, j+1:] * y_1[j, j+1:], r[j+1:])
    # ?? correct for the extra triangle at the start of the integral
    out[:, -2] = integral(g[:, -2:] * y_1[-2, -2:], r[-2:]) / 2
    out[:, -1] = 0  # (last column is always singular)

    # Integration of the segment with r = x, assuming that g is linear there
    if correction:
        # slopes of g
        dg = (g[:, 1:] - g[:, :-1]) / (r[1:] - r[:-1])
        # superdiagonal of y
        yd = np.sqrt(r[1:]**2 - r[:-1]**2)
        # hyperbolic arccosines
        ach = np.append(1, np.arccosh(r[2:] / r[1:-1]))
        # add integrated segments to previous truncated integrals
        out[:, :-1] += ach * g[:, :-1] + (yd - ach * r[:-1]) * dg

    return out
