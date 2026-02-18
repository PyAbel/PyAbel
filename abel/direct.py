from warnings import warn

import numpy as np

from . import _deprecate, _deprecated
from .tools.math import gradient, trapezoid
try:
    from .lib.direct import _cabel_direct_integral
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

def direct_transform(f, dr=None, r=None, direction='inverse', derivative=None,
                     int_func=_deprecated, integral=None, correction=True,
                     background=0, backend='C', **kwargs):
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
        possibly non-uniform mesh along the :math:`r` axis (default is uniform,
        starting at 0 and with **dr** step size). Must be strictly increasing.
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
        (:func:`abel.tools.math.trapezoid` by default); the C backend
        always uses the trapezoidal rule.
    correction : bool, optional
        If ``False``, the pixel where the Abel integrand has a singularity is
        simply skipped, causing a systematic underestimation of the Abel
        transform.
        If ``True`` (default), integration near the singularity is performed
        analytically, by assuming that the data is linear across that pixel.
    background : float or None, optional
        Direct application of the inverse Abel transform uses the derivative of
        **f**, meaning that non-zero intensity of the outermost pixel would be
        effectively subtracted from the whole row. This was the behavior in
        PyAbel < 0.10.0 and can be reproduced by ``background=None``. However,
        usual assumptions are that the function outside the input range is zero
        (``background=0``, current default). Other values can be passed to
        subtract a non-zero background.

        The forward transform with ``background=None`` evaluates the Abel
        integral only to the center of the outermost pixel, thus missing its
        remaining intensity if it is non-zero (behavior in PyAbel < 0.10.0).
        Using ``background=0`` (current default) extends the integral by
        another step, where the intensity linearly drops to zero.
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
    f = np.atleast_2d(f)
    cols = f.shape[1]
    if background is not None:
        f = np.pad(f, ((0, 0), (0, 1)), constant_values=background)

    if dr is not None and r is not None:
        raise ValueError('Specifying both dr and r is meaningless.')
    if r is None:  # use dr
        r = np.arange(f.shape[1], dtype=float)  # (optionally padded width)
        if dr is not None:
            r *= dr
    else:  # use r
        if np.ndim(r) != 1:
            raise ValueError(f'r must be a 1D array (got {r!r}).')
        if r.shape[0] != cols:  # (original width)
            raise ValueError(f'The length of r ({r.shape[0]}) does not match '
                             f'the numer of columns in f ({cols}).')
        if background is not None:
            # extend by one step of the same size as the last one
            r = np.append(r, 2 * r[-1] - r[-2])

    if direction == 'inverse':
        if derivative is None:
            g = np.gradient(f, r, axis=-1)
        else:
            try:
                g = derivative(f, r)
            except TypeError:
                _deprecate('Passing one-argument derivative function to '
                           'abel.direct.direct_transform() is deprecated.')
                g = derivative(f) / (r[1] - r[0])  # assuming uniform
        g /= -np.pi
    else:  # 'forward'
        g = 2 * r[None, :] * f

    backend = backend.lower()

    if backend == 'c' and not cython_ext:
        print('Cython extensions were not built, the C backend is not '
              'available! Falling back to the Python backend...')
        backend = 'python'

    if backend == 'c':
        if integral is not None:
            warn('C backend ignores the integral argument; to use it, '
                 'specify backed="Python"',
                 RuntimeWarning, stacklevel=2)
        g = np.asarray(g, order='C', dtype=float)
        out = _cabel_direct_integral(g, r, int(correction))
    elif backend == 'python':
        if int_func is not _deprecated:
            deprecate('abel.direct.direct_transform() argument "int_func" '
                      'is deprecated, use "integral" instead.')
            integral = int_func
        out = _pyabel_direct_integral(g, r, correction, integral or trapezoid)
    else:
        raise ValueError(f'backend must be "C" or "Python" (got {backend!r})')

    if out.shape[0] == 1:
        return out[0, :cols]
    return out[:, :cols]


def _pyabel_direct_integral(g, r, correction, integral):
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
    # TODO: this "correct for the extra triangle at the start of the integral"
    # from the old implementation is probably an artifact
    out[:, -2] = integral(g[:, -2:] * y_1[-2, -2:], r[-2:]) / 2
    out[:, -1] = 0  # (last column is always singular)

    # Integration of the segment with r = x, assuming that g is linear there
    if correction:
        # slopes of g
        dg = (g[:, 1:] - g[:, :-1]) / (r[1:] - r[:-1])
        # superdiagonal of y
        yd = np.sqrt(r[1:]**2 - r[:-1]**2)
        # hyperbolic arccosines
        ach = np.append(np.arccosh(r[1] / r[0]) if r[0] else 1,
                        np.arccosh(r[2:] / r[1:-1]))
        # add integrated segments to previous truncated integrals
        out[:, :-1] += ach * g[:, :-1] + (yd - ach * r[:-1]) * dg

    return out
