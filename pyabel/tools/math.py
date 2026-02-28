import numpy as np
from scipy.optimize import curve_fit, brentq
from scipy.interpolate import interp1d
from abel import _deprecate


def gradient(f, x=None, dx=1, axis=-1):
    """
    This function is deprecated, use :func:`numpy.gradient` instead.

    .. note ::
        Results for irregular sampling were incorrect before PyAbel 0.10.0.
    """
    _deprecate('abel.tools.math.gradient() is deprecated, '
               'use numpy.gradient() instead.')
    return np.gradient(f, dx if x is None else x, axis=axis)


def trapezoid(f, x):
    """
    Trapezoidal-rule integration along each row of 2D array **f**, with
    coordinates corresponding to each column given by 1D array **x**.

    This function is a faster equivalent of :func:`numpy.trapezoid` (called
    ``numpy.trapz()`` before NumPy 2.0) and is used for integration in
    :func:`abel.direct.direct_transform` by default.

    Parameters
    ----------
    f : numpy 2D array
        integrand

    x : numpy 1D array
        coordinates corresponding to columns of **f**

    Returns
    -------
    out : numpy 1D array
        integrals for each row
    """
    if x.size < 2:
        return np.zeros(f.shape[0], dtype=f.dtype)
    dx = np.empty_like(x)
    dx[0] = x[1] - x[0]     # left endpoint: forward difference
    dx[-1] = x[-1] - x[-2]  # right endpoint: backwards difference
    if x.size > 2:          # interior points: central difference (doubled)
        dx[1:-1] = x[2:] - x[:-2]
    return f.dot(dx / 2)


def gaussian(x, a, mu, sigma, c):
    r"""
    `Gaussian function <https://en.wikipedia.org/wiki/Gaussian_function>`_

    :math:`f(x)=a e^{-(x - \mu)^2 / (2 \sigma^2)} + c`

    Parameters
    ----------
    x : 1D np.array
        coordinate

    a : float
        the height of the curve's peak

    mu : float
        the position of the center of the peak

    sigma : float
        the standard deviation, sometimes called the Gaussian RMS width

    c : float
        non-zero background

    Returns
    -------
    out : 1D np.array
        the Gaussian profile
    """
    return a * np.exp(-((x - mu) ** 2) / 2 / sigma ** 2) + c


def guess_gaussian(x):
    """
    Find a set of better starting parameters for Gaussian function fitting

    Parameters
    ----------
    x : 1D np.array
        1D profile of your data

    Returns
    -------
    out : tuple of float
        estimated value of (a, mu, sigma, c)
    """
    c_guess = (x[0] + x[-1]) / 2
    a_guess = x.max() - c_guess
    mu_guess = x.argmax()
    x_inter = interp1d(range(len(x)), x)

    def _(i):
        return x_inter(i) - a_guess / 2 - c_guess

    try:
        sigma_l_guess = brentq(_, 0, mu_guess)
    except ValueError:
        sigma_l_guess = len(x) / 4
    try:
        sigma_r_guess = brentq(_, mu_guess, len(x) - 1)
    except ValueError:
        sigma_r_guess = 3 * len(x) / 4
    return a_guess, mu_guess, (sigma_r_guess -
                               sigma_l_guess) / 2.35482, c_guess


def fit_gaussian(x):
    """
    Fit a Gaussian function to x and return its parameters

    Parameters
    ----------
    x : 1D np.array
        1D profile of your data

    Returns
    -------
    out : tuple of float
        (a, mu, sigma, c)
    """
    res = curve_fit(gaussian, np.arange(x.size), x, p0=guess_gaussian(x),
                    method='trf')  # default 'lm' is broken, see Scipy #21995
    return res[0]  # extract optimal values
