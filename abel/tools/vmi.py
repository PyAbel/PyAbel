# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from abel.tools.polar import reproject_image_into_polar
from scipy.ndimage import map_coordinates
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit


def angular_integration(IM, origin=None, Jacobian=True, dr=1, dt=None):
    """Angular integration of the image.

    Returns the one-dimentional intensity profile as a function of the
    radial coordinate.

    Note: the use of Jacobian=True applies the correct Jacobian for the
    integration of a 3D object in spherical coordinates.

    Parameters
    ----------
    IM : 2D numpy.array
        The data image.

    origin : tuple
        Image center coordinate relative to *bottom-left* corner
        defaults to ``rows//2+rows%2,cols//2+cols%2``.

    Jacobian : boolean
        Include :math:`r\sin\\theta` in the angular sum (integration).
        Also, ``Jacobian=True`` is passed to
        :func:`abel.tools.polar.reproject_image_into_polar`,
        which includes another value of ``r``, thus providing the appropriate
        total Jacobian of :math:`r^2\sin\\theta`.

    dr : float
        Radial coordinate grid spacing, in pixels (default 1). `dr=0.5` may
        reduce pixel granularity of the speed profile.

    dt : float
        Theta coordinate grid spacing in degrees.
        if ``dt=None``, dt will be set such that the number of theta values
        is equal to the height of the image (which should typically ensure
        good sampling.)

    [eturns
    a#------
    r : 1D numpy.array
         radial coordinates

    speeds : 1D numpy.array
         Integrated intensity array (vs radius).

     """

    polarIM, R, T = reproject_image_into_polar(
        IM, origin, Jacobian=Jacobian, dr=dr, dt=dt)

    dt = T[0, 1] - T[0, 0]

    if Jacobian:  # x r sinθ
        polarIM = polarIM * R * np.abs(np.sin(T))

    speeds = np.trapz(polarIM, axis=1, dx=dt)

    n = speeds.shape[0]

    return R[:n, 0], speeds   # limit radial coordinates range to match speed


def average_radial_intensity(IM, **kwargs):
    """Calculate the average radial intensity of the image, averaged over all
    angles. This differs form :func:`abel.tools.vmi.angular_integration` only
    in that it returns the average intensity, and not the integrated intensity
    of a 3D image. It is equavalent to calling
    :func:`abel.tools.vmi.angular_integration` with
    `Jacobian=True` and then dividing the result by 2*pi.


    Parameters
    ----------
    IM : 2D numpy.array
     The data image.

    kwargs :
      additional keyword arguments to be passed to
      :func:`abel.tools.vmi.angular_integration`

    Returns
    -------
    r : 1D numpy.array
      radial coordinates

    intensity : 1D numpy.array
      one-dimentional intensity profile as a function of the radial coordinate.

    """
    R, intensity = angular_integration(IM, Jacobian=False, **kwargs)
    intensity /= 2*np.pi
    return R, intensity


def radial_integration(IM, radial_ranges=None):
    """ Intensity variation in the angular coordinate.

    This function is the :math:`\\theta`-coordinate complement to
    :func:`abel.tools.vmi.angular_integration`

    (optionally and more useful) returning intensity vs angle for defined
    radial ranges, to evaluate the anisotropy parameter.

    See :doc:`examples/example_O2_PES_PAD.py <examples>`

    Parameters
    ----------
    IM : 2D numpy.array
        Image data

    radial_ranges : list of tuple ranges or int step
        tuple integration ranges
            ``[(r0, r1), (r2, r3), ...]``
            evaluates the intensity vs angle
            for the radial ranges ``r0_r1``, ``r2_r3``, etc.
        int - the whole radial range ``(0, step), (step, 2*step), ..``

    Returns
    -------
    intensity_vs_theta: 2D numpy.array
       Intensity vs angle distribution for each selected radial range.

    theta: 1D numpy.array
       Angle coordinates, referenced to vertical direction.

    radial_midpt: numpy.array
       Array of radial positions of the mid-point of the integration range

    """

    polarIM, r_grid, theta_grid = reproject_image_into_polar(IM)

    theta = theta_grid[0, :]  # theta coordinates
    r = r_grid[:, 0]          # radial coordinates

    if radial_ranges is None:
        radial_ranges = 1
    if isinstance(radial_ranges, int):
        rr = np.arange(0, r[-1], radial_ranges)
        # @DanHickstein clever code to map ranges
        radial_ranges = list(zip(rr[:-1], rr[1:]))

    intensity_vs_theta_at_R = []
    radial_midpt = []
    for rr in radial_ranges:
        subr = np.logical_and(r >= rr[0], r <= rr[1])

        # sum intensity across radius of spectral feature
        intensity_vs_theta_at_R.append(np.sum(polarIM[subr], axis=0))
        radial_midpt.append(np.mean(rr))

    return np.array(intensity_vs_theta_at_R), theta, radial_midpt


def anisotropy_parameter(theta, intensity, theta_ranges=None):
    """
    Evaluate anisotropy parameter :math:`\\beta`, for :math:`I` vs :math:`\\theta` data.

    .. math::

        I = \\frac{\sigma_\\text{total}}{4\pi} [ 1 + \\beta P_2(\cos\\theta) ]

    where :math:`P_2(x)=\\frac{3x^2-1}{2}` is a 2nd order Legendre polynomial.

    `Cooper and Zare "Angular distribution of photoelectrons"
    J Chem Phys 48, 942-943 (1968) <http://dx.doi.org/10.1063/1.1668742>`_


    Parameters
    ----------
    theta: 1D numpy array
       Angle coordinates, referenced to the vertical direction.

    intensity: 1D or 2D numpy array
       Array of intensity variation (with angle)

    theta_ranges: list of tuples
       Angular ranges over which to fit ``[(theta1, theta2), (theta3, theta4)]``.
       Allows data to be excluded from fit, default include all data

    Returns
    -------
    Beta : array of tuple of floats
        array of tuples (anisotropy parameter, fit error)
    Amplitude : array of tuple of floats
       array of tuples (amplitude of signal, fit error)

    """
    def P2(x):   # 2nd order Legendre polynomial
        return (3*x*x-1)/2

    def PAD(theta, beta, amplitude):
        return amplitude*(1 + beta*P2(np.cos(theta)))   # Eq. (1) as above

    # angular range of data to be included in the fit
    if theta_ranges is not None:
        subtheta = np.ones(len(theta), dtype=bool)
        for rt in theta_ranges:
            subtheta = np.logical_and(
                subtheta, np.logical_and(theta >= rt[0], theta <= rt[1]))
        theta = theta[subtheta]
        intensity = intensity[subtheta]

    Beta = []
    Amp = []
    intensity = np.atleast_2d(intensity)
    for i, Ith in enumerate(intensity):
        # fit angular intensity distribution
        try:
            popt, pcov = curve_fit(PAD, theta, Ith)
            beta, amplitude = popt
            error_beta, error_amplitude = np.sqrt(np.diag(pcov))
            # physical range
            if beta > 2 or beta < -1:
                beta, error_beta = np.nan, np.nan
        except:
            beta, error_beta = np.nan, np.nan
            amplitude, error_amplitude = np.nan, np.nan
        Beta.append((beta, error_beta))
        Amp.append((amplitude, error_amplitude))

    Beta = np.array(Beta)
    Amp = np.array(Amp)

    if Beta.shape[0] == 1:
        # flatten to a vector
        Beta = Beta[0]
        Amp = Amp[0]

    return Beta, Amp


def anisotropy(AIM, radial_ranges=None):
    """Evaluates anisotropy parameter for each tuple range of the radial grid.

    Covenience function providing a single call to 'radial_integration' and
    'anisotropy_parameter'

    Note: in general specifying a radial range tuple that encompasses a spectral
    feature that has good intensity, will provide a more accurate
    anisotropy parameter

    Parameters
    ----------
    AIM : numpy 2d float array
        inverse Abel transformed image slice

    radial_ranges : list of tuple ranges or int step
        tuple integration ranges
            ``[(r0, r1), (r2, r3), ...]``
            evaluates the intensity vs angle
            for the radial ranges ``r0_r1``, ``r2_r3``, etc.
        int - the whole radial range ``(0, step), (step, 2*step), ...``

    Returns
    -------
    Beta : array of tuples
        (beta0, error_beta_fit0), (beta1, error_beta_fit1), ...
        corresponding to the radial ranges
    Amplitude : array of tuples
        (amp0, error_amp_fit0), (amp1, error_amp_fit1), ...
        corresponding to the radial ranges
    Rbar : numpy float 1d array
         radial-mid point of each radial range

    """
    IvsTheta, theta, Rbar = radial_integration(AIM, radial_ranges)
    Beta, Amp = anisotropy_parameter(theta, IvsTheta)

    return Beta, Amp, Rbar
