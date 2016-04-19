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


def angular_integration(IM, origin=None, Jacobian=True, dr=1, dt=None,
                        average=False):
    """ 
    Angular integration of the image.

    Returns the one-dimentional intensity profile as a function of the
    radial coordinate.
    
    Note: the use of Jacobian=True applies the correct Jacobian for the integration of a 3D object in spherical coordinates.

    Parameters
    ----------
    IM : 2D np.array
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

    average: bool
        If ``average=True``, return the averaged radial intensity instead.

    Returns
    -------
    r : 1D np.array
         radial coordinates

    speeds : 1D np.array
         Integrated intensity array (vs radius).

     """

    polarIM, R, T = reproject_image_into_polar(
        IM, origin, Jacobian=Jacobian, dr=dr, dt=dt)    

    dt = T[0,1] - T[0,0]

    if Jacobian:  # x r sinθ
        polarIM = polarIM * R * np.abs(np.sin(T))

    
    speeds = np.trapz(polarIM, axis=1, dx=dt)

    if average:
        speeds /= 2*np.pi

    n = speeds.shape[0]

    return R[:n, 0], speeds   # limit radial coordinates range to match speed


def radial_integration(IM, radial_ranges=None):
    """ Intensity variation in the angular coordinate.

    This function is the :math:`\\theta`-coordinate complement to 
    :func:`abel.tools.vmi.angular_integration`

    (optionally and more useful) returning intensity vs angle for defined
    radial ranges, to evaluate the anisotropy parameter.

    See :doc:`examples/example_O2_PES_PAD.py <examples>`

    Parameters
    ----------
    IM : 2D np.array
        Image data

    radial_ranges : list of tuples
        integration ranges
        ``[(r0, r1), (r2, r3), ...]``
        Evaluate the intensity vs angle
        for the radial ranges ``r0_r1``, ``r2_r3``, etc.

    Returns
    -------
    intensity_vs_theta: 2D np.array
       Intensity vs angle distribution for each selected radial range.

    theta: 1D np.array
       Angle coordinates, referenced to vertical direction.

    """

    polarIM, r_grid, theta_grid = reproject_image_into_polar(IM)

    theta = theta_grid[0, :]  # theta coordinates
    r = r_grid[:, 0]          # radial coordinates

    if radial_ranges is None:
        radial_ranges = [(0, r[-1]), ]

    intensity_vs_theta_at_R = []
    for rr in radial_ranges:
        subr = np.logical_and(r >= rr[0], r <= rr[1])

        # sum intensity across radius of spectral feature
        intensity_vs_theta_at_R.append(np.sum(polarIM[subr], axis=0))

    return np.array(intensity_vs_theta_at_R), theta


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
    theta: 1D np.array
       Angle coordinates, referenced to the vertical direction.

    intensity: 1D np.array
       Intensity variation (with angle)

    theta_ranges: list of tuples
       Angular ranges over which to fit ``[(theta1, theta2), (theta3, theta4)]``.
       Allows data to be excluded from fit

    Returns
    -------
    (beta, error_beta) : tuple of floats
        The anisotropy parameters and the errors associated with each one.
    (amplitude, error_amplitude) : tuple of floats
       Amplitude of signal and an error for each amplitude. 
       Compare this with the data to check the fit.

    """
    def P2(x):   # 2nd order Legendre polynomial
        return (3*x*x-1)/2

    def PAD(theta, beta, amplitude):
        return amplitude*(1 + beta*P2(np.cos(theta)))   # Eq. (1) as above

    # select data to be included in the fit by θ
    if theta_ranges is not None:
        subtheta = np.ones(len(theta), dtype=bool)
        for rt in theta_ranges:
            subtheta = np.logical_and(
                subtheta, np.logical_and(theta >= rt[0], theta <= rt[1]))
        theta = theta[subtheta]
        intensity = intensity[subtheta]

    # fit angular intensity distribution
    popt, pcov = curve_fit(PAD, theta, intensity)

    beta, amplitude = popt
    error_beta, error_amplitude = np.sqrt(np.diag(pcov))

    return (beta, error_beta), (amplitude, error_amplitude)
