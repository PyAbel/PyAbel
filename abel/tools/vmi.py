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

    Returns the one-dimensional intensity profile as a function of the
    radial coordinate.

    Note: the use of Jacobian=True applies the correct Jacobian for the
    integration of a 3D object in spherical coordinates.

    Parameters
    ----------
    IM : 2D numpy.array
        The data image.

    origin : tuple
        Image center coordinate relative to *bottom-left* corner
        defaults to ``rows//2, cols//2``.

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
        Theta coordinate grid spacing in radians.
        if ``dt=None``, dt will be set such that the number of theta values
        is equal to the height of the image (which should typically ensure
        good sampling.)

    Returns
    ------
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
    of a 3D image. It is equivalent to calling
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
      one-dimensional intensity profile as a function of the radial coordinate.

    """
    R, intensity = angular_integration(IM, Jacobian=False, **kwargs)
    intensity /= 2 * np.pi
    return R, intensity


def radial_integration(IM, radial_ranges=None):
    """ Intensity variation in the angular coordinate.

    This function is the :math:`\\theta`-coordinate complement to
    :func:`abel.tools.vmi.angular_integration`

    Evaluates intensity vs angle for defined radial ranges.
    Determines the anisotropy parameter for each radial range.

    See :doc:`examples/example_PAD.py <examples>`

    Parameters
    ----------
    IM : 2D numpy.array
        Image data

    radial_ranges : list of tuple ranges or int step
        tuple
            integration ranges
            ``[(r0, r1), (r2, r3), ...]``
            evaluates the intensity vs angle
            for the radial ranges ``r0_r1``, ``r2_r3``, etc.

        int
            the whole radial range ``(0, step), (step, 2*step), ..``

    Returns
    -------
    Beta : array of tuples
        (beta0, error_beta_fit0), (beta1, error_beta_fit1), ...
        corresponding to the radial ranges

    Amplitude : array of tuples
        (amp0, error_amp_fit0), (amp1, error_amp_fit1), ...
        corresponding to the radial ranges

    Rmidpt : numpy float 1d array
        radial-mid point of each radial range

    Intensity_vs_theta: 2D numpy.array
       Intensity vs angle distribution for each selected radial range.

    theta: 1D numpy.array
       Angle coordinates, referenced to vertical direction.


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

    Intensity_vs_theta = []
    radial_midpt = []
    Beta = []
    Amp = []
    for rr in radial_ranges:
        subr = np.logical_and(r >= rr[0], r <= rr[1])

        # sum intensity across radius of spectral feature
        intensity_vs_theta_at_R = np.sum(polarIM[subr], axis=0)
        Intensity_vs_theta.append(intensity_vs_theta_at_R)
        radial_midpt.append(np.mean(rr))

        beta, amp = anisotropy_parameter(theta, intensity_vs_theta_at_R)
        Beta.append(beta)
        Amp.append(amp)

    return Beta, Amp, radial_midpt, Intensity_vs_theta, theta


def anisotropy_parameter(theta, intensity, theta_ranges=None):
    """
    Evaluate anisotropy parameter :math:`\\beta`, for :math:`I` vs
    :math:`\\theta` data.

    .. math::

        I = \\frac{\sigma_\\text{total}}{4\pi} [ 1 + \\beta P_2(\cos\\theta) ]

    where :math:`P_2(x)=\\frac{3x^2-1}{2}` is a 2nd order Legendre polynomial.

    `Cooper and Zare "Angular distribution of photoelectrons"
    J Chem Phys 48, 942-943 (1968) <http://dx.doi.org/10.1063/1.1668742>`_


    Parameters
    ----------
    theta: 1D numpy array
       Angle coordinates, referenced to the vertical direction.

    intensity: 1D numpy array
       Intensity variation with angle

    theta_ranges: list of tuples
       Angular ranges over which to fit
       ``[(theta1, theta2), (theta3, theta4)]``.
       Allows data to be excluded from fit, default include all data

    Returns
    -------
    beta : tuple of floats
        (anisotropy parameter, fit error)

    amplitude : tuple of floats
        (amplitude of signal, fit error)

    """
    def P2(x):   # 2nd order Legendre polynomial
        return (3 * x * x - 1) / 2

    def PAD(theta, beta, amplitude):
        return amplitude * (1 + beta * P2(np.cos(theta)))   # Eq. (1) as above

    # angular range of data to be included in the fit
    if theta_ranges is not None:
        subtheta = np.ones(len(theta), dtype=bool)
        for rt in theta_ranges:
            subtheta = np.logical_and(
                subtheta, np.logical_and(theta >= rt[0], theta <= rt[1]))
        theta = theta[subtheta]
        intensity = intensity[subtheta]

    # fit angular intensity distribution
    try:
        popt, pcov = curve_fit(PAD, theta, intensity)
        beta, amplitude = popt
        error_beta, error_amplitude = np.sqrt(np.diag(pcov))
        # physical range
        if beta > 2 or beta < -1:
            beta, error_beta = np.nan, np.nan
    except:
        beta, error_beta = np.nan, np.nan
        amplitude, error_amplitude = np.nan, np.nan

    return (beta, error_beta), (amplitude, error_amplitude)


def toPES(radial, intensity, energy_cal_factor, per_energy_scaling=True,
          photon_energy=None, Vrep=None, zoom=1):
    """
    Convert speed radial coordinate into electron kinetic or electron binding
    energy.  Return the photoelectron spectrum (PES).

    This calculation uses a single scaling factor ``energy_cal_factor``
    to convert the radial pixel coordinate into electron kinetic energy.

    Additional experimental parameters: ``photon_energy`` will give the
    energy scale as electron binding energy, in the same energy units,
    while ``Vrep``, the VMI lens repeller voltage (volts), provides for a
    voltage independent scaling factor. i.e. ``energy_cal_factor`` should
    remain approximately constant.

    The ``energy_cal_factor`` is readily determined by comparing the
    generated energy scale with published spectra. e.g. for O\ :sup:`-`
    photodetachment, the strongest fine-structure transition occurs at the
    electron affinity :math:`EA = 11,784.676(7)` cm :math:`^{-1}`. Values for
    the ANU experiment are given below, see also
    `examples/example_hansenlaw.py`.

    Parameters
    ----------
    radial : numpy 1D array

        radial coordinates.

    intensity : numpy 1D array

        intensity values, at the radial array.

    energy_cal_factor : float

        energy calibration factor that will convert radius squared into energy.
        The units affect the units of the output. e.g. inputs in
        eV/pixel\ :sup:`2`, will give output energy units in eV.  A value of
        :math:`1.148427\\times 10^{-5}` cm\ :math:`^{-1}/`\ pixel\ :sup:`2`
        applies for "examples/data/O-ANU1024.txt" (with Vrep = -98 volts).

    per_energy_scaling : bool

        sets the intensity Jacobian.
        If `True`, the returned intensities correspond to an "intensity per eV"
        or "intensity per cm\ :sup:`-1` ". If `False`, the returned intensities
        correspond to an "intensity per pixel".

     Optional:

    photon_energy : None or float

        measurement photon energy. The output energy scale is then set to
        electron-binding-energy in units of `energy_cal_factor`. The
        conversion from wavelength (nm) to `photon_energy` in (cm\ :`sup:-1`\ )
        is :math:`10^{7}/\lambda` (nm) e.g. `1.0e7/812.51` for
        "examples/data/O-ANU1024.txt".

    Vrep : None or float

        repeller voltage. Convenience parameter to allow the
        `energy_cal_factor` to remain constant, for different VMI lens repeller
        voltages. Defaults to `None`, in which case no extra scaling is
        applied. e.g. `-98 volts`, for "examples/data/O-ANU1024.txt".

    zoom : float

        additional scaling factor if the input experimental image has been
        zoomed.  Default 1.

    Returns
    -------
    eKBE : numpy 1d-array of floats

        energy scale for the photoelectron spectrum in units of
        `energy_cal_factor`.  Note that the data is no-longer on
        a uniform grid.

    PES : numpy 1d-array of floats

        the photoelectron spectrum, scaled according to the
        `per_energy_scaling` input parameter.

    """

    if Vrep is not None:
        energy_cal_factor *= np.abs(Vrep) / zoom**2

    eKE = radial**2 * energy_cal_factor

    if photon_energy is not None:
        # electron binding energy
        eBKE = photon_energy - eKE
    else:
        eBKE = eKE

    # Jacobian correction to intensity, radius has been squared
    # We have E = c1 - c2 * r**2, where c1 and c2 are constants. To get thei
    # Jacobian, we find dE/dr = 2c2r. Since the coordinates are getting
    # stretched at high E and "squished" at low E, we know that we need to
    # divide by this factor.
    intensity[1:] /= (2 * radial[1:])  # 1: to exclude R = 0
    if per_energy_scaling:
        # intensity per unit energy
        intensity /= energy_cal_factor

    # sort into ascending order
    indx = eBKE.argsort()

    return eBKE[indx], intensity[indx]


class Distributions(object):
    def __init__(self, origin='cc', rmax='MIN', order=2, weight='sin',
                 method='nearest'):
        # remember parameters
        self.origin = origin
        self.rmax = rmax
        self.order = order
        if isinstance(weight, np.ndarray):
            self.weight = 'array'
            self.warray = weight
            self.shape = weight.shape
        else:
            self.weight = weight
            self.shape = None
        self.method = method

        # whether precalculations are done
        self.ready = False

        # do precalculations if image size is known (from weight array)
        if self.weight == 'array':
            self._precalc(self.shape)
        # otherwise postpone them to the first image

    # Note!
    # The following code has several expressions like
    #   A = w * A
    # instead of
    #   A *= w
    # This is intentional: these A can be aliases to or views of the original
    # image (passed by reference), and *= would modify A in place, thus
    # corrupting the image data owned by the caller.

    def _int_nearest(self, a, w=None):
        """
        Angular integration (radial binning) for 'nearest' method.
        a, w : arrays or None (their product is integrated)
        """
        # collect the product (if needed) in array a
        if a is None:
            a = w
        elif w is not None:
            a = w * a  # (not *=)
        # sum values from array a into bins given by array bin
        # (numpy.bincount() is faster than scipy.ndimage.sum())
        if a is not None:
            a = a.reshape(-1)
        # (if a is None, np.bincount assumes unit weights, as needed)
        return np.bincount(self.bin.reshape(-1), a, self.rmax + 1)

    def _precalc(self, shape):
        """
        Precalculate and cache quantities and structures that do not depend on
        the image data.
        shape : (rows, columns) tuple
        """
        if self.ready:  # already done
            return

        height, width = shape

        # Determine origin [row, col].
        if np.ndim(self.origin) == 1:  # explicit numbers
            row, col = self.origin
        else:  # string with codes
            r, c = self.origin
            # vertical
            if   r in ('t', 'u'): row = 0
            elif r == 'c'       : row = height // 2
            elif r in ('b', 'l'): row = height - 1
            else:
                raise ValueError('Incorrect vertical position "{}"'.format(r))
            # horizontal
            if   c == 'l': col = 0
            elif c == 'c': col = width // 2
            elif c == 'r': col = width - 1
            else:
                raise ValueError('Incorrect horizontal position "{}"'.
                                 format(c))
        # from the other side
        row_ = height - 1 - row
        col_ = width - 1 - col
        # min/max spans
        hor, HOR = min(col, col_), max(col, col_)
        ver, VER = min(row, row_), max(row, row_)

        # Determine rmax.
        rmax = self.rmax
        if not isinstance(rmax, int):
            if   rmax == 'hor': rmax = hor
            elif rmax == 'ver': rmax = ver
            elif rmax == 'HOR': rmax = HOR
            elif rmax == 'VER': rmax = VER
            elif rmax == 'min': rmax = min(hor, ver)
            elif rmax == 'max': rmax = max(hor, ver)
            elif rmax == 'MIN': rmax = min(HOR, VER)
            elif rmax == 'MAX': rmax = max(HOR, VER)
            elif rmax == 'all': rmax = int(np.sqrt(HOR**2 + VER**2))
            else:
                raise ValueError('Incorrect radial range "{}"'.format(rmax))
        self.rmax = rmax

        # Folding to one quadrant with origin at [0, 0]
        self.Qheight = Qheight = min(VER, rmax) + 1
        self.Qwidth = Qwidth = min(HOR, rmax) + 1
        if row in (0, height - 1) and col in (0, width - 1):
            # IM is already one quadrant, flip it to proper orientation.
            if row == 0:
                self.flip_row = slice(0, Qheight)
            else:  # row == height - 1
                self.flip_row = slice(-1, -1 - Qheight, -1)
            if col == 0:
                self.flip_col = slice(0, Qwidth)
            else:  # col == width - 1
                self.flip_col = slice(-1, -1 - Qwidth, -1)
            self.fold = False
        else:
            # Define oriented source (IM) slices as
            # neg,neg | neg,pos
            # --------+--------
            # pos,neg | pos,spo
            # (pixel [row, col] belongs to pos,pos)
            # and corresponding destination (Q) slices.
            def slices(pivot, pivot_, size, positive):
                if positive:
                    n = min(pivot_ + 1, size)
                    return (slice(pivot, pivot + n),
                            slice(0, n))
                else:  # negative
                    n = min(pivot + 1, size)
                    return (slice(-1 - (pivot_ + 1), -1 - (pivot_ + n), -1),
                            slice(1, n))

            def slices_row(positive):
                return slices(row, row_, Qheight, positive)

            def slices_col(positive):
                return slices(col, col_, Qwidth, positive)
            # 2D region pairs (source, destination) for direct indexing
            self.regions = []
            for r in (False, True):
                for c in (False, True):
                    self.regions.append(zip(slices_row(r), slices_col(c)))
            self.fold = True

        if self.order != 2:
            raise ValueError('Only order=2 is implemented')

        if self.method in ['nearest']:
            # Quadrant coordinates.
            # x row
            x = np.arange(float(Qwidth))
            # y^2 column
            y2 = np.arange(float(Qheight))[:, None]**2
            # array of r^2
            r2 = x**2 + y2
            # array of r
            r = np.sqrt(r2)

            # Radial bins.
            if self.method == 'nearest':
                self.bin = np.array(r.round(), dtype=int)
            self.bin[self.bin > rmax] = 0  # r = 0 is useless anyway

            # Powers of cosine.
            r2[0, 0] = np.inf  # (avoid division by zero)
            self.c2 = y2 / r2

            # Weights.
            if self.weight is None:
                if self.fold:
                    # count overlapping pixels
                    Qw = np.zeros((Qheight, Qwidth))
                    for src, dst in self.regions:
                        Qw[dst] += 1
                else:
                    Qw = None
            elif self.weight == 'sin':
                # fill with sin θ = x / r
                r[0, 0] = np.inf  # (avoid division by zero)
                self.warray = x / r
                r[0, 0] = 0  # (restore)
                if self.fold:
                    # sum all source regions into one quadrant
                    Qw = np.zeros((Qheight, Qwidth))
                    for src, dst in self.regions:
                        Qw[dst] += self.warray[dst]
                        # (warray is one quadrant, so source is also "dst")
                else:
                    Qw = self.warray
            elif self.weight == 'array':
                if self.fold:
                    # sum all source regions into one quadrant
                    Qw = np.zeros((Qheight, Qwidth))
                    for src, dst in self.regions:
                        Qw[dst] += self.warray[src]
                else:
                    Qw = self.warray[self.flip_row, self.flip_col]
            else:
                raise ValueError('Incorrect weight "{}"'.format(self.weight))

            # Integrals.
            c4 = self.c2 * self.c2
            if self.method == 'nearest':
                pc0 = self._int_nearest(   None, Qw)
                pc2 = self._int_nearest(self.c2, Qw)
                pc4 = self._int_nearest(     c4, Qw)

        else:
            raise ValueError('Incorrect method "{}"'.format(self.method))

        # Conversion matrices (integrals → coofficients).
        # determinants
        d = pc0 * pc4 - pc2**2
        d[0] = np.inf  # bin r = 0 contains junk (all pixels > rmax)
        d[d == 0] = np.inf  # underdetermined bins
        d = 1 / d
        # inverse matrices
        self.C = d * np.array([[ pc4, -pc2],
                               [-pc2,  pc0]])
        # reshape to array of matrices [C[r] for r in range(rmax + 1)]
        self.C = np.swapaxes(self.C, 0, 2)

        self.ready = True

    class Results(object):
        def __init__(self, r, cn):
            self.r = r
            self.cn = cn

        def cos(self):
            return self.cn

        def rcos(self):
            return np.hstack((self.r, self.cn))

        def cossin(self):
            C = np.array([[1.0, 1.0],
                          [1.0, 0.0]])
            cs = self.cn.dot(C)
            return cs

        def rcossin(self):
            return np.hstack((self.r, self.cossin()))

        def harmonics(self):
            C = np.array([[1.0, 0.0],
                          [1/3, 2/3]])
            harm = self.cn.dot(C)
            return harm

        def rharmonics(self):
            return np.hstack((self.r, self.harmonics()))

        def Ibeta(self):
            harm = self.harmonics()
            P0, Pn = np.hsplit(harm, [1])
            I = 4 * np.pi * self.r**2 * P0
            beta = np.divide(Pn, P0, out=np.zeros_like(Pn), where=P0 != 0)
            return np.hstack((I, beta))

        def rIbeta(self):
            return np.hstack((self.r, self.Ibeta()))

    def image(self, IM):
        # do precalculations (if needed)
        self._precalc(IM.shape)

        if self.method in ['nearest']:
            # apply weighting and folding
            if self.weight == 'array':
                IM = self.warray * IM  # (not *=)
            if self.fold:
                Q = np.zeros((self.Qheight, self.Qwidth))
                for src, dst in self.regions:
                    Q[dst] += IM[src]
            else:  # quadrant
                Q = IM[self.flip_row, self.flip_col]
            if self.weight == 'sin':
                Q = self.warray * Q  # (not *=)

            if self.method == 'nearest':
                p0 = self._int_nearest(Q)
                p2 = self._int_nearest(Q, self.c2)

        p = np.vstack((p0, p2)).T

        # multiply all p[i] vectors by C[i] matrices
        # [p[i].dot(C[i]) for i in range(rmax + 1)]
        I = np.einsum('ij,ijk->ik', p, self.C)

        # radii column
        r = np.arange(self.rmax + 1)[:, None]

        return self.Results(r, I)

    def __call__(self, IM):
        return self.image(IM)


def harmonics(IM, origin='cc', rmax='MIN', order=2, **kwargs):
    return Distributions(origin, rmax, order, **kwargs).image(IM).harmonics()


def rharmonics(IM, origin='cc', rmax='MIN', order=2, **kwargs):
    return Distributions(origin, rmax, order, **kwargs).image(IM).rharmonics()


def Ibeta(IM, origin='cc', rmax='MIN', order=2, **kwargs):
    return Distributions(origin, rmax, order, **kwargs).image(IM).Ibeta()


def rIbeta(IM, origin='cc', rmax='MIN', order=2, **kwargs):
    return Distributions(origin, rmax, order, **kwargs).image(IM).rIbeta()
