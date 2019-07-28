# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from abel.tools.polar import reproject_image_into_polar
from scipy.ndimage import map_coordinates, uniform_filter1d
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit
from scipy.linalg import hankel, inv, pascal
from scipy.special import legendre


def angular_integration(IM, origin=None, Jacobian=True, dr=1, dt=None):
    r"""Angular integration of the image.

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
        Include :math:`r\sin\theta` in the angular sum (integration).
        Also, ``Jacobian=True`` is passed to
        :func:`abel.tools.polar.reproject_image_into_polar`,
        which includes another value of ``r``, thus providing the appropriate
        total Jacobian of :math:`r^2\sin\theta`.

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
    r""" Intensity variation in the angular coordinate.

    This function is the :math:`\theta`-coordinate complement to
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
    r"""
    Evaluate anisotropy parameter :math:`\beta`, for :math:`I` vs
    :math:`\theta` data.

    .. math::

        I = \frac{\sigma_\text{total}}{4\pi} [ 1 + \beta P_2(\cos\theta) ]

    where :math:`P_2(x)=\frac{3x^2-1}{2}` is a 2nd order Legendre polynomial.

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
    r"""
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
    generated energy scale with published spectra. e.g. for O\ :sup:`−`
    photodetachment, the strongest fine-structure transition occurs at the
    electron affinity :math:`EA = 11\,784.676(7)` cm\ :math:`^{-1}`. Values for
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
        :math:`1.148427\times 10^{-5}` cm\ :math:`^{-1}/`\ pixel\ :sup:`2`
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
        conversion from wavelength (nm) to `photon_energy` (cm\ :sup:`−1`)
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
    r"""
    Class for calculating various radial distributions.

    Objects of this class hold the analysis parameters and cache some
    intermediate computations that do not depend on the image data. Multiple
    images can be analyzed (using the same parameters) by feeding them to the
    object::

        distr = Distributions(parameters)
        results1 = distr(image1)
        results2 = distr(image2)

    If analyses with different parameters are required, multiple objects can be
    used. For example, to analyze 4 quadrants independently::

        distr0 = Distributions('ll', ...)
        distr1 = Distributions('lr', ...)
        distr2 = Distributions('ur', ...)
        distr3 = Distributions('ul', ...)

        for image in images:
            Q0, Q1, Q2, Q3 = ...
            res0 = distr0(Q0)
            res1 = distr1(Q1)
            res2 = distr2(Q2)
            res3 = distr3(Q3)

    However, if all the quadrants have the same dimensions, it is more
    memory-efficient to flip them all to the same orientation and use a single
    object::

        distr = Distributions('ll', ...)

        for image in images:
            Q0, Q1, Q2, Q3 = ...
            res0 = distr(Q0)
            res1 = distr(Q1[:, ::-1])  # or np.fliplr
            res2 = distr(Q2[::-1, ::-1])  # or np.flip(Q2, (0, 1))
            res3 = distr(Q3[::-1, :])  # or np.flipud

    More concise function to calculate distributions for single images
    (without caching) are also available, see :func:`harmonics`, :func:`Ibeta`
    below.

    Parameters
    ----------
    origin : tuple of int or str
        origin of the radial distributions (the pole of polar coordinates)
        within the image.

        ``(int, int)``:
            explicit row and column indices
        ``str``:
            location string specifying the vertical and horizontal positions
            (in this order!) using the words from the following diagram::

                              left            center             right

                   top/upper  [0, 0]---------[0, n//2]--------[0, n-1]
                              |                                      |
                              |                                      |
                      center  [m//2, 0]    [m//2, n//2]    [m//2, n-1]
                              |                                      |
                              |                                      |
                bottom/lower  [m-1, 0]------[m-1, n//2]-----[m-1, n-1]

            The words can be abbreviated to their first letter each (such as
            ``'top left'`` → ``'tl'``, the space is then not required).

            ``'center center'``/``'cc'`` can also be shortened to
            ``'center'``/``'c'``.

            Examples:

                ``'center'`` or ``'cc'`` (default) for the full centered image

                ``'center left'``/``'cl'`` for the right image half, vertically
                centered

                ``'bottom left'``/``'bl'`` or ``'lower left'``/``'ll'`` for the
                upper-right image quadrant

    rmax : int or str
        largest radius to include in the distributions

        ``int``:
            explicit value
        ``'hor'``:
            fitting inside horizontally
        ``'ver'``:
            fitting inside vertically
        ``'HOR'``:
            touching horizontally
        ``'VER'``:
            touching vertically
        ``'min'``:
            minimum of ``'hor'`` and ``'ver'``, the largest area with 4 full
            quadrants
        ``'max'``:
            maximum of ``'hor'`` and ``'ver'``, the largest area with 2 full
            quadrants
        ``'MIN'`` (default):
            minimum of ``'HOR'`` and ``'VER'``, the largest area with 1 full
            quadrant (thus the largest with the full 90° angular range)
        ``'MAX'``:
            maximum of ``'HOR'`` and  ``'VER'``
        ``'all'``:
            covering all pixels (might have huge errors at large *r*, since the
            angular dependences must be inferred from very small available
            angular ranges)
    order : int
        highest order in the angular distributions. Even number ≥ 0.
    use_sin: bool
        use :math:`|\sin \theta|` weighting. This is the weight implied in
        spherical integration (for the total intensity, for example) and with
        respect to which the Legendre polynomials are orthogonal, so using it
        in the fitting procedure gives the most reasonable results even if the
        data deviates form the assumed angular behavior. It also reduces
        contributions from the centerline noise.
    weights : m × n numpy array, optional
        in addition to the optional :math:`|\sin \theta|` weighting (see
        **use_sin** above), use given weights for each pixel. The array shape
        must match the image shape.

        Parts of the image can be excluded from the fitting by assigning zero
        weights to their pixels.

        (Note: if ``use_sin=False``, a reference to this array is cached
        instead of its content, so if you modify the array between creating the
        object and using it, the results will be surprising. However, if
        needed, you can pass a copy as ``weights=weights.copy()``.)
    method : str
        numerical integration method used in the fitting procedure

        ``'nearest'``:
            each pixel of the image is assigned to the nearest radial bin. The
            fastest, but noisier (especially for high orders).
        ``'linear'`` (default):
            each pixel of the image is linearly distributed over the two
            adjacent radial bins. About twice slower than ``'nearest'``, but
            smoother.
        ``'remap'``:
            the image is resampled to a uniform polar grid, then polar pixels
            are summed over all angles for each radius. The smoothest, but
            significantly slower and might have problems with
            **rmax** > ``'MIN'`` and discontinuous weights.
    """
    def __init__(self, origin='center', rmax='MIN', order=2, use_sin=True,
                 weights=None, method='linear'):
        # remember parameters
        self.origin = origin
        self.rmax_in = rmax
        self.order = order
        self.use_sin = use_sin
        self.weights = weights
        if weights is None:
            self.shape = None
        else:
            self.shape = weights.shape
        self.method = method

        # whether precalculations are done
        self.ready = False

        # do precalculations if image size is known (from weights array)
        if self.shape is not None:
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

    def _int_linear(self, wl, wu, a=None, w=None):
        """
        Angular integration (radial binning) for 'linear' method.

        wl, wu : lower- and upper-bin weights
        a, w : arrays or None (their product is integrated)
        """
        # collect the products (if needed) in wl, wu
        if a is None:
            a = w
        elif w is not None:
            a = w * a  # (not *=)
        if a is not None:
            # (not *=)
            wl = wl * a
            wu = wu * a
        # lower bins
        res = np.bincount(self.bin.reshape(-1),
                          wl.reshape(-1),
                          self.rmax + 1)
        # upper bins (bin 0 must be skipped because it contains junk)
        res[2:] += np.bincount(self.bin.reshape(-1),
                               wu.reshape(-1),
                               self.rmax + 1)[1:-1]
        return res

    def _int_remap(self, a, w=None):
        """
        Angular integration (radial binning) for 'remap' method.

        a, w : arrays or None (their product is integrated)
        """
        # collect the product (if needed) in array a
        if a is None:
            if w is None:
                return [float(self.ntheta)]
            a = w
        elif w is not None:
            a = w * a  # (not *=)
        # sum all angles together
        return a.sum(axis=0)

    def _precalc(self, shape):
        """
        Precalculate and cache quantities and structures that do not depend on
        the image data.

        shape : (rows, columns) tuple
        """
        if self.ready and shape == self.shape:  # already done
            return
        if self.weights is not None and shape != self.shape:
            raise ValueError('Image shape {} does not match weights shape {}'.
                             format(shape, self.shape))

        height, width = self.shape = shape

        # Determine origin [row, col].
        if np.ndim(self.origin) == 1:  # explicit numbers
            row, col = self.origin
        else:  # string with codes
            if len(self.origin) == 2:
                r, c = self.origin
            elif self.origin in ['c', 'center']:
                r, c = 'c', 'c'
            else:
                try:
                    # extract first letters
                    r, c = [word[0] for word in self.origin.split()]
                except ValueError:
                    raise ValueError('Incorrect origin "{}"'.
                                     format(self.origin))
            # vertical
            if   r in ('t', 'u'): row = 0
            elif r == 'c'       : row = height // 2
            elif r in ('b', 'l'): row = height - 1
            else:
                raise ValueError('Incorrect vertical position in "{}"'.
                                 format(self.origin))
            # horizontal
            if   c == 'l': col = 0
            elif c == 'c': col = width // 2
            elif c == 'r': col = width - 1
            else:
                raise ValueError('Incorrect horizontal position in "{}"'.
                                 format(self.origin))
        # from the other side
        row_ = height - 1 - row
        col_ = width - 1 - col
        # min/max spans
        hor, HOR = min(col, col_), max(col, col_)
        ver, VER = min(row, row_), max(row, row_)

        # Determine rmax.
        rmax_in = self.rmax_in
        if isinstance(rmax_in, int):
            rmax = rmax_in
        elif rmax_in == 'hor': rmax = hor
        elif rmax_in == 'ver': rmax = ver
        elif rmax_in == 'HOR': rmax = HOR
        elif rmax_in == 'VER': rmax = VER
        elif rmax_in == 'min': rmax = min(hor, ver)
        elif rmax_in == 'max': rmax = max(hor, ver)
        elif rmax_in == 'MIN': rmax = min(HOR, VER)
        elif rmax_in == 'MAX': rmax = max(HOR, VER)
        elif rmax_in == 'all': rmax = int(np.sqrt(HOR**2 + VER**2))
        else:
            raise ValueError('Incorrect radial range "{}"'.format(rmax_in))
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
                    self.regions.append(list(zip(slices_row(r),
                                                 slices_col(c))))
            self.fold = True

        if self.order < 0 or self.order % 2:
            raise ValueError('Incorrect order={}'.format(self.order))

        if self.method in ['nearest', 'linear']:
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
            else:  # 'linear'
                self.bin = r.astype(int)  # round down (floor)
            self.bin[self.bin > rmax] = 0  # r = 0 is useless anyway

            # Powers of cosine.
            # c[n] is cos^2n, with 2n up to 2×order
            self.c = [None]  # (actually 1s, but never used)
            if self.order > 0:
                r2[0, 0] = np.inf  # (avoid division by zero)
                self.c.append(y2 / r2)  # cos^2 theta
                for n in range(2, self.order + 1):
                    self.c.append(self.c[1] * self.c[n - 1])

            # Weights.
            if self.weights is None:
                if self.fold:
                    # count overlapping pixels
                    Qw = np.zeros((Qheight, Qwidth))
                    for src, dst in self.regions:
                        Qw[dst] += 1
                else:
                    Qw = None
            else:  # array
                if self.fold:
                    # sum all source regions into one quadrant
                    Qw = np.zeros((Qheight, Qwidth))
                    for src, dst in self.regions:
                        Qw[dst] += self.weights[src]
                else:
                    Qw = self.weights[self.flip_row, self.flip_col]

            if self.use_sin:
                r[0, 0] = np.inf  # (avoid division by zero)
                self.Qsin = x / r
                r[0, 0] = 0  # (restore)
                if Qw is None:
                    Qw = self.Qsin
                else:
                    Qw = self.Qsin * Qw  # (not *=)

            if self.method == 'linear':
                # weights for upper and lower bins
                self.wu = r - self.bin
                self.wl = 1 - self.wu

            # Integrals.
            if self.method == 'nearest':
                pc = [self._int_nearest(None, Qw).astype(float)]
                for c in self.c[1:]:
                    pc.append(self._int_nearest(c, Qw))
            else:  # 'linear'
                wu, wl = self.wu, self.wl
                pc = [self._int_linear(wl, wu, None, Qw)]
                for c in self.c[1:]:
                    pc.append(self._int_linear(wl, wu, c, Qw))
            pc = np.array(pc).T  # [r, n]

        elif self.method == 'remap':
            # Coordinates.
            # angular step ~ 1 pixel at rmax
            self.ntheta = int(rmax * np.pi / 2)
            # polar coordinates
            r = np.linspace(0, rmax, rmax + 1)
            theta = np.linspace(0, np.pi / 2, self.ntheta,
                                endpoint=False)[:, None]
            # rectangular coordinates of polar grid
            self.grid = np.array([r * np.cos(theta), r * np.sin(theta)])

            # Powers of cosine.
            # c[n] is cos^2n, with 2n up to 2×order
            self.c = [None]  # (actually 1s, but never used)
            if self.order > 0:
                self.c.append(np.cos(theta)**2)
                for n in range(2, self.order + 1):
                    self.c.append(self.c[1] * self.c[n - 1])

            # Weights.
            if self.weights is None:
                if self.fold:
                    # count overlapping pixels
                    Qw = np.zeros((Qheight, Qwidth))
                    for src, dst in self.regions:
                        Qw[dst] += 1
                elif rmax > min(HOR, VER):
                    Qw = np.ones((Qheight, Qwidth))
                else:
                    Qw = None
            else:  # array
                if self.fold:
                    # sum all source regions into one quadrant
                    Qw = np.zeros((Qheight, Qwidth))
                    for src, dst in self.regions:
                        Qw[dst] += self.weights[src]
                else:
                    Qw = self.weights[self.flip_row, self.flip_col]
            if Qw is not None:
                Qw = map_coordinates(Qw, self.grid)

            if self.use_sin:
                self.Qsin = np.sin(theta)
                if Qw is None:
                    Qw = self.Qsin
                else:
                    Qw *= self.Qsin  # (here Qw is not aliased)

            # Integrals.
            pc = [self._int_remap(None, Qw)]
            for c in self.c[1:]:
                pc.append(self._int_remap(c, Qw))
            pc = np.array(pc).T  # [r, n]

        else:
            raise ValueError('Incorrect method "{}"'.format(self.method))

        # Conversion matrices (integrals → coofficients).
        # Some r might have too few pixels and thus produce lower-rank
        # matrices. For them the highest-rank inverse is computed (including
        # as many lower orders as possible).

        # linalg.inv is very inefficient for small matrices (takes longer than
        # everything else), so they are inverted manually.
        # inv2 and inv3 are for 2×2 and 3×3 Hankel matrices (passed as unique
        # elements).
        def inv2(p):
            p0, p1, p2 = p
            d = p0 * p2 - p1 * p1
            if d == 0:
                C = np.zeros((2, 2))
                if p0 != 0:
                    C[0, 0] = 1 / p0
                return C
            return 1/d * np.array([[p2, -p1],
                                   [-p1, p0]])

        def inv3(p):
            p0, p1, p2, p3, p4 = p
            C00 = p2 * p4 - p3 * p3
            C01 = p2 * p3 - p1 * p4
            C02 = p1 * p3 - p2 * p2
            d = p0 * C00 + p1 * C01 + p2 * C02
            if d == 0:
                C = np.zeros((3, 3))
                C[:2, :2] = inv2(p[:3])
                return C
            C11 = p0 * p4 - p2 * p2
            C12 = p1 * p2 - p0 * p3
            C22 = p0 * p2 - p1 * p1
            return 1/d * np.array([[C00, C01, C02],
                                   [C01, C11, C12],
                                   [C02, C12, C22]])

        def invn(P):
            n = self.order // 2
            C = np.zeros((n + 1, n + 1))
            for m in range(n, 0, -1):
                try:
                    C[:m + 1, :m + 1] = inv(P[:m + 1, :m + 1])
                    # (this is faster than np.pad)
                    return C
                except np.linalg.LinAlgError:
                    pass  # try lower rank
            # rank <= 1
            if P[0, 0] != 0:
                C[0, 0] = 1 / P[0, 0]
            return C

        if self.order == 0:
            pc[pc == 0] = np.inf  # to obtain inv([[0]]) = [[0]]
            self.C = 1 / pc[:, :, None]  # (new dimension to make matrices)
        elif self.order == 2:
            self.C = np.array([inv2(p) for p in pc])
        elif self.order == 4:
            self.C = np.array([inv3(p) for p in pc])
        else:
            self.C = np.array([invn(hankel(p[:self.order // 2 + 1],
                                           p[self.order // 2:]))
                               for p in pc])

        if self.method in ['nearest', 'linear']:
            # bin r = 0 contains junk (all pixels > rmax), so ignore it
            self.C[0] = 0.0

        self.ready = True

    class Results(object):
        r"""
        Class for holding the results of image analysis.

        :meth:`Distributions.image` returns an object of this class, from which
        various distributions can be retrieved using the methods described
        below, for example::

            distr = Distributions(...)
            res = distr(IM)
            harmonics = res.harmonics()

        All distributions are returned as 2D arrays with the rows (1st index)
        corresponding to particular terms of the expansion and the columns (2nd
        index) corresponding to the radii. The terms can be easily separated
        like ``I, beta2, beta4 = res.Ibeta()``. Python 3 users can also collect
        all :math:`\beta` parameters as ``I, *beta = res.Ibeta()`` for any
        **order**. Alternatively, transposing the results as ``Ibeta =
        res.Ibeta().T`` allows accessing all terms :math:`\big(I(r),
        \beta_2(r), \beta_4(r), \dots\big)` at particular radius *r* as
        ``Ibeta[r]``.

        Attributes
        ----------
        r : numpy array
            radii from 0 to **rmax**
        """
        def __init__(self, r, cn):
            self.r = r
            self.cn = cn

        def cos(self):
            r"""
            Radial distributions of :math:`\cos^n \theta` terms
            (0 ≤ *n* ≤ **order**).

            (You probably do not need them.)

            Returns
            -------
            cosn : (# terms) × (rmax + 1) numpy array
                radial dependences of the :math:`\cos^n \theta` terms, ordered
                from the lowest to the highest power
            """
            return self.cn

        def rcos(self):
            """
            Same as :meth:`cos`, but prepended with the radii row.
            """
            return np.vstack((self.r, self.cn))

        def cossin(self):
            r"""
            Radial distributions of
            :math:`\cos^n \theta \cdot \sin^m \theta` terms
            (*n* + *m* = **order**).

            For **order** = 0:

                :math:`\cos^0 \theta` is the total intensity.

            For **order** = 2

                :math:`\cos^2 \theta` corresponds to “parallel” (∥)
                transitions,

                :math:`\sin^2 \theta` corresponds to “perpendicular” (⟂)
                transitions.

            For **order** = 4

                :math:`\cos^4 \theta` corresponds to ∥,∥,

                :math:`\cos^2 \theta \cdot \sin^2 \theta` corresponds
                to ∥,⟂ and ⟂,∥.

                :math:`\sin^4 \theta` corresponds to ⟂,⟂.

            And so on.

            Notice that higher orders can represent lower orders as well:

               :math:`\cos^2 \theta + \sin^2 \theta = \cos^0 \theta
               \quad` (∥ + ⟂ = 1),

               :math:`\cos^4 \theta + \cos^2 \theta \cdot \sin^2 \theta
               = \cos^2 \theta \quad` (∥,∥ + ∥,⟂ = ∥,∥ + ⟂,∥ = ∥),

               :math:`\cos^2 \theta \cdot \sin^2 \theta + \sin^4
               \theta = \sin^2 \theta \quad` (∥,⟂ + ⟂,⟂ = ⟂,∥ + ⟂,⟂ = ⟂),

               and so forth.

            Returns
            -------
            cosnsinm : (# terms) × (rmax + 1) numpy array
                radial dependences of the :math:`\cos^n \theta \cdot \sin^m
                \theta` terms, ordered from the highest :math:`\cos \theta`
                power to the highest :math:`\sin \theta` power
            """
            # conversion matrix (cos^k → cos^n sin^m)
            CS = pascal(self.cn.shape[0], 'upper')[:, ::-1]
            # apply to all radii
            cs = CS.dot(self.cn)
            return cs

        def rcossin(self):
            """
            Same as :meth:`cossin`, but prepended with the radii row.
            """
            return np.vstack((self.r, self.cossin()))

        def harmonics(self):
            r"""
            Radial distributions of spherical harmonics
            (Legendre polynomials :math:`P_n(\cos \theta)`).

            Spherical harmonics are orthogonal with respect to integration over
            the full sphere:

            .. math::
                \iint P_n P_m \,d\Omega =
                \int_0^{2\pi} \int_0^\pi P_n(\cos \theta) P_m(\cos \theta)
                    \,\sin\theta d\theta \,d\varphi = 0

            for *n* ≠ *m*; and :math:`P_0(\cos \theta)` is the spherically
            averaged intensity.

            Returns
            -------
            Pn : (# terms) × (rmax + 1) numpy array
                radial dependences of the :math:`P_n(\cos \theta)` terms
            """
            # conversion matrix (cos^k → P_n)
            terms = self.cn.shape[0]
            CH = np.zeros((terms, terms))
            for i in range(terms):
                CH[:i + 1, i] = legendre(2 * i).c[::-2]
            CH = inv(CH)
            # apply to all radii
            harm = CH.dot(self.cn)
            return harm

        def rharmonics(self):
            """
            Same as :meth:`harmonics`, but prepended with the radii row.
            """
            return np.vstack((self.r, self.harmonics()))

        def Ibeta(self, window=1):
            r"""
            Radial intensity and anisotropy distributions.

            A cylindrically symmetric 3D intensity distribution can be expanded
            over spherical harmonics (Legendre polynomials
            :math:`P_n(\cos \theta)`) as

            .. math::
                I(r, \theta, \varphi) \, d\Omega =
                \frac{1}{4\pi} I(r) \big[1 + \beta_2(r) P_2(\cos \theta) +
                                         \beta_4(r) P_4(\cos \theta) +
                                         \dots\big],

            where :math:`I(r)` is the “radial intensity distribution”
            integrated over the full sphere:

            .. math::
                I(r) = \int_0^{2\pi} \int_0^\pi I(r, \theta, \varphi)
                       \,r^2 \sin\theta d\theta \,d\varphi,

            and :math:`\beta_n(r)` are the dimensionless “anisotropy
            parameters” describing relative contributions of each harmonic
            order (:math:`\beta_0(r) = 1` by definition). In particular:

                :math:`\beta_2 = 2` for the :math:`\cos^2 \theta` (∥)
                angular distribution,

                :math:`\beta_2 = 0` for the isotropic distribution,

                :math:`\beta_2 = -1` for the :math:`\sin^2 \theta` (⟂)
                angular distribution.

            The radial intensity distribution alone for data with arbitrary
            angular variations can be obtained by using ``weight='sin'`` and
            ``order=0``.

            Parameters
            ----------
            window : int
                window size in pixels for radial averaging of :math:`\beta`.
                Since anisotropy parameters are non-linear, the central moving
                average is applied to the harmonics (which are linear), and
                then :math:`\beta` is calculated from them. In case of well
                separated peaks, setting **window** to the peak width will
                result in :math:`\beta` values at peak centers equal to total
                peak anisotropies (beware of the background, however).

            Returns
            -------
            Ibeta : (# terms) × (rmax + 1) numpy array
                radial intensity distribution (0-th term) and radial
                dependences of anisotropy parameters (other terms)
            """
            harm = self.harmonics()
            P0, Pn = np.vsplit(harm, [1])
            I = 4 * np.pi * self.r**2 * P0
            if window > 1:
                P0 = uniform_filter1d(P0, window, axis=1, mode='nearest')
                Pn = uniform_filter1d(Pn, window, axis=1, mode='nearest')
            beta = np.divide(Pn, P0, out=np.zeros_like(Pn), where=P0 != 0)
            return np.vstack((I, beta))

        def rIbeta(self, window=1):
            """
            Same as :meth:`Ibeta`, but prepended with the radii row.
            """
            return np.vstack((self.r, self.Ibeta(window)))

    def image(self, IM):
        """
        Analyze an image.

        This method can be also conveniently accessed by “calling” the object
        itself::

            distr = Distributions(...)
            Ibeta = distr(IM).Ibeta()

        Parameters
        ----------
        IM : m × n numpy array
            the image to analyze

        Returns
        -------
        results : Distributions.Results object
            the object with analysis results, from which various distributions
            can be retrieved, see :class:`Results`
        """
        # do precalculations (if needed)
        self._precalc(IM.shape)

        # apply weighting and folding
        if self.weights is not None:
            IM = self.weights * IM  # (not *=)

        if self.fold:
            Q = np.zeros((self.Qheight, self.Qwidth))
            for src, dst in self.regions:
                Q[dst] += IM[src]
        else:  # quadrant
            Q = IM[self.flip_row, self.flip_col]

        if self.method == 'remap':
            # resample to polar grid
            Q = map_coordinates(Q, self.grid)

        if self.use_sin:
            Q = self.Qsin * Q  # (not *=)

        # calculate integrals
        if self.method == 'nearest':
            p = [self._int_nearest(Q)]
            for c in self.c[1:self.order // 2 + 1]:
                p.append(self._int_nearest(Q, c))
        elif self.method == 'linear':  # 'linear'
            Ql, Qu = self.wl * Q, self.wu * Q
            p = [self._int_linear(Ql, Qu)]
            for c in self.c[1:self.order // 2 + 1]:
                p.append(self._int_linear(Ql, Qu, c))
        else:  # 'remap'
            p = [self._int_remap(Q)]
            for c in self.c[1:self.order // 2 + 1]:
                p.append(self._int_remap(Q, c))

        # convert integrals to coefficients (I(r) = C(r)·p(r) for each r)
        I = np.einsum('jik,kj->ij', self.C, p)

        # radii
        r = np.arange(self.rmax + 1)

        return self.Results(r, I)

    def __call__(self, IM):
        return self.image(IM)


def harmonics(IM, origin='cc', rmax='MIN', order=2, **kwargs):
    """
    Convenience function to calculate harmonic distributions for a single
    image. Equivalent to ``Distributions(...).image(IM).harmonics()``.

    Notice that this function does not cache intermediate calculations, so
    using it to process multiple images is several times slower than through a
    :class:`Distributions` object.
    """
    return Distributions(origin, rmax, order, **kwargs).image(IM).harmonics()


def rharmonics(IM, origin='cc', rmax='MIN', order=2, **kwargs):
    """
    Same as :func:`harmonics`, but prepended with the radii row.
    """
    return Distributions(origin, rmax, order, **kwargs).image(IM).rharmonics()


def Ibeta(IM, origin='cc', rmax='MIN', order=2, window=1, **kwargs):
    """
    Convenience function to calculate radial intensity and anisotropy
    distributions for a single image. Equivalent to
    ``Distributions(...).image(IM).Ibeta(window)``.

    Notice that this function does not cache intermediate calculations, so
    using it to process multiple images is several times slower than through a
    :class:`Distributions` object.
    """
    return Distributions(origin, rmax, order, **kwargs).image(IM).Ibeta(window)


def rIbeta(IM, origin='cc', rmax='MIN', order=2, window=1, **kwargs):
    """
    Same as :func:`Ibeta`, but prepended with the radii row.
    """
    return Distributions(origin, rmax, order, **kwargs).image(IM).rIbeta(window)
