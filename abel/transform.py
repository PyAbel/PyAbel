#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import time
from warnings import warn
import platform
import os
import sys

from . import basex
from . import dasch
from . import daun
from . import direct
from . import hansenlaw
from . import linbasex
from . import onion_bordas
from . import rbasex
from . import tools

from abel import _deprecated, _deprecate


class Transform(object):
    r"""
    Abel transform image class. Also accessible as :class:`abel.Transform`.

    This class provides whole-image forward and inverse Abel
    transforms, together with preprocessing (centering, symmetrizing)
    and postprocessing (integration) functions.

    Parameters
    ----------

    IM : a N×M numpy array
        This is the image to be transformed

    direction : str
        The type of Abel transform to be performed.

        ``forward``
            A forward Abel transform takes a (2D) slice of a 3D image and
            returns the 2D projection.

        ``inverse`` (default)
            An inverse Abel transform takes a 2D projection and reconstructs
            a 2D slice of the 3D image.

    method : str
        specifies which numerical approximation to the Abel transform should be
        employed (see below). The options are

        ``basex``
            the Gaussian "basis set expansion" method of Dribinski et al.
            (2002).
        ``daun``
            the deconvolution method with Tikhonov regularization of Daun et
            al. and its extensions.
        ``direct``
            a naive implementation of the analytical formula by Roman Yurchuk.
        ``hansenlaw``
            the recursive algorithm described by Hansen and Law (1985).
        ``linbasex``
            the 1D projections of velocity-mapping images in terms of 1D
            spherical functions by Gerber et al. (2013).
        ``onion_bordas``
            the algorithm of Bordas and co-workers (1996),
            re-implemented by Rallis, Wells and co-workers (2014).
        ``onion_peeling``
            the onion peeling deconvolution as described by Dasch (1992).
        ``rbasex``
            a method similar to pBasex by Garcia et al. (2004) for
            velocity-mapping images, but with analytical basis functions
            developed by Ryazanov (2012).
        ``three_point``
            the three-point transform of Dasch (1992).
        ``two_point``
            the two-point transform of Dasch (1992).

    origin : tuple or str
        Before applying Abel transform, the image is centered around this
        point.

        If a tuple (float, float) is provided, this specifies the image origin
        in the (row, column) format. If a string is provided, an automatic
        centering algorithm is used:

        ``image_center``
            The origin is assumed to be the center of the image.
        ``convolution``
            The origin is found from autoconvolution of image projections
            along each axis.
        ``slice``
            The origin is found by comparing slices in the horizontal and
            vertical directions.
        ``com``
            The origin is calculated as the center of mass.
        ``gaussian``
            The origin is found using a fit to a Gaussian function. This only
            makes sense if your data looks like a Gaussian.
        ``none`` (default)
            No centering is performed. An image with an odd number of columns
            must be provided.

    symmetry_axis : None, int or tuple
        Symmetrize the image about the numpy axis
        0 (vertical), 1 (horizontal), (0, 1) (both axes). Note that the
        Abel transform is always performed around the vertical axis.
        This parameter only affects how the image is modified before
        (and after) applying the Abel transform. For more information,
        see the "Quadrant combining" note below.

    use_quadrants : tuple of 4 booleans
        select quadrants to be used in the analysis: (Q0, Q1, Q2, Q3).
        Quadrants are numbered counter-clockwide from upper right.
        See note below for description of quadrants.
        Default is ``(True, True, True, True)``, which uses all quadrants.

    symmetrize_method : str
        Method used for symmetrizing the image.

        ``average``
            Average the quadrants, in accordance with the **symmetry_axis**.
        ``fourier``
            Axial symmetry implies that the Fourier components of the 2D
            projection should be real. Removing the imaginary components in
            reciprocal space leaves a symmetric projection.

            K. R. Overstreet, P. Zabawa, J. Tallant, A. Schwettmann,
            J. P. Shaffer,
            "Multiple scattering and the density distribution of a Cs MOT",
            `Optics Express 13, 9672–9682 (2005)
            <https://doi.org/10.1364/OPEX.13.009672>`__.

    angular_integration : bool
        Integrate the image over angle to give the radial (speed) intensity
        distribution.

        *Note: in PyAbel ≤0.8.4 the intensity distribution was off by a factor
        of π, please keep this in mind when comparing absolute intensities.*

    transform_options : dict
        Additional arguments passed to the individual transform functions.
        See the documentation for the individual transform method for options:
        :func:`basex <abel.basex.basex_transform>`,
        :func:`daun <abel.daun.daun_transform>`,
        :func:`direct <abel.direct.direct_transform>`,
        :func:`hansenlaw <abel.hansenlaw.hansenlaw_transform>`,
        :func:`linbasex <abel.linbasex.linbasex_transform>`,
        :func:`onion_bordas <abel.onion_bordas.onion_bordas_transform>`,
        :func:`onion_peeling <abel.dasch.onion_peeling_transform>`,
        :func:`rbasex <abel.rbasex.rbasex_transform>`,
        :func:`three_point <abel.dasch.three_point_transform>`,
        :func:`two_point <abel.dasch.two_point_transform>`.

    center_options : dict
        Additional arguments to be passed to the centering function,
        see :func:`abel.tools.center.center_image()`.

    angular_integration_options : dict
        Additional arguments passed to the angular integration functions,
        see :func:`abel.tools.vmi.angular_integration_3D()`.

    recast_as_float64 : bool
        determines whether the input image should be recast to
        ``float64``. Many images are imported in other formats (such as
        ``uint8`` or ``uint16``), and this does not always play well with the
        transorm algorithms. This should probably always be set to ``True``
        (default).

    verbose : bool
        determines whether non-critical output should be printed.


    .. note::
        Quadrant combining:
        The quadrants can be combined (averaged) using the ``use_quadrants``
        keyword in order to provide better data quality.

        The quadrants are numbered starting from Q0 in the upper right and
        proceeding counter-clockwise::

            +--------+--------+
            | Q1   * | *   Q0 |
            |   *    |    *   |
            |  *     |     *  |                                 AQ1 | AQ0
            +--------o--------+ --([inverse] Abel transform)--> ----o----
            |  *     |     *  |                                 AQ2 | AQ3
            |   *    |    *   |
            | Q2  *  | *   Q3 |          AQi == [inverse] Abel transform
            +--------+--------+                 of quadrant Qi

        Three cases are possible:

        1) symmetry_axis = 0 (vertical)::

            Combine:  Q01 = Q0 + Q1, Q23 = Q2 + Q3
            inverse image   AQ01 | AQ01
                            -----o----- (left and right sides equivalent)
                            AQ23 | AQ23


        2) symmetry_axis = 1 (horizontal)::

            Combine: Q12 = Q1 + Q2, Q03 = Q0 + Q3
            inverse image   AQ12 | AQ03
                            -----o----- (top and bottom equivalent)
                            AQ12 | AQ03


        3) symmetry_axis = (0, 1) (both)::

            Combine: Q = Q0 + Q1 + Q2 + Q3
            inverse image   AQ | AQ
                            ---o---  (all quadrants equivalent)
                            AQ | AQ

    Notes
    -----
    As mentioned above, PyAbel offers several different approximations to the
    the exact Abel transform. All the methods should produce similar
    results, but depending on the level and type of noise found in the image,
    certain methods may perform better than others. Please see the
    :ref:`TransformMethods` section of the documentation for complete
    information.

    The methods marked with a * indicate methods that generate basis sets. The
    first time they are run for a new image size, it takes seconds to minutes
    to generate the basis set. However, this basis set is saved to disk can
    be reloaded, meaning that future transforms are performed much more
    quickly.

    ``basex`` *
        The "basis set exapansion" algorithm describes the data in terms of
        gaussian-like functions, which themselves can be Abel-transformed
        analytically. With the default functions, centered at each pixel,
        this method also does not make any assumption about the
        shape of the data. This method is one of the de-facto standards in
        photoelectron/photoion imaging.

        V. Dribinski, A. Ossadtchi, V. A. Mandelshtam, H. Reisler,
        "Reconstruction of Abel-transformable images: The Gaussian basis-set
        expansion Abel transform method",
        `Rev. Sci. Instrum. 73, 2634–2642 (2002)
        <https://doi.org/10.1063/1.1482156>`__.

    ``daun`` *
        Methods based on onion-peeling deconvolution using Tikhonov
        regularization described in

        K. J. Daun, K. A. Thomson, F. Liu, G. J. Smallwood,
        "Deconvolution of axisymmetric flame properties using Tikhonov
        regularization",
        `Appl. Opt. 45, 4638–4646 (2006)
        <https://doi.org/10.1364/AO.45.004638>`__.

        In addition to the original implicit step-functions basis (“onion
        peeling”) and the derivative regularization, linear and quadratic basis
        functions are implemented, as well as the :math:`L_2`-norm Tikhonov
        regularization (like in ``basex``) and non-negative least-squares
        solution.

    ``direct``
        This method attempts a direct integration of the Abel-transform
        integral. It makes no assumptions about the data (apart from
        cylindrical symmetry), but it typically requires fine sampling to
        converge. Such methods are typically inefficient, but thanks to this
        Cython implementation (by Roman Yurchuk), this "direct" method is
        competitive with the other methods.

    ``hansenlaw``
        This "recursive algorithm" produces reliable results and is quite fast
        (~0.1 s for a 1001×1001 image). It makes no assumptions about the data
        (apart from cylindrical symmetry). It tends to require that the data is
        finely sampled for good convergence.

        E. W. Hansen, P.-L. Law,
        "Recursive methods for computing the Abel transform and its inverse",
        `J. Opt. Soc. Am. A 2, 510–520 (1985)
        <https://doi.org/10.1364/JOSAA.2.000510>`__.

    ``linbasex`` *
        Velocity-mapping images are composed of projected Newton spheres with
        a common centre. The 2D images are usually evaluated by a
        decomposition into base vectors, each representing the 2D projection
        of a set of particles starting from a centre with a specific velocity
        distribution. Lin-BASEX evaluates 1D projections of VM images in terms
        of 1D projections of spherical functions, instead.

        Th. Gerber, Yu. Liu, G. Knopp, P. Hemberger, A. Bodi, P. Radi,
        Ya. Sych,
        "Charged particle velocity map image reconstruction with
        one-dimensional projections of spherical functions",
        `Rev. Sci. Instrum. 84, 033101 (2013)
        <https://doi.org/10.1063/1.4793404>`__.

    ``onion_bordas``
        The onion peeling method, also known as "back projection", originates
        from
        C. Bordas, F. Paulig,
        "Photoelectron imaging spectrometry: Principle and inversion method",
        `Rev. Sci. Instrum. 67, 2257–2268 (1996)
        <https://doi.org/10.1063/1.1147044>`__.

        The algorithm was subsequently coded in MatLab by
        C. E. Rallis, T. G. Burwitz, P. R. Andrews, M. Zohrabi, R. Averin,
        S. De, B. Bergues, B. Jochim, A. V. Voznyuk, N. Gregerson, B. Gaire,
        I. Znakovskaya, J. McKenna, K. D. Carnes, M. F. Kling, I. Ben-Itzhak,
        E. Wells,
        "Incorporating real time velocity map image reconstruction into
        closed-loop coherent control",
        `Rev. Sci. Instrum. 85, 113105 (2014)
        <https://doi.org/10.1063/1.4899267>`__,
        which was used as the basis of this Python port. See `issue #56
        <https://github.com/PyAbel/PyAbel/issues/56>`__.

    ``onion_peeling`` *
        This is one of the most compact and fast algorithms, with the inverse
        Abel transform achieved in one Python code-line, `PR #155
        <https://github.com/PyAbel/PyAbel/pull/155>`__. See also
        ``three_point`` is the onion peeling algorithm as described by Dasch
        (1992), reference below.

    ``rbasex`` *
        The pBasex method by
        G. A. Garcia, L. Nahon, I. Powis,
        “Two-dimensional charged particle image inversion using a polar basis
        function expansion”,
        `Rev. Sci. Instrum. 75, 4989–2996 (2004)
        <https://doi.org/10.1063/1.1807578>`__
        adapts the BASEX ("basis set expansion") method to the specific case of
        velocity-mapping images by using a basis of 2D functions in polar
        coordinates, such that the reconstructed radial distributions are
        obtained directly from the expansion coefficients.

        This method employs the same approach, but uses more convenient basis
        functions, which have analytical Abel transforms separable into radial
        and angular parts, developed in
        M. Ryazanov,
        “Development and implementation of methods for sliced velocity map
        imaging. Studies of overtone-induced dissociation and isomerization
        dynamics of hydroxymethyl radical (CH\ :sub:`2`\ OH and
        CD\ :sub:`2`\ OH)”,
        Ph.D. dissertation, University of Southern California, 2012
        (`ProQuest <https://www.proquest.com/docview/1289069738>`__,
        `USC <https://digitallibrary.usc.edu/asset-management/2A3BF169XWB4>`__).

    ``three_point`` *
        The "Three Point" Abel transform method exploits the observation that
        the value of the Abel inverted data at any radial position r is
        primarily determined from changes in the projection data in the
        neighborhood of r. This method is also very efficient once it has
        generated the basis sets.

        C. J. Dasch,
        "One-dimensional tomography: a comparison of Abel, onion-peeling, and
        filtered backprojection methods",
        `Appl. Opt. 31, 1146–1152 (1992)
        <https://doi.org/10.1364/AO.31.001146>`__.

    ``two_point`` *
        Another Dasch method. Simple, and fast, but not as accurate as the
        other methods.

    The following class attributes are available, depending on the calculation.

    Returns
    -------
    transform : numpy 2D array
        the 2D forward/inverse Abel-transformed image.

    angular_integration : tuple
        (radial-grid, radial-intensity)
        radial coordinates and the radial intensity (speed) distribution,
        evaluated using :func:`abel.tools.vmi.angular_integration_3D()`.

    residual : numpy 2D array
        residual image (not currently implemented).

    IM : numpy 2D array
        the input image, re-centered (optional) with an odd-size width.

    method : str
        transform method, as specified by the input option.

    direction : str
        transform direction, as specified by the input option.

    radial : numpy 1D array
        with ``method='linbasex'``:
        radial grid for **Beta** array

    Beta : numpy 2D array
        with ``method='linbasex'``:
        coefficients of Newton-sphere spherical harmonics

            **Beta[0]** — the radial intensity variation

            **Beta[1]** — the anisotropy parameter variation

            ... **Beta[n]** — higher-order terms up to **legedre_orders** =
            [0, ..., n]

    projection : numpy 2D array
        with ``method='linbasex'``:
        radial projection profiles at angles **proj_angles**

    distr : Distributions.Results object
        with ``method='rbasex'``: the object from which various radial
        distributions can be retrieved
    """
    def __init__(self, IM,
                 direction='inverse', method='three_point', origin='none',
                 symmetry_axis=None, use_quadrants=(True, True, True, True),
                 symmetrize_method='average', angular_integration=False,
                 transform_options=dict(), center_options=dict(),
                 angular_integration_options=dict(),
                 recast_as_float64=True, verbose=False, center=_deprecated):
        """
        The one-stop transform function.
        """
        if center is not _deprecated:
            _deprecate('abel.transform.Transform() '
                       'argument "center" is deprecated, use "origin" instead.')
            origin = center

        # public class variables
        self.IM = IM  # (optionally) centered, odd-width image
        self.method = method
        self.direction = direction

        # private internal variables
        self._origin = origin
        self._symmetry_axis = symmetry_axis
        self._symmetrize_method = symmetrize_method
        self._use_quadrants = use_quadrants
        self._transform_options = transform_options
        self._recast_as_float64 = recast_as_float64

        self._verboseprint = print if verbose else lambda *a, **k: None

        # image processing
        self._verify_some_inputs()

        self._center_image(origin, **center_options)

        self._abel_transform_image(**transform_options)

        self._integration(angular_integration, transform_options,
                          **angular_integration_options)

    # end of class instance

    def _verify_some_inputs(self):
        if self.IM.ndim == 1 or np.shape(self.IM)[0] <= 2:
            raise ValueError('Data must be 2-dimensional. '
                             'To transform a single row, '
                             'use the individual transform function.')

        if not np.any(self._use_quadrants):
            raise ValueError('No image quadrants selected to use')

        if not isinstance(self._symmetry_axis, (list, tuple)):
            # if the user supplies an int, make it into a 1-element list:
            self._symmetry_axis = [self._symmetry_axis]
        elif len(self._symmetry_axis) == 0:
            # treat symmetry_axis=[] as symmetry_axis=None
            self._symmetry_axis = [None]

        if self.method == 'rbasex' and self._origin != 'none':
            if self._transform_options.get('origin') is not None:
                raise ValueError('Either use the "origin" argument to center '
                                 'the image, or pass "origin" to rbasex in '
                                 '"transform_options" to use the image as '
                                 'is, but don\'t do both.')
            if self._transform_options.get('weights') is not None:
                raise ValueError('Using the "origin" argument will center '
                                 'the image but not the "weights" array '
                                 'passed to rbasex. If you want to specify '
                                 'the image origin, pass it in '
                                 '"transform_options".')

        if self._recast_as_float64:
            self.IM = self.IM.astype('float64')

    def _center_image(self, method, **center_options):
        if method != "none":
            self.IM = tools.center.center_image(self.IM, method,
                                                **center_options)

    def _abel_transform_image(self, **transform_options):
        self._verboseprint('Calculating {0} Abel transform using {1} method -'
                          .format(self.direction, self.method),
                          '\n    image size: {:d}x{:d}'.format(*self.IM.shape))
        t0 = time.time()

        if self.method == "linbasex":
            self._abel_transform_image_full_linbasex(**transform_options)
        elif self.method == "rbasex":
            self._abel_transform_image_full_rbasex(**transform_options)
        else:
            self._abel_transform_image_by_quadrant(**transform_options)

        self._verboseprint("{:.2f} seconds".format(time.time() - t0))

    def _abel_transform_image_full_linbasex(self, **transform_options):
        self.transform, self.radial, self.Beta, self.projection = \
            linbasex.linbasex_transform_full(self.IM, **transform_options)

    def _abel_transform_image_full_rbasex(self, **transform_options):
        self.transform, self.distr = \
            rbasex.rbasex_transform(self.IM, direction=self.direction,
                                    **transform_options)

    def _abel_transform_image_by_quadrant(self, **transform_options):

        abel_transform = {
            "basex": basex.basex_transform,
            "daun": daun.daun_transform,
            "direct": direct.direct_transform,
            "hansenlaw": hansenlaw.hansenlaw_transform,
            "onion_bordas": onion_bordas.onion_bordas_transform,
            "onion_peeling": dasch.onion_peeling_transform,
            "two_point": dasch.two_point_transform,
            "three_point": dasch.three_point_transform,
        }

        self._verboseprint('Calculating {0} Abel transform using {1} method -'
                          .format(self.direction, self.method),
                          '\n    image size: {:d}x{:d}'.format(*self.IM.shape))

        t0 = time.time()

        # split image into quadrants
        Q0, Q1, Q2, Q3 = tools.symmetry.get_image_quadrants(
                         self.IM, reorient=True,
                         use_quadrants=self._use_quadrants,
                         symmetry_axis=self._symmetry_axis,
                         symmetrize_method=self._symmetrize_method)

        def selected_transform(Z):
            return abel_transform[self.method](Z, direction=self.direction,
                                               **transform_options)

        AQ0 = AQ1 = AQ2 = AQ3 = None

        # Inverse Abel transform for quadrant 1 (all include Q1)
        AQ1 = selected_transform(Q1)

        if 1 not in self._symmetry_axis:
            AQ2 = selected_transform(Q2)

        if 0 not in self._symmetry_axis:
            AQ0 = selected_transform(Q0)

        if None in self._symmetry_axis:
            AQ3 = selected_transform(Q3)

        # reassemble image
        self.transform = tools.symmetry.put_image_quadrants(
                                (AQ0, AQ1, AQ2, AQ3),
                                original_image_shape=self.IM.shape,
                                symmetry_axis=self._symmetry_axis)

        self._verboseprint("{:.2f} seconds".format(time.time()-t0))

    def _integration(self, angular_integration, transform_options,
                     **angular_integration_options):
        if angular_integration:
            if 'dr' in transform_options and\
               'dr' not in angular_integration_options:
                # assume user forgot to pass grid size
                angular_integration_options['dr'] = transform_options['dr']

            self.angular_integration = tools.vmi.angular_integration_3D(
                                             self.transform,
                                             **angular_integration_options)


# Default directory for cached basis sets;
# used by set_basis_dir() and get_basis_dir().
# DON'T access this variable directly!
_basis_dir = ''


def set_basis_dir(basis_dir='', make=True):
    """
    Changes the path to the directory for saving/loading cached basis sets that
    transform methods use by default.

    Parameters
    ----------
    basis_dir : str or None
        absolute or relative path. Passing ``''`` (default) resets to the
        system-dependent default path, see :func:`default_basis_dir`. For the
        current working directory (as in PyAbel up to v0.8.4), use ``'.'``. To
        disable basis-set caching on disk, use ``None``.
    make : bool
        create the directory if it does not exist (default: yes)

    Returns
    -------
    None
    """
    global _basis_dir

    if basis_dir == '':
        _basis_dir = default_basis_dir()
    else:
        _basis_dir = basis_dir

    if make:
        _make_basis_dir()  # (safe for None)


def get_basis_dir(make=False):
    """
    Gets the path to the directory for saving/loading cached basis sets that
    transform methods use by default. If not changed by :func:`set_basis_dir`,
    it depends on the operating system, see :func:`default_basis_dir`.

    Parameters
    ----------
    make : bool
        create the directory if it does not exist (default: no)

    Returns
    -------
    path : str or None
        absolute or relative path if disk caching is enabled,
        otherwise ``None``
    """
    global _basis_dir

    if _basis_dir == '':
        set_basis_dir('', make)

    return _basis_dir


def _make_basis_dir():
    """
    Internal utility function.

    Makes sure that the default basis-set directory (if enabled) exists.
    """
    if _basis_dir is None or os.path.exists(_basis_dir):
        return

    try:
        os.makedirs(_basis_dir)
    except Exception as e:
        print('Cannot create the directory\n"{}"\n'
              'for saving/loading basis sets:'.format(_basis_dir))
        raise


def default_basis_dir():
    r"""
    Gets full path to the system-dependent default directory for saving/loading
    cached basis sets:

    Linux (and other Unix-like):
        ``~/.cache/PyAbel`` (or ``$XDG_CACHE_HOME/PyAbel`` if set)
    macOS:
        ``/Users/<user>/Library/Caches/PyAbel``
    Windows:
        ``<user profile>\AppData\Local\PyAbel\cache`` (or
        ``%LOCALAPPDATA%\PyAbel\cache`` if set). See important notes below.

    Parameters
    ----------
    None

    Returns
    -------
    path : str
        full path to the system-dependent default basis-sets directory


    .. admonition:: Notes for MS Windows users

      * Python installed from Microsoft Store redirects subdirectory creation
        in ``AppData\Local`` to a `"private per-user, per-app location"
        <https://docs.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-behind-the-scenes>`__
        ``AppData\Local\Packages\Python...\LocalCache\Local`` (see `Using
        Python on Windows / Known Issues
        <https://docs.python.org/3/using/windows.html#known-issues>`__).
        However, if ``AppData\Local\PyAbel\`` already exists (for example, was
        manually created *not* from Python), apparently it should be usable.

      * Old Windows versions (2000, XP, Server 2003) by default don't set the
        ``LOCALAPPDATA`` environment variable, so PyAbel will create and use
        the ``AppData\Local`` subtree in the user profile folder. This is
        probably fine, but not how it should be. To use the standard location,
        please do ::

            set LOCALAPPDATA=%USERPROFILE%\Local Settings\Application Data

        before starting Python. Or permanently set it in “Environment
        Variables” from Windows “System Properties”.
    """
    # Based on the appdirs module, https://github.com/ActiveState/appdirs
    # (see https://github.com/ActiveState/appdirs/blob/master/appdirs.py)
    system = platform.system()

    if system == 'Darwin':  # macOS
        return os.path.expanduser('~/Library/Caches/PyAbel')

    if system == 'Windows':
        return os.path.join(os.getenv('LOCALAPPDATA',
                                      os.path.expanduser(r'~\AppData\Local')),
                            r'PyAbel\cache')

    # Linux and other
    return os.path.join(os.getenv('XDG_CACHE_HOME',
                                  os.path.expanduser('~/.cache')),
                        'PyAbel')

    # system == 'Java' is ignored as useless -- Jython does not support NumPy


def basis_dir_cleanup(basis_dir='', method=None):
    """
    Deletes saved basis sets.

    Parameters
    ----------
    basis_dir : str or None
        path to the directory with saved basis sets. Use ``''`` for the default
        directory, see :func:`get_basis_dir`. (For convenience, ``None`` can be
        passed to do nothing.)
    method : str or list of str or None
        transform methods for which basis sets should be deleted. Can be a
        single string (see the :attr:`method` parameter in :class:`Transform`)
        or a list of strings. Use ``'all'`` to delete basis sets for all
        methods. ``None`` does nothing.

    Returns
    -------
    None
    """
    if basis_dir is None or method is None:
        return

    # make the list of methods
    if method == 'all':
        methods = ['basex', 'daun', 'linbasex', 'onion_peeling', 'rbasex',
                   'three_point', 'two_point']
    elif np.ndim(method) == 0:  # single string
        methods = [method]
    else:  # already a list
        methods = method

    for method in methods:
        if method in ['onion_peeling', 'three_point', 'two_point']:
            dasch.basis_dir_cleanup(method, basis_dir)
        else:
            module = sys.modules.get('abel.' + method)
            if not module:
                warn('Unknown method "{}"!'.format(method),
                     SyntaxWarning, stacklevel=2)
                continue
            func = getattr(module, 'basis_dir_cleanup', None)
            if func:
                func(basis_dir)
            else:
                warn('Method "{}" does not save basis sets.'.format(method),
                     SyntaxWarning, stacklevel=2)
