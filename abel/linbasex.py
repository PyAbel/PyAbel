# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from glob import glob
import numpy as np
import scipy
from scipy.special import eval_legendre
from scipy.ndimage import rotate, shift, gaussian_filter1d

import abel
from abel import _deprecated, _deprecate

###############################################################################
# linbasex - inversion procedure based on 1-dimensional projections of
#            velocity-map images
#
# As described in:
#   Gerber, Thomas, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi,
#   Peter Radi, and Yaroslav Sych,
#     “Charged Particle Velocity Map Image Reconstruction with One-Dimensional
#      Projections of Spherical Functions.”
#     Review of Scientific Instruments 84, no. 3 (March 1, 2013):
#                                      033101–033101 – 10.
#     doi:10.1063/1.4793404.
#
# 2016-04- Thomas Gerber and Daniel Hickstein - theory and code updates
# 2016-04- Stephen Gibson core code extracted from the supplied jupyter
#          notebook (see #167: https://github.com/PyAbel/PyAbel/issues/167)
#
###############################################################################

# cache basis
_basis = None
_los = None   # legendre_orders string
_pas = None   # proj_angles string
_radial_step = None
_clip = None


def linbasex_transform(IM, basis_dir=None, proj_angles=[0, np.pi/2],
                       legendre_orders=[0, 2], radial_step=1, smoothing=0,
                       rcond=0.0005, threshold=0.2, return_Beta=False, clip=0,
                       norm_range=(0, -1), direction="inverse", verbose=False,
                       dr=None):
    """
    Wrapper function for linbasex to process a single image quadrant in the
    upper right orientation (Q0).
    *Is not applicable to images with odd Legendre orders.*

    Parameters not described below are passed directly to
    :func:`linbasex_transform_full`.

    Parameters
    ----------
    IM : numpy 2D array
        upper right quadrant of the image data, must be square in shape
    return_Beta : bool
        in addition to the transformed image, return the **radial**, **Beta**
        and **projections** arrays
    dr : any
        dummy variable for call compatibility with the other methods

    Returns
    -------
    inv_IM : numpy 2D array
        upper right quadrant of the inverse Abel transformed image
    radial : numpy 1D array
        (only if **return_Beta** = ``True``)
        radii of each Newton sphere
    Beta : numpy 2D array
        (only if **return_Beta** = ``True``)
        contributions of each spherical harmonic :math:`Y_{i0}` to the 3D
        distribution contain all the information one can get from an experiment.
        For the case **legendre_orders** = [0, 2]:

           **Beta[0]** vs **radial** is the speed distribution

           **Beta[1]** vs **radial** is the anisotropy of each Newton sphere

    projections : numpy 2D array
        (only if **return_Beta** = ``True``)
        projection profiles at angles **proj_angles**
    """
    IM = np.atleast_2d(IM)

    # duplicate the quadrant, re-forming the whole image.
    quad_rows, quad_cols = IM.shape
    full_image = abel.tools.symmetry.put_image_quadrants((IM, IM, IM, IM),
                 original_image_shape=(quad_rows*2-1, quad_cols*2-1))

    # inverse Abel transform
    recon, radial, Beta, QLz = linbasex_transform_full(full_image,
                               basis_dir=basis_dir, proj_angles=proj_angles,
                               legendre_orders=legendre_orders,
                               radial_step=radial_step, smoothing=smoothing,
                               threshold=threshold, clip=clip,
                               norm_range=norm_range,
                               verbose=verbose)

    # unpack upper right quadrant
    inv_IM = abel.tools.symmetry.get_image_quadrants(recon)[0]

    if return_Beta:
        return inv_IM, radial, Beta, QLz
    else:
        return inv_IM


def linbasex_transform_full(IM, basis_dir=None, proj_angles=[0, np.pi/2],
                            legendre_orders=[0, 2],
                            radial_step=1, smoothing=0,
                            rcond=0.0005, threshold=0.2, clip=0,
                            return_Beta=_deprecated, norm_range=(0, -1),
                            direction="inverse", verbose=False):
    r"""Inverse Abel transform using 1D projections of images.

    Th. Gerber, Yu. Liu, G. Knopp, P. Hemberger, A. Bodi, P. Radi, Ya. Sych,
    "Charged particle velocity map image reconstruction with one-dimensional
    projections of spherical functions",
    `Rev. Sci. Instrum. 84, 033101 (2013)
    <https://doi.org/10.1063/1.4793404>`__.

    :doc:`Lin-Basex <transform_methods/linbasex>` models the image using a sum
    of Legendre polynomials at each radial pixel. As such, it should only be
    applied to situations that can be adequately represented by Legendre
    polynomials, i.e., images that feature spherical-like structures.  The
    reconstructed 3D object is obtained by adding all the contributions, from
    which slices are derived.

    This function operates on the whole image.

    Parameters
    ----------
    IM : numpy 2D array
        image data must have square shape of odd size
    basis_dir : str or None
        path to the directory for saving / loading the basis sets. Use ``''``
        for the default directory. If ``None`` (default), the basis set will
        not be loaded from or saved to disk.
    proj_angles : list of float
        projection angles, in radians (default :math:`[0, \pi/2]`)
        e.g. :math:`[0, \pi/2]` or :math:`[0, 0.955, \pi/2]` or
        :math:`[0, \pi/4, \pi/2, 3\pi/4]`
    legendre_orders : list of int
        orders of Legendre polynomials to be used as the expansion

        * even polynomials [0, 2, ...] gerade
        * odd polynomials [1, 3, ...] ungerade
        * all orders [0, 1, 2, ...].

        In a single-photon experiment there are only anisotropies up to
        second order. The interaction of 4 photons (four-wave mixing) yields
        anisotropies up to order 8.
    radial_step : int
        number of pixels per Newton sphere (default 1)
    smoothing: float
        convolve **Beta** array with a Gaussian function of :math:`1/e`
        halfwidth equal to **smoothing**.
    rcond : float
        (default 0.0005) :func:`scipy.linalg.lstsq` fit conditioning value.
        Use 0 to switch conditioning off.
        Note: In the presence of noise the equation system may be ill-posed.
        Increasing **rcond** smoothes the result, lowering it beyond a minimum
        renders the solution unstable. Tweak **rcond** to get a "reasonable"
        solution with acceptable resolution.
    threshold : float
        threshold for normalization of higher-order Newton spheres (default
        0.2): if **Beta[0]** < **threshold**, the associated **Beta[j]** for
        all j ⩾ 1 are set to zero
    clip : int
        clip first vectors (smallest Newton spheres) to avoid singularities
        (default 0)
    norm_range : tuple of int
        (low, high)
        normalization of Newton spheres, maximum in range **Beta[0, low:high]**.
        Note: **Beta[0, i]**, the total number of counts integrated over sphere
        i, becomes 1.
    direction : str
        Abel transform direction. Only "inverse" is implemented.
    verbose : bool
        print information about processing (normally used for debugging)

    Returns
    -------
    inv_IM : numpy 2D array
        inverse Abel transformed image
    radial : numpy 1D array
        radii of each Newton sphere
    Beta : numpy 2D array
        contributions of each spherical harmonic :math:`Y_{i0}` to the 3D
        distribution contain all the information one can get from an experiment.
        For the case **legendre_orders** = [0, 2]:

           **Beta[0]** vs **radial** is the speed distribution

           **Beta[1]** vs **radial** is the anisotropy of each Newton sphere

    projections : numpy 2D array
        projection profiles at angles **proj_angles**
    """

    IM = np.atleast_2d(IM)

    rows, cols = IM.shape

    if cols % 2 == 0:
        raise ValueError('image width ({}) must be odd and equal to the height'
                         .format(cols))

    if rows != cols:
        raise ValueError('image has shape ({}, {}), '.format(rows, cols) +
                         'must be square for a "linbasex" transform')

    if return_Beta is not _deprecated:
        _deprecate('abel.linbasex.linbasex_transform_full() '
                   'argument "return_Beta" is deprecated, these arrays are '
                   'returned always.')

    # generate basis or read from file if available
    Basis = get_bs_cached(cols, basis_dir=basis_dir, proj_angles=proj_angles,
                  legendre_orders=legendre_orders, radial_step=radial_step,
                  clip=clip, verbose=verbose)

    # Number of used polynoms
    pol = len(legendre_orders)

    # How many projections
    proj = len(proj_angles)

    QLz = np.zeros((proj, cols))  # array for projections.

    # Rotate and project VMI-image for each angle (as many as projections)
    # if proj_angles == [0, np.pi/2]:
    #     # If coordinates of the detector coincide with the projection
    #     # directions unnecessary rotations are avoided
    #     # i.e. proj_angles=[0, np.pi/2] degrees
    #     QLz[0] = np.sum(IM, axis=1)
    #     QLz[1] = np.sum(IM, axis=0)
    # else:
    for i in range(proj):
        Rot_IM = rotate(IM, proj_angles[i] * 180 / np.pi, reshape=False)
        QLz[i, :] = np.sum(Rot_IM, axis=1)

    # arrange all projections for input into "lstsq"
    bb = np.concatenate(QLz, axis=0)

    Beta = _beta_solve(Basis, bb, pol, rcond=rcond)

    # compensate 1/2-pixel shift (basis issue? see PR #357)
    Beta = shift(Beta, (0, 0.5 / radial_step), mode='nearest')

    # reverse the sign for odd orders (the basis is historically upside down)
    for i in range(pol):
        if legendre_orders[i] % 2:
            Beta[i] = -Beta[i]

    R = cols // 2  # outer radius: cols = 2R + 1
    radial = np.linspace(clip * radial_step + R % radial_step, R, len(Beta[0]))

    inv_IM, Beta_convol = _Slices(radial, Beta, legendre_orders,
                                  smoothing=smoothing)

    # normalize
    Beta = _single_Beta_norm(Beta_convol, threshold=threshold,
                             norm_range=norm_range)
    inv_IM /= radial_step

    # Fix Me! Issue #202 the correct scaling factor for inv_IM intensity?
    return inv_IM, radial, Beta, QLz


def _beta_solve(Basis, bb, pol, rcond=0.0005):
    # set rcond to zero to switch conditioning off

    # solve equation
    Sol = np.linalg.lstsq(Basis, bb, rcond)

    # arrange solutions into subarrays for each β
    Beta = Sol[0].reshape((pol, len(Sol[0]) // pol))

    return Beta


def _Slices(radial, Beta, legendre_orders, smoothing=0):
    """Convolve Beta with a Gaussian function of 1/e width smoothing.

    """
    R = int(radial[-1])  # outer radius
    pol = len(legendre_orders)
    NP = len(Beta[0])  # number of Newton spheres

    # Convolve Beta with Gaussian smoothing function
    if smoothing > 0:
        Beta_convol = gaussian_filter1d(Beta, smoothing, axis=1,
                                        mode='constant', cval=0)
    else:
        Beta_convol = Beta

    Slice = np.zeros((2 * R + 1, 2 * R + 1))  # full image size
    col = np.arange(-R, R + 1)
    row = col[:, None]
    r = np.sqrt(row**2 + col**2 + 0.1)  # + 0.1 to avoid division by zero

    # Sum ordered slices up:
    for i in range(pol):
        # interpolated β(r), where r=radius
        BB = np.interp(r, radial, Beta_convol[i, :], left=0)
        # multiplied by angular part, -row / r = cos θ
        Slice += BB * eval_legendre(legendre_orders[i], -row / r)

    # normalize: division by sphere area
    Slice /= 4 * np.pi * r**2

    return Slice, Beta_convol


def int_beta(Beta, radial_step=1, threshold=0.1, regions=None):
    """Integrate beta over a range of Newton spheres.

    .. warning::
        This function is deprecated and will be remove in the future. See
        `issue #356 <https://github.com/PyAbel/PyAbel/issues/356>`__.

        For integrating the speed distribution and averaging the anisotropy,
        please use :func:`mean_beta`.

    Parameters
    ----------
    Beta : numpy array
        Newton spheres
    radial_step : int
        number of pixels per Newton sphere (default 1)
    threshold : float
        threshold for normalisation of higher orders, 0.0 ... 1.0.
    regions : list of tuple radial ranges
        [(min0, max0), (min1, max1), ...]

    Returns
    -------
    Beta_in : numpy array
        integrated normalized Beta array [Newton sphere, region]

    """
    _deprecate('int_beta() is deprecated, consider using mean_beta().')

    pol = Beta.shape[0]
    # Define new array for normalized beta's, independent of Beat_norm
    Beta_n = np.zeros(Beta.shape)

    # Normalized to Newton sphere with maximal counts.
    max_counts = max(Beta[0, :])

    Beta_n[0] = Beta[0]/max_counts
    for i in range(1, pol):
        Beta_n[i] = np.where(Beta[0]/max_counts > threshold, Beta[i]/Beta[0],
                             0)

    Beta_int = np.zeros((pol, len(regions)))   # arrays for results

    for j, reg in enumerate(regions):
        for i in range(pol):
            Beta_int[i, j] = np.sum(Beta_n[i, range(*reg)])/(reg[1]-reg[0])

    return Beta_int


def mean_beta(radial, Beta, regions):
    """
    Integrate normalized intensity (``Beta[0]``) and perform intensity-weighted
    averaging of anisotropy (``Beta[1:]``) over ranges of Newton spheres.

    Parameters
    ----------
    radial : numpy 1D array
        radii of Newton spheres
    Beta : numpy 2D array
        speed and anisotropy distribution from :func:`linbasex_transform_full`
    regions : list of tuple of int
        radial ranges [(min0, max0), (min1, max1), ...].
        Note that inclusion of regions where **Beta[0]** is below **threshold**
        set in :func:`linbasex_transform_full` will bias the mean anisotropies
        towards zero.

    Returns
    -------
    Beta_mean : 2D numpy array
        overall intensity (``Beta_mean[0]``) and mean anisotropy values
        (``Beta_mean[1:]``) in each region
    """
    pol = Beta.shape[0]

    Beta_mean = np.empty((pol, len(regions)))

    for i, (rmin, rmax) in enumerate(regions):
        # radial indices within region
        idx = np.where((rmin <= radial) & (radial <= rmax))[0]  # until PEP 535
        # intensity: average
        Imean = Beta[0, idx].mean()
        # integrated
        Beta_mean[0, i] = Imean * (rmax - rmin + 1)
        # anisotropies: intensity-weighted average
        Beta_mean[1:, i] = (Beta[1:, idx] * Beta[0, idx]).mean(axis=1) / Imean

    return Beta_mean


def _single_Beta_norm(Beta, threshold=0.2, norm_range=(0, -1)):
    """Normalize Newton spheres.

    Parameters
    ----------
    Beta : numpy array
        Newton spheres
    threshold : float
        choose only Beta's for which Beta0 is greater than the maximal Beta0
        times threshold in the chosen range
        Set all βi, i>=1 to zero if the associated β0 is smaller than threshold

    norm_range : tuple (int, int)
        (low, high)
        normalize to Newton sphere with maximum counts in chosen range.
        Beta[0, low:high]

    Return
    ------
    Beta : numpy array
        normalized Beta array

    """
    Beta_norm = np.zeros_like(Beta)
    # Normalized to Newton sphere with maximum counts in chosen range.
    max_counts = Beta[0, norm_range[0]:norm_range[1]].max()
    if max_counts > 0:  # (don't fail for zero images)
        Beta_norm[0] = Beta[0] / max_counts
    np.divide(Beta[1:], Beta[0], out=Beta_norm[1:],
              where=Beta_norm[0] > threshold)

    return Beta_norm


def _bas(order, angle, COS, TRI):
    """Define basis vectors for a given polynomial order "order" and a
       given projection angle "angle".
    """
    return eval_legendre(order, angle) * eval_legendre(order, COS) * TRI


def _bs_linbasex(cols, proj_angles=[0, np.pi/2], legendre_orders=[0, 2],
                 radial_step=1, clip=0):

    n = cols // 2 + 1  # 0 to outer R
    proj = len(proj_angles)
    pol = len(legendre_orders)

    # Calculation of base vectors,
    # using only each radial_step other vector but keeping the outer radius

    # Matrix representing cos θ (rows z / columns r_k)
    Index = np.indices((n, n))
    Index[:, 0, 0] = 1
    cos = np.triu(Index[0] / np.diag(Index[0]))
    cos = cos[:, ::-radial_step][:, ::-1]  # decimate r_k

    # Matrix representing step functions Pi (decimated upper triangular)
    tri = np.tri(n)[::-1, ::radial_step][:, ::-1]

    # Concatenate to double-sided matrices (-R to R)
    COS = np.concatenate((-cos[::-1], cos[1:]), axis=0)
    TRI = np.concatenate((tri[::-1], tri[1:]), axis=0)

    if clip > 0:
        # clip first vectors (smallest Newton spheres) to avoid singularities
        COS = COS[:, clip:]
        # It is difficult to trace the effect on the SVD solver used below.
        TRI = TRI[:, clip:]  # usually no clipping works fine.

    # Calculate base vectors for each projection and each order.
    B = np.zeros((pol, proj) + COS.shape)

    Norm = np.sum(_bas(0, 1, COS, TRI), axis=0)  # normalization
    cos_an = np.cos(proj_angles)  # cosines of projection angles

    for p in range(pol):
        for u in range(proj):
            B[p, u] = _bas(legendre_orders[p], cos_an[u], COS, TRI) / Norm

    # concatenate vectors to one matrix of bases
    Bpol = np.concatenate(B, axis=2)
    Basis = np.concatenate(Bpol, axis=0)

    return Basis


def get_bs_cached(cols, basis_dir=None, legendre_orders=[0, 2],
                  proj_angles=[0, np.pi/2],
                  radial_step=1, clip=0, verbose=False):
    """load basis set from disk, generate and store if not available.

    Checks whether file:
    ``linbasex_basis_{cols}_{legendre_orders}_{proj_angles}_{radial_step}_{clip}*.npy`` is present in `basis_dir`

    Either, read basis array or generate basis, saving it to the file.


    Parameters
    ----------
    cols : int
        width of image

    basis_dir : str or None
        path to the directory for saving / loading the basis. Use ``''`` for
        the default directory. If ``None``, the basis set will not be loaded
        from or saved to disk.

    legendre_orders : list
        default [0, 2] = 0 order and 2nd order polynomials

    proj_angles : list
        default [0, np.pi/2] in radians

    radial_step : int
        pixel grid size, default 1

    clip : int
        image edge clipping, default 0 pixels

    verbose: boolean
        print information for debugging

    Returns
    -------
    D : tuple (B, Bpol)
       of ndarrays B (pol, proj, cols, cols) Bpol (pol, proj)

    file.npy: file
       saves basis to file name ``linbasex_basis_{cols}_{legendre_orders}_{proj_angles}_{radial_step}_{clip}.npy``

    """

    # cached basis
    global _basis, _los, _pas, _radial_step, _clip

    # legendre_orders string
    los = ''.join(map(str, legendre_orders))
    # convert to % of pi
    proj_angles_fractpi = np.array(proj_angles)*100/np.pi
    # projection angles string
    pas = ''.join(map(str, proj_angles_fractpi.astype(int)))

    if _basis is not None:
        # check basis array sizes, warning may not be unique
        if _basis.shape == (2*cols, cols+1):
            if _los == los and _pas == pas and _radial_step == radial_step and\
               _clip == clip:
                if verbose:
                    print('Using memory cached basis')
                return _basis

    # Fix Me! not a simple unique naming mechanism
    basis_name = "linbasex_basis_{}_{}_{}_{}_{}.npy".format(cols, los, pas,
                                                            radial_step, clip)

    _los = los
    _pas = pas
    _radial_step = radial_step
    _clip = clip
    if basis_dir == '':
        basis_dir = abel.transform.get_basis_dir(make=True)
    if basis_dir is not None:
        path_to_basis_file = os.path.join(basis_dir, basis_name)
        if os.path.exists(path_to_basis_file):
            if verbose:
                print('loading {} ...'.format(path_to_basis_file))
            _basis = np.load(path_to_basis_file)
            return _basis

    if verbose:
        print("A suitable basis for linbasex was not found.\n"
              "A new basis will be generated.")

    _basis = _bs_linbasex(cols, proj_angles=proj_angles,
                     legendre_orders=legendre_orders, radial_step=radial_step,
                     clip=clip)

    if basis_dir is not None:
        path_to_basis_file = os.path.join(basis_dir, basis_name)
        np.save(path_to_basis_file, _basis)
        if verbose:
            print("linbasex basis saved for later use to {}"
                  .format(path_to_basis_file))

    return _basis


def cache_cleanup():
    """
    Utility function.

    Frees the memory caches created by :func:`get_bs_cached`.
    This is usually pointless, but might be required after working
    with very large images, if more RAM is needed for further tasks.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    global _basis, _los, _pas, _radial_step, _clip

    _basis = None
    _los = None
    _pas = None
    _radial_step = None
    _clip = None


def basis_dir_cleanup(basis_dir=''):
    """
    Utility function.

    Deletes basis sets saved on disk.

    Parameters
    ----------
    basis_dir : str or None
        relative or absolute path to the directory with saved basis sets. Use
        ``''`` for the default directory. ``None`` does nothing.

    Returns
    -------
    None
    """
    if basis_dir == '':
        basis_dir = abel.transform.get_basis_dir(make=False)

    if basis_dir is None:
        return

    files = glob(os.path.join(basis_dir, 'linbasex_basis_*.npy'))
    for fname in files:
        os.remove(fname)
