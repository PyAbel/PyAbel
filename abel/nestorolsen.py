import os.path
from glob import glob

import numpy as np
from scipy.linalg import inv

import abel

###############################################################################
#
#  Nestor–Olsen "inversion of line probe data"
#  as described in SIAM Review, vol. 2, no. 3, 1960, pp. 200–207.
#  https://epubs.siam.org/doi/10.1137/1002042
#
###############################################################################

# Cached inverse-transform coefficients
_D = None


def nestorolsen_transform(IM, basis_dir='', dr=1, direction='inverse',
                          verbose=False):
    """
    The :doc:`Nestor–Olsen method <transform_methods/nestorolsen>` for the
    inverse Abel transform. The forward transform is also supported but was not
    described in the original publication.

    O. H. Nestor, H. N. Olsen,
    "Numerical methods for reducing line and surface probe data",
    `SIAM Rev. 2(3), 200–207 (1960) <https://doi.org/10.1137/1002042>`__.

    This function operates on the “right side” of an image, that it, just one
    half of a cylindrically symmetric image, with the axial pixels located in
    the 0-th column.

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    basis_dir: str or None
        path to the directory for saving / loading the transform coefficients
        (called ``basis_dir`` for consistency with the other methods).
        Use ``''`` for the default directory. If ``None``, the coefficients
        will not be loaded from or saved to disk.

    dr : float
        sampling step (=1 for pixel images), used for Jacobian scaling.
        The resulting transform is simply scaled by 1/dr.

    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed

    verbose : bool
        trace printing

    Returns
    -------
    recon: 1D or 2D numpy array
        the Abel-transformed half-image
    """
    # make sure that the data has 2D shape
    IM = np.atleast_2d(IM)

    rows, cols = IM.shape

    D = get_bs_cached(cols, basis_dir=basis_dir, verbose=verbose) / dr

    if direction == 'forward':
        D = inv(D)

    recon = np.tensordot(IM, D, axes=(1, 1))

    if rows == 1:
        recon = recon[0]  # 1D array

    return recon


def _bs_nestorolsen(cols):
    """
    Calculation of the inverse-transform coefficients.

    Parameters
    ----------
    cols : int
        width of the half-image
    """
    A = np.zeros((cols, cols))

    k, n = np.diag_indices(cols)
    A[k, n] = (np.sqrt(n**2-k**2)-np.sqrt((n+1)**2-k**2))/(2*n+1)

    ku, nu = np.triu_indices(cols, k=1)
    A[ku, nu] = (np.sqrt(nu**2-ku**2)-np.sqrt((nu+1)**2-ku**2))/(2*nu+1) -\
        (np.sqrt((nu-1)**2-ku**2)-np.sqrt(nu**2-ku**2))/(2*nu-1)

    A *= -2/np.pi

    return A


def get_bs_cached(cols, basis_dir='', verbose=False):
    """
    Load the inverse-transform coefficients from memory cache or disk.
    Generate and store if not available.

    Checks whether the coefficients have been previously calculated, or whether
    the file ``nestorolsen_basis_{cols}.npy`` is present in `basis_dir`.

    Either assign, read, or generate the coefficients (saving them to file).

    Parameters
    ----------
    cols : int
        width of half-image

    basis_dir : str or None
        path to the directory for saving or loading the coefficients.
        Use ``''`` for the default directory.
        For ``None``, do not load or save the coefficients.

    verbose: bool
        print information (mainly for debugging purposes)

    Returns
    -------
    D: numpy 2D array of shape (cols, cols)
        inverse-transform coefficients
    """
    global _D

    # check whether the coefficients are cached
    if _D is not None:
        if _D.shape[0] >= cols:
            if verbose:
                print(f'Using memory-cached coefficients, shape {_D.shape}.')
            return _D[:cols, :cols]  # sliced to correct size

    if basis_dir == '':
        basis_dir = abel.transform.get_basis_dir(make=True)

    # read the coefficients from a file if available
    if basis_dir is not None:
        files = glob(os.path.join(basis_dir, 'nestorolsen_basis_*.npy'))
        for bf in files:
            if int(bf.split('_')[-1].split('.')[0]) >= cols:
                # relies on file order
                if verbose:
                    print('Loading coefficients from file', bf)
                # slice to size
                _D = np.load(bf)[:cols, :cols]
                return _D

    if verbose:
        print('Suitable stored coefficients for "nestorolsen" were not found.'
              '\nA new array will be generated.')

    _D = _bs_nestorolsen(cols)

    if basis_dir is not None:
        file_path = os.path.join(basis_dir, f'nestorolsen_basis_{cols}.npy')
        np.save(file_path, _D)
        if verbose:
            print(f'\nCoefficients saved to "{file_path}".')

    return _D


def cache_cleanup():
    """
    Utility function.

    Frees the memory cache created by :func:`get_bs_cached`.
    This is usually pointless, but might be required after working
    with very large images, if more RAM is needed for further tasks.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    global _D

    _D = None


def basis_dir_cleanup(basis_dir=''):
    """
    Utility function.

    Deletes all coefficients saved on disk.

    Parameters
    ----------
    basis_dir : str or None
        absolute or relative path to the directory with saved coefficients.
        Use ``''`` for the default directory. ``None`` does nothing.

    Returns
    -------
    None
    """
    if basis_dir == '':
        basis_dir = abel.transform.get_basis_dir(make=False)

    if basis_dir is None:
        return

    files = glob(os.path.join(basis_dir, 'nestorolsen_basis_*.npy'))
    for fname in files:
        os.remove(fname)
