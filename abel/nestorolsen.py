import os.path
import numpy as np
from glob import glob
import abel
from scipy.linalg import inv

###############################################################################
#
#  Nestor, Olsen line probe deconvolution
#  as described in SIAM Review, vol. 2, no. 3, 1960, pp. 200–07.
#  https://epubs.siam.org/doi/10.1137/1002042
#
###############################################################################

# cache deconvolution operator array
_D = None
_source = None   # 'cache', 'generated', or 'file', for unit testing


def nestorolsen_transform(IM, basis_dir='', dr=1, direction="inverse", verbose=False):
    """
    The :doc:`nestorolsen-method deconvolution method
    <transform_methods/nestorolsen_method>`.

    O. H. Nestor and H. N. Olsen,
    "Numerical Methods for Reducing Line and Surface Probe Data",
    `SIAM Review, vol. 2, no. 3, 1960, pp. 200–07
    <https://doi.org/10.1137/1002042>`__.

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    basis_dir: str or None
        path to the directory for saving / loading the "nestorolsen_method"
        deconvolution operator array. Here, called ``basis_dir`` for
        consistency with the other true basis methods. Use ``''`` for the
        default directory. If ``None``, the operator array will not be loaded
        from or saved to disk.

    dr : float
        sampling size (=1 for pixel images), used for Jacobian scaling.
        The resulting inverse transform is simply scaled by 1/dr.

    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed

    verbose : bool
        trace printing


    Returns
    -------
    inv_IM: 1D or 2D numpy array
        the "nestorolsen_method" inverse Abel transformed half-image

    """
    global _D

    # make sure that the data has 2D shape
    IM = np.atleast_2d(IM)

    rows, cols = IM.shape

    _D = get_bs_cached(cols, basis_dir=basis_dir, verbose=verbose)
    D = _D / dr

    if direction == 'forward':
        D = inv(D)

    inv_IM = np.tensordot(IM, D, axes=(1, 1))

    if rows == 1:
        inv_IM = inv_IM[0]  # flatten array

    return inv_IM


def _bs_nestorolsen(cols):
    """deconvolution function for nestorolsen.

    Parameters
    ----------
    cols : int
        width of the image
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
    """Load nestorolsen method deconvolution operator array from cache, or disk.
    Generate and store if not available.

    Checks whether nestorolsen deconvolution array has been previously
    calculated, or whether the file ``nestorolsen_basis_{cols}.npy`` is
    present in `basis_dir`.

    Either, assign, read, or generate the deconvolution array
    (saving it to file).


    Parameters
    ----------
    cols : int
        width of image

    basis_dir : str or None
        path to the directory for saving or loading the deconvolution array.
        Use ``''`` for the default directory.
        For ``None``, do not load or save the deconvolution operator array

    verbose: boolean
        print information (mainly for debugging purposes)

    Returns
    -------
    D: numpy 2D array of shape (cols, cols)
       deconvolution operator array

    file.npy: file
       saves `D`, the deconvolution array to file name:
       ``nestorolsen_basis_{cols}.npy``

    """

    global _D, _source

    # check whether the deconvolution operator array is cached
    if _D is not None:
        if _D.shape[0] >= cols:
            if verbose:
                print('Using memory cached deconvolution operator array,'
                      f' shape {_D.shape}')
            _source = 'cache'
            return _D[:cols, :cols]  # sliced to correct size

    D_name = f'nestorolsen_basis_{cols}.npy'

    if basis_dir == '':
        basis_dir = abel.transform.get_basis_dir(make=True)

    # read deconvolution operator array if available
    if basis_dir is not None:
        path_to_basis_files = os.path.join(basis_dir, 'nestorolsen_basis*')
        basis_files = glob(path_to_basis_files)
        for bf in basis_files:
            if int(bf.split('_')[-1].split('.')[0]) >= cols:
                # relies on file order
                if verbose:
                    print('Loading deconvolution operator array from file', bf)
                # slice to size
                _D = np.load(bf)[:cols, :cols]
                _source = 'file'
                return _D

    if verbose:
        print(f'A suitable deconvolution array for "nestorolsen" was not found.\n'
              'A new array will be generated.')

    _D = _bs_nestorolsen(cols)
    _source = 'generated'

    if basis_dir is not None:
        path_to_basis_file = os.path.join(basis_dir, D_name)
        np.save(path_to_basis_file, _D)
        if verbose:
            print('\ndeconvolution operator array saved to'
                  f' "{path_to_basis_file}"')

    return _D


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

    global _D, _source

    _D = None
    _source = None


def basis_dir_cleanup(basis_dir=''):
    """
    Utility function.

    Deletes deconvolution operator arrays saved on disk.

    Parameters
    ----------
    basis_dir : str or None
        absolute or relative path to the directory with saved deconvolution
        operator arrays. Use ``''`` for the default directory. ``None`` does
        nothing.

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
