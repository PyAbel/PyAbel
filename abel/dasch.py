# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import numpy as np
import glob
import abel
from scipy.linalg import inv

###############################################################################
#
#  Dasch two-point, three_point, and onion-peeling deconvolution
#    as described in Applied Optics 31, 1146 (1992), page 1147-8 sect. B & C.
#        https://www.osapublishing.org/ao/abstract.cfm?uri=ao-31-8-1146
#    see also discussion in PR #155  https://github.com/PyAbel/PyAbel/pull/155
#
# 2016-03-25 Dan Hickstein - one line Abel transform
# 2016-03-24 Steve Gibson - Python code framework
# 2015-12-29 Dhrubajyoti Das - original three_point code and
#                              highlighting the Dasch paper,see issue #61
#                              https://github.com/PyAbel/PyAbel/issues/61
#
###############################################################################

_dasch_parameter_docstring = \
    """dasch_method deconvolution
        C. J. Dasch Applied Optics 31, 1146 (1992).
        http://dx.doi.org/10.1364/AO.31.001146

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    basis_dir: str
        path to the directory for saving / loading
        the "dasch_method" deconvolution operator array. Here, called
        `basis_dir` for consistency with the other true basis methods.
        If `None`, the operator array will not be saved to disk.

    dr : float
        sampling size (=1 for pixel images), used for Jacobian scaling.
        The resulting inverse transform is simply scaled by 1/dr.

    direction : str
        only the `direction="inverse"` transform is currently implemented

    verbose : bool
        trace printing


    Returns
    -------
    inv_IM: 1D or 2D numpy array
        the "dasch_method" inverse Abel transformed half-image

    """

# cache deconvolution operator array
_D = None
_method = None
_source = None   # 'cache', 'generated', or 'file', for unit testing


def two_point_transform(IM, basis_dir='.', dr=1, direction="inverse",
                        verbose=False):
    return _dasch_transform(IM, basis_dir=basis_dir, dr=dr,
                            direction=direction, method="two_point",
                            verbose=verbose)


def three_point_transform(IM, basis_dir='.', dr=1, direction="inverse",
                          verbose=False):
    return _dasch_transform(IM, basis_dir=basis_dir, dr=dr,
                            direction=direction, method="three_point",
                            verbose=verbose)


def onion_peeling_transform(IM, basis_dir='.', dr=1, direction="inverse",
                            verbose=False):
    return _dasch_transform(IM, basis_dir=basis_dir, dr=dr,
                            direction=direction, method="onion_peeling",
                            verbose=verbose)


two_point_transform.__doc__ =\
            _dasch_parameter_docstring.replace("dasch_method", "two-point")
three_point_transform.__doc__ =\
            _dasch_parameter_docstring.replace("dasch_method", "three-point")
onion_peeling_transform.__doc__ =\
            _dasch_parameter_docstring.replace("dasch_method", "onion-peeling")


def _dasch_transform(IM, basis_dir='.', dr=1, direction="inverse",
                     method="three_point", verbose=False):
    global _D, _method

    if direction != 'inverse':
        raise ValueError('Forward "two_point" transform not implemented')

    # make sure that the data has 2D shape
    IM = np.atleast_2d(IM)

    rows, cols = IM.shape

    if cols < 2 and method == "two_point":
        raise ValueError('"two_point" requires image width (cols) > 2')

    if cols < 3 and method == "three_point":
        raise ValueError('"three_point" requires image width (cols) > 3')

    _D = get_bs_cached(method, cols, basis_dir=basis_dir, verbose=verbose)

    inv_IM = dasch_transform(IM, _D)

    if rows == 1:
        inv_IM = inv_IM[0]  # flatten array

    return inv_IM/dr


def dasch_transform(IM, D):
    """Inverse Abel transform using the given deconvolution D-operator array.

    Parameters
    ----------
    IM : 2D numpy array
        image data

    D : 2D numpy array
        deconvolution operator array, of shape (cols, cols)

    Returns
    -------
    inv_IM : 2D numpy array
        inverse Abel transform according to deconvolution operator D
    """

    # one-line Abel transform - dot product of each row of IM with D
    return np.tensordot(IM, D, axes=(1, 1))


def _bs_two_point(cols):
    """deconvolution function for two_point.

    Parameters
    ----------
    cols : int
        width of the image
    """

    # function Eq. (9)  for j >= i
    def J(i, j):
        return np.log((np.sqrt((j + 1)**2 - i**2) + j + 1) /
                      (np.sqrt(j**2 - i**2) + j))/np.pi

    # Eq. (8, 9) D-operator basis, is 0 for j < i
    D = np.zeros((cols, cols))

    # diagonal i == j
    Ii, Jj = np.diag_indices(cols)
    Ii = Ii[1:]  # exclude special case i=j=0
    Jj = Jj[1:]
    D[Ii, Jj] = J(Ii, Jj)

    # upper triangle j > i
    Iu, Ju = np.triu_indices(cols, k=1)
    Iu = Iu[1:]  # exclude special case [0, 1]
    Ju = Ju[1:]
    D[Iu, Ju] = J(Iu, Ju) - J(Iu, Ju-1)

    # special cases
    D[0, 1] = J(0, 1) - 2/np.pi
    D[0, 0] = 2/np.pi

    return D


def _bs_three_point(cols):
    """deconvolution function for three_point.

    Parameters
    ----------
    cols : int
        width of the image
    """

    # function Eq. (7)  for j >= i
    def I0diag(i, j):
        return np.log((np.sqrt((2*j+1)**2-4*i**2) + 2*j+1)/(2*j))/(2*np.pi)

    # j > i
    def I0(i, j):
        return np.log(((np.sqrt((2*j + 1)**2 - 4*i**2) + 2*j + 1)) /
                      (np.sqrt((2*j - 1)**2 - 4*i**2) + 2*j - 1))/(2*np.pi)

    # i = j  NB minus -2I_ij typo in Dasch paper
    def I1diag(i, j):
        return np.sqrt((2*j+1)**2 - 4*i**2)/(2*np.pi) - 2*j*I0diag(i, j)

    # j > i
    def I1(i, j):
        return (np.sqrt((2*j+1)**2 - 4*i**2) -
                np.sqrt((2*j-1)**2 - 4*i**2))/(2*np.pi) - 2*j*I0(i, j)

    D = np.zeros((cols, cols))

    # matrix indices ------------------
    # i = j
    I, J = np.diag_indices(cols)
    I = I[1:]
    J = J[1:]  # drop special cases (0,0), (0,1)

    # j = i - 1
    Ib, Jb = I, J-1

    # j = i + 1
    Iu, Ju = I-1, J
    Iu = Iu[1:]  # drop special case (0, 1)
    Ju = Ju[1:]

    # j > i + 1
    Iut, Jut = np.triu_indices(cols, k=2)
    Iut = Iut[1:]  # drop special case (0, 2)
    Jut = Jut[1:]

    # D operator matrix ------------------
    # j = i - 1
    D[Ib, Jb] = I0diag(Ib, Jb+1) - I1diag(Ib, Jb+1)

    # j = i
    D[I, J] = I0(I, J+1) - I1(I, J+1) + 2*I1diag(I, J)

    # j = i + 1
    D[Iu, Ju] = I0(Iu, Ju+1) - I1(Iu, Ju+1) + 2*I1(Iu, Ju) -\
                I0diag(Iu, Ju-1) - I1diag(Iu, Ju-1)

    # j > i + 1
    D[Iut, Jut] = I0(Iut, Jut+1) - I1(Iut, Jut+1) + 2*I1(Iut, Jut) -\
                  I0(Iut, Jut-1) - I1(Iut, Jut-1)

    # special cases (that switch between I0, I1 cases)
    D[0, 2] = I0(0, 3) - I1(0, 3) + 2*I1(0, 2) - I0(0, 1) - I1(0, 1)
    D[0, 1] = I0(0, 2) - I1(0, 2) + 2*I1(0, 1) - 1/np.pi
    D[0, 0] = I0(0, 1) - I1(0, 1) + 1/np.pi

    return D


def _bs_onion_peeling(cols):
    """deconvolution function for onion_peeling.

    Parameters
    ----------
    cols : int
        width of the image
    """

    # weight matrix
    W = np.zeros((cols, cols))

    # diagonal elements i = j, Eq. (11)
    I, J = np.diag_indices(cols)
    W[I, J] = np.sqrt((2*J+1)**2 - 4*I**2)

    # upper triangle j > i,  Eq. (11)
    Iu, Ju = np.triu_indices(cols, k=1)
    W[Iu, Ju] = np.sqrt((2*Ju + 1)**2 - 4*Iu**2) -\
                np.sqrt((2*Ju - 1)**2 - 4*Iu**2)

    # operator used in Eq. (1)
    D = inv(W)

    return D


def get_bs_cached(method, cols, basis_dir='.', verbose=False):
    """load Dasch method deconvolution operator array from cache, or disk.
    Generate and store if not available.

    Checks whether ``method`` deconvolution array has been previously
    calculated, or whether the file ``{method}_basis_{cols}.npy`` is
    present in `basis_dir`.

    Either, assign, read, or generate the deconvolution array
    (saving it to file).


    Parameters
    ----------
    method : str
        Abel transform method ``onion_peeling``, ``three_point``, or
        ``two_point``

    cols : int
        width of image

    basis_dir : str or None
        path to the directory for saving or loading the deconvolution array.
        For `None` do not save the deconvolution operator array

    verbose: boolean
        print information (mainly for debugging purposes)

    Returns
    -------
    D: numpy 2D array of shape (cols, cols)
       deconvolution operator array for the associated method

    file.npy: file
       saves `D`, the deconvolution array to file name:
       ``{method}_basis_{cols}.npy``

    """

    global _D, _method, _source

    # check whether the deconvolution operator array is cached
    if _D is not None:
        if _D.shape[0] >= cols and _method == method:
            if verbose:
                print('Using memory cached deconvolution operator array,'
                      ' shape {}'.format(_D.shape))
            _source = 'cache'
            return _D[:cols, :cols]  # sliced to correct size

    D_name = "{}_basis_{}.npy".format(method, cols)
    D_generator = {
        "onion_peeling": abel.dasch._bs_onion_peeling,
        "three_point": abel.dasch._bs_three_point,
        "two_point": abel.dasch._bs_two_point
    }

    _method = method

    # read deconvolution operator array if available
    if basis_dir is not None:
        path_to_basis_files = os.path.join(basis_dir, method+'_basis*')
        basis_files = glob.glob(path_to_basis_files)
        for bf in basis_files:
            if int(bf.split('_')[-1].split('.')[0]) >= cols:
                # relies on file order
                if verbose:
                    print("Loading deconvolution operator array from"
                          " file {:s}".format(bf))
                # slice to size
                _D = np.load(bf)[:cols, :cols]
                _source = 'file'
                return _D

    if verbose:
        print("A suitable deconvolution array for '{:s}' was not found.".
              format(method))
        print("A new array will be generated.")

    _D = D_generator[method](cols)
    _source = 'generated'

    if basis_dir is not None:
        path_to_basis_file = os.path.join(basis_dir, D_name)
        np.save(path_to_basis_file, _D)
        if verbose:
            print("\ndeconvolution operator array saved to '{:s}"
                  .format(path_to_basis_file))

    return _D


def cache_cleanup():
    """
    Utility function.

    Frees the memory caches created by ``get_bs_cached()``.
    This is usually pointless, but might be required after working
    with very large images, if more RAM is needed for further tasks.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    global _D, _method, _source

    _D = None
    _method = None
    _source = None
