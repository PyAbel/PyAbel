# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .tools import get_image_quadrants

class AbelTiming(object):
    def __init__(self, n=[201, 401], n_max_bs=500):
        """
        Benchmark performance of different iAbel/fAbel implementations.

        Parameters
        ----------
          - n: a list of arrays sizes for the benchmark (assuming 2D
            square arrays (n,n) )
          - n_max_bs: since the basis sets generation takes a long time,
            do not run this benchmark for implementations that use basis sets
            for n > n_max_bs
        """
        from .basex import get_basis_sets_cached, basex_transform
        from .hansenlaw import iabel_hansenlaw
        from timeit import Timer

        self.n = n

        NREPEAT = 10

        res_fabel = {}
        res_iabel = {'BASEX':     {'bs': [], 'tr': []},
               'HansenLaw': {'tr': []}
               }
        for ni in n:
            x = np.random.randn(ni,ni)
            if ni <= n_max_bs:
                bs = get_basis_sets_cached(ni, basis_dir=None)
                res_iabel['BASEX']['bs'].append(
                    Timer(lambda: get_basis_sets_cached(ni, basis_dir=None)).timeit(number=1))
                res_iabel['BASEX']['tr'].append(
                    Timer(lambda: basex_transform(x, *bs)).timeit(number=NREPEAT)/NREPEAT)
            else:
                res_iabel['BASEX']['bs'].append( np.nan)
                res_iabel['BASEX']['tr'].append( np.nan)

            res_iabel['HansenLaw']['tr'].append(
                Timer(lambda: iabel_hansenlaw(x, verbose=False)).timeit(number=NREPEAT)/NREPEAT)
        self.fabel = res_fabel
        self.iabel = res_iabel


    def __repr__(self):
        import platform
        from itertools import chain

        out = []
        out += ['PyAbel benchmark run on {}\n'.format(platform.processor())]

        LABEL_FORMAT =     '|'.join([' Implementation '] \
                + ['    n = {:<12} '.format(ni) for ni in self.n])
        TR_ROW_FORMAT = '|'.join(['{:>15} '] + [ ' {:8.1e}            ' ]*len(self.n))
        BS_ROW_FORMAT = '|'.join(['{:>15} '] + [ ' {:8.1e} ({:8.1e}) ' ]*len(self.n))
        SEP_ROW = ' ' + '-'*(22 + (17+1)*len(self.n))

        HEADER_ROW = ' ========= {:>10} Abel implementations ==========\n' \
                ' time to solution [s] -> transform (basis sets generation)\n'


        def print_benchmark(name, res):
            out = [HEADER_ROW.format(name)]
            if res:
                out += [LABEL_FORMAT]
                out += [SEP_ROW]
                for name, row  in res.items():
                    if 'bs' in row:
                        pars = list(chain(*zip(row['tr'], row['bs'])))
                        out += [BS_ROW_FORMAT.format(name, *pars)]
                    else:
                        out += [TR_ROW_FORMAT.format(name, *row['tr'])]
            return out

        out += print_benchmark('Direct', self.fabel)
        out += print_benchmark('Inverse', self.iabel)

        return '\n'.join(out)



def is_symmetric(arr, i_sym=True, j_sym=True):
    """
    Takes in an array of shape (n, m) and check if it is symmetric

    Parameters:
       - arr: 1D or 2D array
       - i_sym: array is symmetric with respect to the 1st axis
       - j_sym: array is symmetric with respect to the 2nd axis

    Returns:
       a binary array with the symmetry condition for the corresponding quadrants.
       The global validity can be checked with `array.all()`

    Note: if both i_sym=True and i_sym=True, the input array is checked
    for polar symmetry.

    See https://github.com/PyAbel/PyAbel/issues/34#issuecomment-160344809 for
    the defintion of a center of the image.
    """

    Q0, Q1, Q2, Q3 = get_image_quadrants(arr)


    if i_sym and not j_sym:
        valid_flag = [ np.allclose(np.fliplr(Q1), Q0),
                       np.allclose(np.fliplr(Q2), Q3) ]
    elif not i_sym and j_sym:
        valid_flag = [ np.allclose(np.flipud(Q1), Q2),
                       np.allclose(np.flipud(Q0), Q3) ]
    elif i_sym and j_sym:
        valid_flag = [ np.allclose(np.flipud(np.fliplr(Q1)), Q3),
                       np.allclose(np.flipud(np.fliplr(Q0)), Q2) ]
    else:
        raise ValueError('Checking for symmetry with both i_sym=False and j_sym=False'\
                         'does not make sens!') 

    return np.array(valid_flag)



def absolute_ratio_benchmark(analytical, recon, kind='inverse'):
    """
    Check the absolute ratio between an analytical function and the result
     of a inv. Abel reconstruction.

    Parameters
    ----------
      - analytical: one of the classes from abel.analytical, initialized
      - recon: 1D ndarray: a reconstruction (i.e. inverse abel) given by some PyAbel implementation
    """
    mask = analytical.mask_valid

    if kind == 'inverse':
        func = analytical.func
    elif kind == 'direct':
        func = analytical.abel

    err = func[mask]/recon[mask]
    return err

