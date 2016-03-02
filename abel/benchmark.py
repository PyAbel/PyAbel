# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


import abel


class AbelTiming(object):
    def __init__(self, n=[301, 501], n_max_bs=700):
        """
        Benchmark performance of different iAbel/fAbel implementations.

        Parameters
        ----------
        n: integer
            a list of arrays sizes for the benchmark (assuming 2D square arrays (n,n))
        n_max_bs: integer
            since the basis sets generation takes a long time,
            do not run this benchmark for implementations that use basis sets
            for n > n_max_bs
        """
        from timeit import Timer

        self.n = n

        NREPEAT = 5

        res_fabel = {'HansenLaw':     {'tr': []},
                     'direct_Python': {'tr': []}}
        res_iabel = {'BASEX':         {'bs': [], 'tr': []},
                     'Three_point':   {'bs': [], 'tr': []},
                     'HansenLaw':     {'tr': []},
                     'direct_Python': {'tr': []}}
        if abel.direct.cython_ext:
            res_fabel['direct_C'] = {'tr': []}
            res_iabel['direct_C'] = {'tr': []}

        for ni in n:
            x = np.random.randn(ni, ni)
            # direct implementations
            if ni <= n_max_bs:
                bs = abel.basex.get_bs_basex_cached(ni, ni)
                res_iabel['BASEX']['bs'].append(
                    Timer(lambda: abel.basex.get_bs_basex_cached(ni, ni)).
                    timeit(number=1))
                res_iabel['BASEX']['tr'].append(
                    Timer(lambda: abel.basex.basex_core_transform(x, *bs)).
                    timeit(number=NREPEAT)/NREPEAT)
                res_iabel['Three_point']['bs'].append(
                    Timer(lambda: abel.three_point.
                          get_bs_three_point_cached(ni)).timeit(number=1))
                res_iabel['Three_point']['tr'].append(
                    # currently this is wrong because it also generated 
                    # the basis sets!
                    Timer(lambda: abel.three_point.three_point_transform(x)).
                          timeit(number=NREPEAT)/NREPEAT)
            else:
                res_iabel['BASEX']['bs'].append(np.nan)
                res_iabel['BASEX']['tr'].append(np.nan)
                res_iabel['Three_point']['bs'].append(np.nan)
                res_iabel['Three_point']['tr'].append(np.nan)

            res_fabel['HansenLaw']['tr'].append(
                Timer(lambda: abel.hansenlaw.hansenlaw_transform(
                      x, direction='forward')).timeit(number=NREPEAT)/NREPEAT)
            res_iabel['HansenLaw']['tr'].append(
                Timer(lambda: abel.hansenlaw.hansenlaw_transform(x,
                      x, direction='inverse')).timeit(number=NREPEAT)/NREPEAT)
            res_iabel['direct_Python']['tr'].append(
                Timer(lambda: abel.direct.direct_transform(
                      x, backend='Python',
                      direction='inverse')).timeit(number=NREPEAT)/NREPEAT)
            res_fabel['direct_Python']['tr'].append(
                Timer(lambda: abel.direct.direct_transform(
                      x, backend='Python',
                      direction='forward')).timeit(number=NREPEAT)/NREPEAT)
            if abel.direct.cython_ext:
                res_iabel['direct_C']['tr'].append(
                    Timer(lambda: abel.direct.direct_transform(
                        x, backend='C',
                        direction='inverse')).timeit(number=NREPEAT)/NREPEAT)
                res_fabel['direct_C']['tr'].append(
                    Timer(lambda: abel.direct.direct_transform(
                        x, backend='C',
                        direction='forward')).timeit(number=NREPEAT)/NREPEAT)

        self.fabel = res_fabel
        self.iabel = res_iabel

    def __repr__(self):
        import platform
        from itertools import chain

        out = []
        out += ['PyAbel benchmark run on {}\n'.format(platform.processor())]

        LABEL_FORMAT = '|'.join([' Implementation '] +
                                ['    n = {:<8} '.
                                format(ni) for ni in self.n])
        TR_ROW_FORMAT = '|'.join(['{:>15} '] + [' {:.4f}          ']\
                                 * len(self.n))
        BS_ROW_FORMAT = '|'.join(['{:>15} '] + [' {:.4f} ({:.4f}) ']\
                                 * len(self.n))
        SEP_ROW = ' ' + '-'*(22 + (17+1)*len(self.n))

        HEADER_ROW = ' ========= {:>10} Abel implementations ==========\n' \
                     'time to solution [s] -> transform'\
                     ' (basis sets generation)\n'

        def print_benchmark(name, res):
            out = [HEADER_ROW.format(name)]
            if res:
                out += [LABEL_FORMAT]
                out += [SEP_ROW]
                for name, row in res.items():
                    if 'bs' in row:
                        pars = list(chain(*zip(row['tr'], row['bs'])))
                        out += [BS_ROW_FORMAT.format(name, *pars)]
                    else:
                        out += [TR_ROW_FORMAT.format(name, *row['tr'])]
            return out

        out += print_benchmark('Direct', self.fabel)
        out += ['']
        out += print_benchmark('Inverse', self.iabel)

        return '\n'.join(out)


def is_symmetric(arr, i_sym=True, j_sym=True):
    """
    Takes in an array of shape (n, m) and check if it is symmetric

    Parameters
    ----------
    arr : 1D or 2D array
    i_sym : array
        symmetric with respect to the 1st axis
    j_sym : array
        symmetric with respect to the 2nd axis

    Returns
    -------
    a binary array with the symmetry condition for the corresponding quadrants.
    The global validity can be checked with `array.all()`

    Note: if both i_sym=True and i_sym=True, the input array is checked
    for polar symmetry.

    See https://github.com/PyAbel/PyAbel/issues/34#issuecomment-160344809
    for the defintion of a center of the image.
    """

    Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(
                                                 arr, reorient=False)

    if i_sym and not j_sym:
        valid_flag = [np.allclose(np.fliplr(Q1), Q0),
                      np.allclose(np.fliplr(Q2), Q3)]
    elif not i_sym and j_sym:
        valid_flag = [np.allclose(np.flipud(Q1), Q2),
                      np.allclose(np.flipud(Q0), Q3)]
    elif i_sym and j_sym:
        valid_flag = [np.allclose(np.flipud(np.fliplr(Q1)), Q3),
                      np.allclose(np.flipud(np.fliplr(Q0)), Q2)]
    else:
        raise ValueError('Checking for symmetry with both i_sym=False \
                          and j_sym=False does not make sense!')

    return np.array(valid_flag)


def absolute_ratio_benchmark(analytical, recon, kind='inverse'):
    """
    Check the absolute ratio between an analytical function and the result
    of a inv. Abel reconstruction.

    Parameters
    ----------
    analytical : one of the classes from abel.analytical, initialized

    recon : 1D ndarray
        a reconstruction (i.e. inverse abel)
        given by some PyAbel implementation
    """
    mask = analytical.mask_valid

    if kind == 'inverse':
        func = analytical.func
    elif kind == 'direct':
        func = analytical.abel

    err = func[mask]/recon[mask]
    return err


def main():
    # run some benchmarks!!
    time = AbelTiming(n=[201], n_max_bs=101)
    print(time)


if __name__ == '__main__':
    main()
