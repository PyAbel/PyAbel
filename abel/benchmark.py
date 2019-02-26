# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
from . import basex
from . import dasch
from . import direct
from . import linbasex
from . import hansenlaw
from . import onion_bordas
from . import tools

from timeit import default_timer as timer
import itertools


class Timent(object):
    """
    Helper class for measuring execution times.

    The constructor only initializes the timing-procedure parameters.
    Use the :py:meth:`.time` method to run it for particular functions.

    Parameters
    ----------
    skip : int
        number of "warm-up" iterations to perform before the measurements.
        Can be specified as a negative number, then ``abs(skip)``
        "warm-up" iterations are performed, but if this took more than
        **duration** seconds, they are accounted towards the measured
        iterations.
    repeat : int
        minimal number of measured iterations to perform.
        Must be positive.
    duration : float
        minimal duration (in seconds) of the measurements.
    """
    def __init__(self, skip=0, repeat=1, duration=0.0):
        self.skip = int(skip)
        self.repeat = int(repeat)
        self.duration = float(duration)

    def time(self, func, *args, **kwargs):
        """
        Repeatedly executes a function at least **repeat** times and for at
        least **duration** seconds (see above), then returns the average time
        per iteration.
        The actual number of measured iterations can be retrieved from
        :py:attr:`Timent.count`.

        Parameters
        ----------
        func : callable
            function to execute
        *args, **kwargs : any, optional
            parameters to pass to **func**

        Returns
        -------
        float
            average function execution time

        Notes
        -----
        The measurements overhead can be estimated by executing ::

            Timent(...).time(lambda: None)

        with a sufficiently large number of iterations (to avoid rounding
        errors due to the finite timer precision).
        In 2018, this overhead was on the order of 100 ns per iteration.
        """
        # Execute "skip" iterations unconditionally
        t0 = timer()
        for i in range(abs(self.skip)):
            func(*args, **kwargs)
        t = timer()
        # if they took longer than "duration" and should be included
        if self.skip < 0 and t - t0 > self.duration:
            # account for them in the "repeat" loop
            start = -self.skip
            if start > self.repeat:
                self.repeat = start
        else:
            # otherwise -- reset the timer and proceed normally
            start = 0
            t0 = timer()

        # Execute "repeat" iterations (maybe accounting for the "skipped")
        for i in range(start, self.repeat):
            func(*args, **kwargs)

        # Continue iterations until "duration" time passes
        for i in itertools.count(self.repeat):
            t = timer()
            if t - t0 > self.duration:
                self.count = i  # save the total number of measured iterations
                break
            func(*args, **kwargs)

        # Return the average time per iteration
        return (t - t0) / self.count


class AbelTiming(object):
    """
    Benchmark performance of different Abel implementations
    (basis generation, forward and inverse transforms, as applicable).

    Parameters
    ----------
    n : list of int
        array sizes for the benchmark (assuming 2D square arrays (*n*, *n*))
    select : list of str
        methods to benchmark. Use ``['all']`` (default) for all available or
        choose any combination of individual methods::

            select=['basex', 'direct_C', 'direct_Python', 'hansenlaw',
                    'linbasex', 'onion_bordas, 'onion_peeling', 'two_point',
                    'three_point']

    n_max_bs : int
        since the basis-sets generation takes a long time, do not run
        benchmarks with *n* > **n_max_bs** for implementations that use basis
        sets
    n_max_slow : int
        do not run benchmarks with *n* > **n_max_slow** for "slow"
        implementations, so far including only "direct_Python"
    repeat : int
        repeat each benchmark at least this number of times to get the average
        values
    duration : float
        repeat each benchmark for at least this number of seconds to get the
        average values

    Attributes
    -------
    bs, fabel, iabel : dict of list of float
        benchmark results — dictionaries for

            bs
                basis-set generation
            fabel
                forward Abel transform
            iabel
                inverse Abel transform

        with methods as keys and lists of timings in milliseconds as entries
        (corresponding to array sizes from **n**, also available as
        :py:attr:`AbelTiming.n`).

    Notes
    -----
    The results can be output in a nice format by simply
    ``print(AbelTiming(...))``.

    Keep in mind that most methods have :math:`O(n^2)` memory and
    :math:`O(n^3)` time complexity, so going from *n* = 501 to *n* = 5001
    would require about 100 times more memory and take about 1000 times longer.
    """
    def __init__(self, n=[301, 501], select=['all', ], n_max_bs=700,
                 n_max_slow=700, repeat=1, duration=0.1):
        self.n = n
        self.n_max_bs = n_max_bs
        self.n_max_slow = n_max_slow
        # create the timing function
        self._time = Timent(skip=-1, repeat=repeat, duration=duration).time

        # which methods need half and whole images
        need_half = frozenset([
            'basex',
            'direct_C',
            'direct_Python',
            'hansenlaw',
            'onion_bordas',
            'onion_peeling',
            'two_point',
            'three_point',
        ])
        need_whole = frozenset([
            'linbasex',
        ])
        # all available methods (= union of the above sets)
        all_methods = need_half | need_whole
        # remove direct_C, if not supported
        if not direct.cython_ext:
            all_methods = all_methods - frozenset(['direct_C'])

        # Select methods
        if 'all' in select:
            methods = all_methods
        else:
            methods = set()
            for method in select:
                if method not in all_methods:
                    print('Warning: Unsupported method "{}" ignored!'.
                          format(method))
                else:
                    methods.add(method)
        if not methods:
            raise ValueError('At least one valid method must be specified!')

        # dictionary for the results
        self.res = {'bs': {},
                    'forward': {},
                    'inverse': {}}
        # same results as separate dictionaries (aliases to the above)
        self.bs = self.res['bs']
        self.fabel = self.res['forward']
        self.iabel = self.res['inverse']

        # Loop over all image sizes
        for ni in self.n:
            self.ni = int(ni)
            # image height and half-width
            self.h, self.w = self.ni, self.ni // 2 + 1
            # We transform a rectangular image, since we are making the
            # assumption that we are transforming just the "right side" of
            # a square image.
            # see: https://github.com/PyAbel/PyAbel/issues/207

            # create needed images (half and/or whole)
            if methods & need_half:
                self.half_image = np.random.randn(self.h, self.w)
            if methods & need_whole:
                self.whole_image = np.random.randn(self.h, self.h)

            # call benchmark (see below) for each method at this image size
            for method in methods:
                getattr(self, '_time_' + method)()

            # discard images
            self.half_image = None
            self.whole_image = None

    def _append(self, kind, method, result):
        """
        Store one result, ensuring that the results array exists.
        """
        if method not in self.res[kind]:
            self.res[kind][method] = []
        self.res[kind][method].append(result)

    def _benchmark(self,
                   kind, method,
                   func, *args, **kwargs):
        """
        Run benchmark for the function with arguments and store the result.
        """
        self._append(kind, method,
                     self._time(func, *args, **kwargs) * 1000)  # [s] to [ms]

    # Benchmarking functions for each method.
    # Must be named "_time_method", where "method" is as in "select".
    # Do not take or return anything, but use instance variables:
    # parameters:
    #   self.ni, self.h, self.w -- image size, height, half-width,
    #   self.n_max_bs, self.n_max_slow -- image-size limits
    #   self.whole_image, self.half_image -- image (part) to transform
    # results:
    #   self.res[kind][method] = [timings] -- appended for each image size,
    #                                         use np.nan for skipped points
    #     kind = 'bs' (basis), 'forward', 'inverse' -- as applicable
    #     method -- as above, but can also include variants (like in basex)

    def _time_basex(self):
        # skip if too large
        if self.ni > self.n_max_bs:
            self._append('bs', 'basex_bs', np.nan)
            for direction in ['inverse', 'forward']:
                for method in ['basex', 'basex(var.reg.)']:
                    self._append(direction, method, np.nan)

        # benchmark the basis generation (default parameters)
        def gen_basis():
            basex.cache_cleanup()
            basex.get_bs_cached(self.w, basis_dir=None)
        self._benchmark('bs', 'basex_bs',
                        gen_basis)

        # benchmark all transforms
        for direction in ['inverse', 'forward']:  # (default first)
            # get the transform matrix (default parameters)
            A = basex.get_bs_cached(self.w, basis_dir=None,
                                    direction=direction)
            # benchmark the transform itself
            self._benchmark(direction, 'basex',
                            basex.basex_core_transform,
                            self.half_image, A)

            # benchmark the transform with variable regularization
            def basex_var():
                A = basex.get_bs_cached(self.w, reg=1.0+np.random.random(),
                                        basis_dir=None, direction=direction)
                basex.basex_core_transform(self.half_image, A)
            self._benchmark(direction, 'basex(var.reg.)',
                            basex_var)

            # discard the unneeded transform matrix
            basex.cache_cleanup(direction)

        # discard all caches
        basex.cache_cleanup()

    def _time_direct_C(self):
        for direction in ['inverse', 'forward']:
            self._benchmark(direction, 'direct_C',
                            direct.direct_transform,
                            self.half_image, direction=direction, backend='C')

    def _time_direct_Python(self):
        for direction in ['inverse', 'forward']:
            if self.ni > self.n_max_slow:  # skip if too large
                self._append(direction, 'direct_Python', np.nan)
            else:
                self._benchmark(direction, 'direct_Python',
                                direct.direct_transform,
                                self.half_image, direction=direction,
                                backend='python')

    def _time_hansenlaw(self):
        for direction in ['inverse', 'forward']:
            self._benchmark(direction, 'hansenlaw',
                            hansenlaw.hansenlaw_transform,
                            self.half_image, direction=direction)

    def _time_linbasex(self):
        # skip if too large
        if self.ni > self.n_max_bs:
            self._append('bs', 'linbasex_bs', np.nan)
            self._append('inverse', 'linbasex', np.nan)

        # benchmark the basis generation (default parameters)
        def gen_basis():
            linbasex.cache_cleanup()
            linbasex.get_bs_cached(self.h, basis_dir=None)
        self._benchmark('bs', 'linbasex_bs',
                        gen_basis)

        # get the basis (is already cached)
        basis = linbasex.get_bs_cached(self.h, basis_dir=None)
        # benchmark the transform
        self._benchmark('inverse', 'linbasex',
                        linbasex._linbasex_transform_with_basis,
                        self.whole_image, basis)

        # discard all caches
        linbasex.cache_cleanup()

    def _time_onion_bordas(self):
        self._benchmark('inverse', 'onion_bordas',
                        onion_bordas.onion_bordas_transform,
                        self.half_image)

    # (Generic function for all Dasch methods; not called directly.)
    def _time_dasch(self, method):
        # skip if too large
        if self.ni > self.n_max_bs:
            self._append('bs', method + '_bs', np.nan)
            self._append('inverse', method, np.nan)

        # benchmark the basis generation (default parameters)
        def gen_basis():
            dasch.cache_cleanup()
            dasch.get_bs_cached(method, self.w, basis_dir=None)
        self._benchmark('bs', method + '_bs',
                        gen_basis)

        # get the transform matrix (is already cached)
        D = dasch.get_bs_cached(method, self.w, basis_dir=None)
        # benchmark the transform
        self._benchmark('inverse', method,
                        dasch.dasch_transform,
                        self.half_image, D)

        # discard all caches
        dasch.cache_cleanup()

    def _time_onion_peeling(self):
        self._time_dasch('onion_peeling')

    def _time_two_point(self):
        self._time_dasch('two_point')

    def _time_three_point(self):
        self._time_dasch('three_point')

    # (End of benchmarking functions.)

    def __repr__(self):
        import platform
        from itertools import chain

        out = []
        out += ['PyAbel benchmark run on {}\n'.format(platform.processor())]
        out += ['time in milliseconds']

        LABEL_FORMAT = 'Implementation  ' +\
                       ''.join(['    n = {:<9} '.format(ni) for ni in self.n])
        ROW_FORMAT = '{:>16} ' + ' {:8.1f}         '*len(self.n)
        SEP_ROW = '' + '-'*(22 + (17+1)*len(self.n))

        HEADER_ROW = '\n========= {:>8} Abel implementations ==========\n'

        def print_benchmark(name, res):
            out = [HEADER_ROW.format(name)]
            if res:
                out += [LABEL_FORMAT]
                out += [SEP_ROW]
                for name, row in sorted(res.items()):
                    out += [ROW_FORMAT.format(name, *row)]
            return out

        out += print_benchmark('Basis', self.bs)
        out += ['']
        out += print_benchmark('Forward', self.fabel)
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
    The globa

    Notes
    -----
    If both **i_sym** = ``True`` and **j_sym** = ``True``, the input array is
    checked for polar symmetry.

    See https://github.com/PyAbel/PyAbel/issues/34#issuecomment-160344809
    for the defintion of a center of the image.
    """

    Q0, Q1, Q2, Q3 = tools.symmetry.get_image_quadrants(
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
    of a inverse Abel reconstruction.

    Parameters
    ----------
    analytical : one of the classes from analytical, initialized

    recon : 1D ndarray
        a reconstruction (i.e. inverse abel)
        given by some PyAbel implementation
    """
    mask = analytical.mask_valid

    if kind == 'inverse':
        func = analytical.func
    elif kind == 'forward':
        func = analytical.abel

    err = func[mask]/recon[mask]
    return err
