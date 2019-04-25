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
import sys


def _ensure_list(x):
    """
    Convert the argument to a list (a scalar becomes a single-element list).
    """
    return [x] if np.ndim(x) == 0 else list(x)


def _roundsf(x, n):
    """
    Round to n significant digits
    """
    return float('{:.{p}g}'.format(x, p=n))


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
    n : int or sequence of int
        array size(s) for the benchmark (assuming 2D square arrays (*n*, *n*))
    select : str or sequence of str
        methods to benchmark. Use ``'all'`` (default) for all available or
        choose any combination of individual methods::

            select=['basex', 'direct_C', 'direct_Python', 'hansenlaw',
                    'linbasex', 'onion_bordas, 'onion_peeling', 'two_point',
                    'three_point']

    repeat : int
        repeat each benchmark at least this number of times to get the average
        values
    t_min : float
        repeat each benchmark for at least this number of seconds to get the
        average values
    t_max : float
        do not benchmark methods at array sizes when this is expected to take
        longer than this number of seconds. Notice that benchmarks for the
        smallest size from **n** are always run and that the estimations can be
        off by a factor of 2 or so.
    verbose : boolean
        determines whether benchmark progress should be reported (to stderr)

    Attributes
    -------
    n : list of int
        array sizes from the parameter **n**, sorted in ascending order
    bs, fabel, iabel : dict of list of float
        benchmark results — dictionaries for

            bs
                basis-set generation
            fabel
                forward Abel transform
            iabel
                inverse Abel transform

        with methods as keys and lists of timings in milliseconds as entries.
        Timings correspond to array sizes in :py:attr:`AbelTiming.n`; for
        skipped benchmarks (see **t_max**) they are ``np.nan``.

    Notes
    -----
    The results can be output in a nice format by simply
    ``print(AbelTiming(...))``.

    Keep in mind that most methods have :math:`O(n^2)` memory and
    :math:`O(n^3)` time complexity, so going from *n* = 501 to *n* = 5001
    would require about 100 times more memory and take about 1000 times longer.
    """
    def __init__(self, n=[301, 501], select='all',
                 repeat=1, t_min=0.1, t_max=np.inf,
                 verbose=True):
        self.n = sorted(_ensure_list(n))
        select = _ensure_list(select)
        self.repeat = repeat
        self.t_max = t_max
        self.verbose = verbose
        # create the timing function
        self._time = Timent(skip=-1, repeat=repeat, duration=t_min).time

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
        # inverse speed for time estimations
        self._pace = {}

        # Loop over all image sizes
        for ni in self.n:
            self.ni = int(ni)
            # image height and half-width
            self.h, self.w = self.ni, self.ni // 2 + 1
            # We transform a rectangular image, since we are making the
            # assumption that we are transforming just the "right side" of
            # a square image.
            # see: https://github.com/PyAbel/PyAbel/issues/207
            self._vprint('n =', self.ni)

            # The following code tries to catch the interruption signal
            # (Ctrl+C) to abort as soon as possible but preserve the available
            # results. Setting a negative time limit makes all remaining
            # benchmarks to skip (calling them is still needed to fill the
            # results with nans).

            # create needed images (half and/or whole)
            if (self.t_max >= 0):  # (do not create while aborting)
                try:
                    if methods & need_half:
                        self.half_image = np.random.randn(self.h, self.w)
                    if methods & need_whole:
                        self.whole_image = np.random.randn(self.h, self.h)
                except (KeyboardInterrupt, MemoryError) as e:
                    self._vprint(repr(e) + ' during image creation!'
                                 ' Skipping the rest...')
                    self.t_max = -1.0
                    # (the images will not be used, so leaving them as is)

            # call benchmark (see below) for each method at this image size
            for method in methods:
                self._vprint(' ', method)
                try:
                    getattr(self, '_time_' + method)()
                except (KeyboardInterrupt, MemoryError) as e:
                    self._vprint('\n' + repr(e) + '! Skipping the rest...')
                    self.t_max = -1.0
                    # rerun this interrupted benchmark to nan-fill its results
                    getattr(self, '_time_' + method)()

            # discard images
            self.half_image = None
            self.whole_image = None
        self._vprint('')

    def _vprint(self, *args, **kwargs):
        """
        Print to stderr, only if verbose=True.
        """
        if self.verbose:
            print(*args, file=sys.stderr, **kwargs)
            sys.stderr.flush()  # (Python 3 buffers stderr. Why?!)

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

    def _skip(*param):
        """
        Decorator for benchmarking functions.
        Adds a check whether the estimated execution time would exceed t_max.
        If so, fills the results with np.nan, otherwise executes the
        benchmarking code.

        Parameters are tuples "(kind, method)". Either item can be a list, then
        all combinations of kind(s) and method(s) are implied. Altogether the
        set of these kind–method pairs must be the same as in the "normal"
        execution results.
        """
        # assemble all kind–method pairs
        res_keys = []
        for p in param:
            res_keys += itertools.product(*map(_ensure_list, p))

        def decorator(f):
            method = f.__name__[6:]  # (remove initial "_time_")

            def decorated(self):
                # get the estimated time (use 0 if cannot) and report it
                t_est = self._pace.get(method, 0) * self.ni**3
                self._vprint('    estimated ' +
                             ('{:g} s'.format(_roundsf(t_est, 2)) if t_est
                              else '???'), end='')
                # skip the benchmark if it would take too long
                if t_est > self.t_max:
                    self._vprint(' -- skipped')
                    # fill the results with nan
                    for k, m in res_keys:
                        self._append(k, m, np.nan)
                    return
                else:  # otherwise run the benchmark
                    f(self)
                    # calculate the actual total time and report it
                    t = (sum(self.res[k][m][-1] for k, m in res_keys) *
                         self.repeat) / 1000  # [ms] -> [s]
                    self._vprint(', actually {:.3f} s'.format(t))
                    # save the pace for future estimations
                    self._pace[method] = t / self.ni**3
            return decorated
        return decorator

    # Benchmarking functions for each method.
    # Must be named "_time_method", where "method" is as in "select".
    # Do not take or return anything, but use instance variables:
    # parameters:
    #   self.ni, self.h, self.w -- image size, height, half-width,
    #   self.whole_image, self.half_image -- image (part) to transform
    # results:
    #   self.res[kind][method] = [timings] -- appended for each image size,
    #                                         use np.nan for skipped points
    #     kind = 'bs' (basis), 'forward', 'inverse' -- as applicable
    #     method -- as above, but can also include variants (like in basex)

    @_skip(('bs', 'basex'),
           (['inverse', 'forward'], ['basex', 'basex(var.reg.)']))
    def _time_basex(self):
        # benchmark the basis generation (default parameters)
        def gen_basis():
            basex.cache_cleanup()
            basex.get_bs_cached(self.w, basis_dir=None)
        self._benchmark('bs', 'basex',
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

    @_skip((['inverse', 'forward'], 'direct_C'))
    def _time_direct_C(self):
        for direction in ['inverse', 'forward']:
            self._benchmark(direction, 'direct_C',
                            direct.direct_transform,
                            self.half_image, direction=direction, backend='C')

    @_skip((['inverse', 'forward'], 'direct_Python'))
    def _time_direct_Python(self):
        for direction in ['inverse', 'forward']:
            self._benchmark(direction, 'direct_Python',
                            direct.direct_transform,
                            self.half_image, direction=direction,
                            backend='python')

    @_skip((['inverse', 'forward'], 'hansenlaw'))
    def _time_hansenlaw(self):
        for direction in ['inverse', 'forward']:
            self._benchmark(direction, 'hansenlaw',
                            hansenlaw.hansenlaw_transform,
                            self.half_image, direction=direction)

    @_skip((['bs', 'inverse'], 'linbasex'))
    def _time_linbasex(self):
        # benchmark the basis generation (default parameters)
        def gen_basis():
            linbasex.cache_cleanup()
            linbasex.get_bs_cached(self.h, basis_dir=None)
        self._benchmark('bs', 'linbasex',
                        gen_basis)

        # get the basis (is already cached)
        basis = linbasex.get_bs_cached(self.h, basis_dir=None)
        # benchmark the transform
        self._benchmark('inverse', 'linbasex',
                        linbasex._linbasex_transform_with_basis,
                        self.whole_image, basis)

        # discard all caches
        linbasex.cache_cleanup()

    @_skip(('inverse', 'onion_bordas'))
    def _time_onion_bordas(self):
        self._benchmark('inverse', 'onion_bordas',
                        onion_bordas.onion_bordas_transform,
                        self.half_image)

    # (Generic function for all Dasch methods; not called directly.)
    def _time_dasch(self, method):
        # benchmark the basis generation (default parameters)
        def gen_basis():
            dasch.cache_cleanup()
            dasch.get_bs_cached(method, self.w, basis_dir=None)
        self._benchmark('bs', method,
                        gen_basis)

        # get the transform matrix (is already cached)
        D = dasch.get_bs_cached(method, self.w, basis_dir=None)
        # benchmark the transform
        self._benchmark('inverse', method,
                        dasch.dasch_transform,
                        self.half_image, D)

        # discard all caches
        dasch.cache_cleanup()

    @_skip((['bs', 'inverse'], 'onion_peeling'))
    def _time_onion_peeling(self):
        self._time_dasch('onion_peeling')

    @_skip((['bs', 'inverse'], 'two_point'))
    def _time_two_point(self):
        self._time_dasch('two_point')

    @_skip((['bs', 'inverse'], 'three_point'))
    def _time_three_point(self):
        self._time_dasch('three_point')

    # (End of benchmarking functions.)

    def __repr__(self):
        import platform

        out = ['PyAbel benchmark run on {}\n'.format(platform.processor()),
               'time in milliseconds']

        # field widths are chosen to accommodate up to:
        #   method = 15 characters
        #   ni = 99999 (would require at least 75 GB RAM)
        #   time = 9999999.9 ms (almost 3 hours)
        # data columns are 9 characters wide and separated by 3 spaces
        TITLE_FORMAT = '=== {} ==='
        HEADER_ROW = 'Method         ' + \
                     ''.join(['   {:>9}'.
                              format('n = {}'.format(ni)) for ni in self.n])
        SEP_ROW = '-' * len(HEADER_ROW)
        ROW_FORMAT = '{:15}' + '   {:9.1f}' * len(self.n)

        def print_benchmark(name, res):
            title = '{:=<{w}}'.format(TITLE_FORMAT.format(name),
                                      w=len(SEP_ROW))
            out = ['\n' + title + '\n']
            out += [HEADER_ROW]
            out += [SEP_ROW]
            for name, row in sorted(res.items()):
                out += [ROW_FORMAT.format(name, *row)]
            return out

        if self.bs:
            out += print_benchmark('Basis generation', self.bs)
            out += ['']
        if self.fabel:
            out += print_benchmark('Forward Abel transform', self.fabel)
            out += ['']
        if self.iabel:
            out += print_benchmark('Inverse Abel transform', self.iabel)

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
