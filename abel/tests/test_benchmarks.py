from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from abel.benchmark import AbelTiming, DistributionsTiming, \
                           absolute_ratio_benchmark, is_symmetric
from abel.tools.analytical import StepAnalytical, GaussianAnalytical


def test_AbelTiming():
    """ Sanity check that AbelTiming works for all methods
    """
    AbelTiming(n=11, t_min=0.01, verbose=False)


def test_DistributionsTiming():
    """ Sanity check that DistributionsTiming works for all methods
    """
    DistributionsTiming(n=11, t_min=0.01)


def test_symmetry_function():
    """ Check the consistency of abel.benchmark.is_symmetric
    """
    # checking all combination of odd and even images
    for n, m in [(11, 11),
                 (10, 10),
                 (10, 12),
                 (11, 12),
                 (10, 13),
                 (11, 13)
                 ]:
        x = np.linspace(-np.pi, np.pi, n)
        y = np.linspace(-np.pi, np.pi, m)

        XX, YY = np.meshgrid(x, y)

        f1 = np.cos(XX)*np.cos(YY)
        assert_equal(is_symmetric(f1, i_sym=True, j_sym=True), True,
                     'cos(x)*cos(y) polar symmetry: n={}, m={}'.format(n, m))

        f2 = np.cos(XX)*np.sin(YY)
        assert_equal(is_symmetric(f2, i_sym=True, j_sym=False), True,
                     'cos(x)*sin(y) i symmetry: n={}, m={}'.format(n, m))

        f3 = np.sin(XX)*np.cos(YY)
        assert_equal(is_symmetric(f3, i_sym=False, j_sym=True), True,
                     'sin(x)*cos(y) j symmetry: n={}, m={}'.format(n, m))

        assert_equal(is_symmetric(f3, i_sym=True, j_sym=False), False,
                     'sin(x)*cos(y) must not have i symmetry: n={}, m={}'.
                     format(n, m))

    for n in [10, 11]:
        x = np.linspace(-np.pi, np.pi, n)
        f1 = np.cos(x)
        assert_equal(is_symmetric(f1, i_sym=True, j_sym=False), True,
                     'cos(x) symmetry in 1D: n={}, m={}'.format(n, m))


def test_absolute_ratio_benchmark():
    """ Mostly sanity check that Analytical functions don't have typos
        or syntax errors
    """
    n = 501
    r_max = 50

    for symmetric in [True, False]:
        for Backend, options in [(StepAnalytical,
                                  dict(A0=10.0, r1=6.0, r2=14.0,
                                       ratio_valid_step=1.0)),
                                 (GaussianAnalytical, dict(sigma=2))]:

            ref = Backend(n, r_max, symmetric=symmetric, **options)
            ratio = absolute_ratio_benchmark(ref, ref.func)

            backend_name = type(ref).__name__

            assert_allclose(ratio.mean(), 1.0,
                            err_msg='Sanity test, sym={}: {}: ratio == 1.0'.
                                    format(symmetric, backend_name))
            assert_allclose(ratio.std(), 0.0,
                            err_msg='Sanity test, sym={}: {}: std == 0.0'.
                                    format(symmetric, backend_name))


if __name__ == '__main__':
    test_AbelTiming()
    test_DistributionsTiming()
    test_symmetry_function()
    test_absolute_ratio_benchmark()
