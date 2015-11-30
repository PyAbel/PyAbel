from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from numpy.testing import assert_allclose, assert_equal

from abel.analytical import StepAnalytical, GaussianAnalytical
from abel.benchmark import absolute_ratio_benchmark, is_symmetric


def assert_allclose_msg(x, y, message):
    assert np.allclose(x, y), message


def test_symmetry_function():
    # Check the consistency of abel.benchmark.is_symmetric

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
        yield assert_equal, is_symmetric(f1, i_sym=True, j_sym=True), True,\
              'Checking f1 function for polar symmetry n={},m={}'.format(n,m)

        f2 = np.cos(XX)*np.sin(YY)
        yield assert_equal, is_symmetric(f2, i_sym=True, j_sym=False), True,\
              'Checking f2 function for i symmetry n={},m={}'.format(n,m)

        f3 = np.sin(XX)*np.cos(YY)
        yield assert_equal, is_symmetric(f3, i_sym=False, j_sym=True), True, \
              'Checking f3 function for j symmetry n={},m={}'.format(n,m)

        yield assert_equal, is_symmetric(f3, i_sym=True, j_sym=False), False, \
              'Checking that f3 function does not have i symmetry n={},m={}'.format(n,m)


def test_absolute_ratio_benchmark():
    # Mostly sanity to check that Analytical functions don't have typos
    # or syntax errors
    n = 501
    r_max = 50

    for symmetric in [True, False]:
        for Backend, options in [(StepAnalytical, dict(A0=10.0, r1=6.0,
                                            r2=14.0, ratio_valid_step=1.0)),
                                 (GaussianAnalytical, dict(sigma=2))]:

            ref = Backend(n, r_max, symmetric=symmetric, **options)
            ratio_mean, ratio_std, _ = absolute_ratio_benchmark(ref, ref.func)

            backend_name = type(ref).__name__


            yield assert_allclose_msg, ratio_mean, 1.0,\
                    "Sanity test, sym={}: {}: ratio == 1.0".format(symmetric, backend_name)
            yield assert_allclose_msg, ratio_std, 0.0,\
                    "Sanity test, sym={}: {}: std == 0.0".format(symmetric, backend_name)
