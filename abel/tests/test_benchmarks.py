from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from numpy.testing import assert_allclose

from abel.analytical import StepAnalytical, GaussianAnalytical
from abel.benchmark import absolute_ratio_benchmark


def assert_equal(x, y, message):
    assert np.allclose(x, y), message


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


            yield assert_equal, ratio_mean, 1.0, "Sanity test, sym={}: {}: ratio == 1.0".format(symmetric, backend_name)
            yield assert_equal, ratio_std, 0.0,   "Sanity test, sym={}: {}: std == 0.0".format(symmetric, backend_name)
