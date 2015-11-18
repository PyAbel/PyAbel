from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from numpy.testing import assert_allclose

from abel.benchmark import SymStepBenchmark


def test_sym_step_benchmark():
    """
    Check that SymStepBenchmark passes if reconstuction = original signal
    """
    n = 501
    r_max = 50
    A0 = 10.0
    r1 = 6.0
    r2 = 14.0

    sbench = SymStepBenchmark(n, r_max, r1, r2, A0)

    err_mean, err_std, _ = sbench.run(sbench.step.func, 1.0)


    yield assert_allclose, err_mean, 1.0  # signal is identical
    yield assert_allclose, err_std, 0.0   # the standard deviation is 0.0
