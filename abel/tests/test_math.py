import numpy as np
from numpy.testing import assert_allclose
from abel.tools.math import gradient


def test_gradient():
    x = np.random.randn(10,10)

    dx = np.gradient(x)[1]
    dx2 = gradient(x)

    assert_allclose(dx, dx2)

