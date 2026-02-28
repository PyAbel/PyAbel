from warnings import catch_warnings, simplefilter
import numpy as np
from numpy.testing import assert_allclose
from abel.tools.math import gradient, trapezoid


def test_gradient():
    x = np.random.randn(10,10)

    dx = np.gradient(x)[1]
    with catch_warnings():
        simplefilter('ignore', category=DeprecationWarning)
        dx2 = gradient(x)

    assert_allclose(dx, dx2)


def test_trapezoid():
    if hasattr(np, 'trapezoid'):  # numpy >= 2
        nptrapezoid = np.trapezoid
    else:
        nptrapezoid = np.trapz

    rnd = np.random.RandomState(0)
    for cols in range(6):  # 0 and 1 are degenerate, 2 is special
        x = np.cumsum(rnd.uniform(size=cols) + 0.1)
        for rows in [1, 10]:
            f = rnd.randn(rows, cols)
            out = trapezoid(f, x)
            ref = nptrapezoid(f, x)
            assert_allclose(out, ref, err_msg=f'-> {rows=}, {cols=}')
