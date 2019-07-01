from __future__ import division

import numpy as np
from numpy.testing import assert_equal, assert_allclose
import itertools

from abel.tools.vmi import Distributions, harmonics


def test_origin():
    # image size
    n = 11  # width
    m = 20  # height
    # origin (symbolic)
    row = 'utclb'
    col = 'lcr'
    # origin (numeric)
    xc = [0, n // 2, n - 1]
    yc = [0, 0, m // 2, m - 1, m - 1]

    np.random.seed(0)
    IM = np.random.random((m, n))

    for r, y in zip(row, yc):
        for c, x in zip(col, xc):
            assert_equal(harmonics(IM, r + c), harmonics(IM, (y, x)),
                         err_msg='-> origin "{}" != {}'.format(r + c, (y, x)))


def run_order(method, tol):
    sigma = 5.0
    step = 6 * sigma

    for n in range(len(tol)):  # order = 2n
        size = (n + 2) * step
        # squared coordinates:
        x2 = np.arange(float(size))**2
        y2 = x2[:, None]
        r2 = x2 + y2
        # cos^2, sin^2
        c2 = np.divide(y2, r2, out=np.zeros_like(r2), where=r2 != 0)
        s2 = 1 - c2
        # radius
        r = np.sqrt(r2)

        def peak(i):
            return c2 ** (n - i) * s2 ** i * \
                   np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))

        Q = peak(0)
        for i in range(1, n + 1):
            Q += peak(i)

        for rmax in ['MIN', 'all']:
            param = '-> rmax = {}, order={}, method = {}'.\
                    format(rmax, 2 * n, method)
            res = Distributions('ul', rmax, 2 * n, method=method).image(Q)
            cossin = res.cossin()
            cs = np.array([cossin[:, int((i + 1) * step)]
                           for i in range(n + 1)])
            assert_allclose(cs, np.identity(n + 1), atol=tol[n],
                            err_msg=param)
            # test that Ibeta, and thus harmonics, work in principle
            res.Ibeta()


def test_order_nearest():
    run_order('nearest', [0.0018, 0.0020, 0.0019, 0.0022, 0.0055])


def test_order_linear():
    run_order('linear', [0.0031, 0.0033, 0.0034, 0.0034, 0.011])


def test_order_remap():
    run_order('remap', [5e-6, 6e-6, 7e-6, 8e-6, 2e-5])


def run_method(method, rmax, tolP0, tolP2, tolI, tolbeta, weq=True):
    """
    method = method name
    rmax = 'MIN' or 'all'
    tol... = (atol, rmstol) for ...
    tolbeta = atol for beta
    weq = compare symbolic and array weights
    """
    # image size
    n = 81  # width
    m = 71  # height
    # origin coordinates
    xc = [0, 20, n // 2, 60, n - 1]
    yc = [0, 25, m // 2, 45, m - 1]
    # peak SD
    sigma = 2.0

    def peak(i, r):
        return np.exp(-(r - i * step)**2 / (2 * sigma**2))

    def image():
        # squared coordinates:
        x2 = (np.arange(float(n)) - x0)**2
        y2 = (np.arange(float(m))[:, None] - y0)**2
        r2 = x2 + y2
        # radius:
        r = np.sqrt(r2)
        # cos^2, sin^2:
        c2 = np.divide(y2, r2, out=np.zeros_like(r2), where=r2 != 0)
        s2 = 1 - c2
        # image: 3 peaks with different anisotropies
        IM = s2 * peak(1, r) + \
                  peak(2, r) + \
             c2 * peak(3, r)
        return IM, np.sqrt(s2)  # image, sin theta

    def ref_distr():
        r = np.arange(R + 1)
        P0 = 2/3 * peak(1, r) + \
                   peak(2, r) + \
             1/3 * peak(3, r)
        P2 = -2/3 * peak(1, r) + \
              2/3 * peak(3, r)
        I = 4 * np.pi * r**2 * P0
        beta = P2 / P0
        return r, P0, P2, I, beta

    for y0, x0 in itertools.product(yc, xc):
        param = ' @ y0 = {}, x0 = {}, rmax = {}, method = {}'.\
                format(y0, x0, rmax, method)

        if rmax == 'MIN':
            R = min(max(x0, n - 1 - x0), max(y0, m - 1 - y0))
        elif rmax == 'all':
            R = max([int(np.sqrt((x - x0)**2 + (y - y0)**2))
                     for x in (0, n - 1) for y in (0, m - 1)])
        step = (R - 5 * sigma) / 3
        refr, refP0, refP2, refI, refbeta = ref_distr()
        f = 1 / (4 * np.pi * (1 + refr**2))  # rescaling factor for I

        IM, ws = image()
        w1 = np.ones_like(IM)

        IMcopy = IM.copy()
        w1copy = w1.copy()
        wscopy = ws.copy()

        weights = [(False, None, None),
                   (True, None, None),
                   (False, '1', w1),
                   (False, 'sin', ws),
                   (True, '1', w1)]
        P0, P2, r, I, beta = {}, {}, {}, {}, {}
        for use_sin, wname, warray in weights:
            weight_param = param + \
                           ', sin = {}, weights = {}'.format(use_sin, wname)
            key = (use_sin, wname)

            distr = Distributions((y0, x0), rmax, use_sin=use_sin,
                                  weights=warray, method=method)
            res = distr(IM)
            P0[key], P2[key] = res.harmonics()
            r[key], I[key], beta[key] = res.rIbeta()

            def assert_cmp(msg, a, ref, tol):
                atol, rmstol = tol
                assert_allclose(a, ref, atol=atol,
                                err_msg=msg + weight_param)
                rms = np.sqrt(np.mean((a - ref)**2))
                assert rms < rmstol, \
                       '\n' + msg + weight_param + \
                       '\nRMS error = {} > {}'.format(rms, rmstol)

            assert_cmp('-> P0', P0[key], refP0, tolP0)
            assert_cmp('-> P2', P2[key], refP2, tolP2)

            assert_equal(r[key], refr, err_msg='-> r' + weight_param)
            assert_cmp('-> I', f * I[key], f * refI, tolI)
            # beta values at peak centers
            b = [round(beta[key][int(i * step)], 5) for i in (1, 2, 3)]
            assert_allclose(b, [-1, 0, 2], atol=tolbeta,
                            err_msg='-> beta' + weight_param)

            assert_equal(IM, IMcopy,
                         err_msg='-> IM corrupted' + weight_param)
            assert_equal(w1, w1copy,
                         err_msg='-> weights corrupted' + weight_param)
            assert_equal(ws, wscopy,
                         err_msg='-> weights corrupted' + weight_param)

        if not weq:
            continue
        for key1, key2 in [((False, '1'), (False, None)),
                           ((False, 'sin'), (True, None)),
                           ((True, '1'), (False, 'sin'))]:
            pair_param = param + ', sin + weights {} != {}'.format(key1, key2)
            assert_allclose(P0[key1], P0[key2],
                            err_msg='-> P0' + pair_param)
            assert_allclose(P2[key1], P2[key2],
                            err_msg='-> P2' + pair_param)
            assert_allclose(I[key1], I[key2],
                            err_msg='-> I' + pair_param)
            assert_allclose(beta[key1], beta[key2],
                            err_msg='-> beta' + pair_param)


def test_nearest():
    run_method('nearest', 'MIN',
               (0.030, 0.011), (0.035, 0.012), (0.030, 0.011), 0.017)
    run_method('nearest', 'all',
               (0.047, 0.012), (0.071, 0.017), (0.047, 0.012), 0.093)


def test_linear():
    run_method('linear', 'MIN',
               (0.020, 0.0074), (0.016, 0.0049), (0.020, 0.0073), 0.0076)
    run_method('linear', 'all',
               (0.055, 0.012), (0.083, 0.017), (0.055, 0.012), 0.068)


def test_remap():
    run_method('remap', 'MIN',
               (0.0035, 0.00078), (0.0086, 0.0019), (0.0035, 0.00078), 0.0021,
               weq=False)
    run_method('remap', 'all',
               (0.027, 0.0046), (0.041, 0.0071), (0.027, 0.0046), 0.073,
               weq=False)
    # (resampling of weights array in 'remap' makes "weq" differ)


def run_random(method, use_sin, parts=True):
    """
    method = method name
    parts = test that masking by weights array equals image cropping
    """
    # image size
    n = 51  # width
    m = 41  # height

    np.random.seed(0)
    IM = np.random.random((m, n)) + 1
    weights = np.random.random((m, n)) + 1

    # quadrant flips
    for y in [0, m - 1]:
        ydir = -1 if y > 0 else 1
        for x in [0, n - 1]:
            xdir = -1 if x > 0 else 1
            ho = harmonics(IM, (y, x), 'all', use_sin=use_sin,
                           weights=weights, method=method)
            hf = harmonics(IM[::ydir, ::xdir], (0, 0), 'all', use_sin=use_sin,
                           weights=weights[::ydir, ::xdir], method=method)
            assert_equal(ho, hf,
                         err_msg='-> flip({}, {}) @ method = {}, sin = {}'.
                                 format(ydir, xdir, method, use_sin))

    # parts
    if not parts:
        return
    for y0, x0 in itertools.product([15, m // 2, 25], [10, n // 2, 40]):
        param = ' @ y0 = {}, x0 = {}, method = {}'.format(y0, x0, method)

        # trim border
        trim = (slice(1, -1), slice(1, -1))
        IMtrim = IM[trim]
        wtrim = weights[trim]
        ht = harmonics(IMtrim, (y0 - 1, x0 - 1), 'all', use_sin=use_sin,
                       weights=wtrim, method=method)

        wmask = np.zeros_like(IM)
        wmask[trim] = weights[trim]
        hm = harmonics(IM, (y0, x0), 'all', use_sin=use_sin,
                       weights=wmask, method=method)

        assert_allclose(ht, hm[:, :ht.shape[1]], err_msg='-> trim' + param)

        # cut quadrants
        regions = [
            ((slice(0, y0 + 1), slice(0, x0 + 1)), 'lr'),
            ((slice(0, y0 + 1), slice(x0, None)),  'll'),
            ((slice(y0, None),  slice(0, x0 + 1)), 'ur'),
            ((slice(y0, None),  slice(x0, None)),  'ul'),
        ]
        for region, origin in regions:
            Q = IM[region]
            Qw = weights[region]
            hc = harmonics(Q, origin, 'all', use_sin=use_sin,
                           weights=Qw, method=method)

            wmask = np.zeros_like(IM)
            wmask[region] = weights[region]
            hm = harmonics(IM, (y0, x0), 'all', use_sin=use_sin,
                           weights=wmask, method=method)

            assert_allclose(hc, hm[:, :hc.shape[1]],
                            err_msg='-> Q (origin = ' + origin + ')' + param)


def test_nearest_random():
    run_random('nearest', False)
    run_random('nearest', True)


def test_linear_random():
    run_random('linear', False)
    run_random('linear', True)


def test_remap_random():
    run_random('remap', False, parts=False)
    run_random('remap', True, parts=False)
    # (interpolation and different sampling in 'remap' makes "parts" differ)


if __name__ == '__main__':
    test_origin()

    test_order_nearest()
    test_nearest()
    test_nearest_random()

    test_order_linear()
    test_linear()
    test_linear_random()

    test_order_remap()
    test_remap()
    test_remap_random()
