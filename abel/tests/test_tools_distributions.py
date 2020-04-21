from __future__ import division

import numpy as np
from numpy.testing import assert_equal, assert_allclose
import itertools

from abel.tools.vmi import Distributions, harmonics


def test_origin():
    """
    Test symbolic origin wrt corresponding numeric form.
    """
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
    """
    Test cossin distributions for even orders using Gaussian peaks.

    tol = list of tolerances for order=0, order=2, order=4, ...
    """
    sigma = 5.0  # peak SD
    step = 6 * sigma  # distance between peak centers

    for n in range(len(tol)):  # order = 2n
        size = int((n + 2) * step)
        # squared coordinates:
        x2 = np.arange(float(size))**2
        y2 = x2[:, None]
        r2 = x2 + y2
        # cos^2, sin^2
        c2 = np.divide(y2, r2, out=np.zeros_like(r2), where=r2 != 0)
        s2 = 1 - c2
        # radius
        r = np.sqrt(r2)

        # Gaussian peak with one cossin angular term.
        def peak(i):
            return c2 ** i * s2 ** (n - i) * \
                   np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))

        # quadrant with all peaks
        Q = peak(0)
        for i in range(1, n + 1):
            Q += peak(i)

        for rmax in ['MIN', 'all']:
            param = '-> rmax = {}, order={}, method = {}'.\
                    format(rmax, 2 * n, method)
            res = Distributions('ul', rmax, 2 * n, method=method).image(Q)
            cossin = res.cossin()
            # extract values at peak centers
            cs = np.array([cossin[:, int((i + 1) * step)]
                           for i in range(n + 1)])
            assert_allclose(cs, np.identity(n + 1), atol=tol[n],
                            err_msg=param)
            # test that Ibeta, and thus harmonics, work in principle
            res.Ibeta()


def test_order_nearest():
    run_order('nearest', [0.0018, 0.0018, 0.0022, 0.0062, 0.020])


def test_order_linear():
    run_order('linear', [0.0031, 0.0034, 0.0034, 0.0067, 0.029])


def test_order_remap():
    run_order('remap', [5e-6, 6e-6, 6e-6, 2e-5, 7e-5])


def run_order_odd(method, tol):
    """
    Test cossin distributions including odd orders using Gaussian peaks.

    tol = list of tolerances for order=0, order=1, order=2, ...
    """
    sigma = 5.0  # peak SD
    step = 6 * sigma  # distance between peak centers

    for n in range(0, len(tol)):  # order = n
        size = int((n + 2) * step)
        # coordinates:
        x = np.arange(float(size))
        y = size - np.arange(2 * float(size) + 1)[:, None]
        # radius
        r = np.sqrt(x**2 + y**2)
        # cos, sin
        r[size, 0] = np.inf
        c = y / r
        s = x / r
        r[size, 0] = 0

        # Gaussian peak with one cossin angular term
        def peak(i):
            m = i  # cos power
            k = (n - m) & ~1  # sin power (round down to even)
            return c ** m * s ** k * \
                   np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))

        # quadrant with all peaks
        Q = peak(0)
        for i in range(1, n + 1):
            Q += peak(i)

        for rmax in ['MIN', 'all']:
            param = '-> rmax = {}, order={}, method = {}'.\
                    format(rmax, n, method)
            res = Distributions((int(size), 0), rmax, n, odd=True,
                                method=method).image(Q)
            cossin = res.cossin()
            # extract values at peak centers
            cs = np.array([cossin[:, int((i + 1) * step)]
                           for i in range(n + 1)])
            assert_allclose(cs, np.identity(n + 1), atol=tol[n],
                            err_msg=param)
            # test that Ibeta, and thus harmonics, work in principle
            res.Ibeta()


def test_order_odd_nearest():
    run_order_odd('nearest', [0.0018, 0.0018, 0.0018, 0.0018, 0.0020])


def test_order_odd_linear():
    run_order_odd('linear', [0.0032, 0.0034, 0.0034, 0.0034, 0.0035])


def test_order_odd_remap():
    run_order_odd('remap', [5e-6, 5e-6, 6e-6, 6e-6, 6e-6])


def run_method(method, rmax, tolP0, tolP2, tolI, tolbeta, weq=True):
    """
    Test harmonics and Ibeta for various combinations of origins and weights
    for default order=2, odd=False.

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

    # Gaussian peak.
    def peak(i, r):
        return np.exp(-(r - i * step)**2 / (2 * sigma**2))

    # Test image.
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

    # Reference distribution for test image.
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

        # determine largest radius extracted from image
        if rmax == 'MIN':
            R = min(max(x0, n - 1 - x0), max(y0, m - 1 - y0))
        elif rmax == 'all':
            R = max([int(np.sqrt((x - x0)**2 + (y - y0)**2))
                     for x in (0, n - 1) for y in (0, m - 1)])
        step = (R - 5 * sigma) / 3  # distance between peak centers
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
        # check that results for symbolic and explicit weights match
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


def run_method_odd(method, rmax, tolP0, tolP1, tolP2,
                   tolI, tolbeta1, tolbeta2, weq=True):
    """
    Test harmonics and Ibeta for various combinations of origins and weights
    for default order=2, but with odd=True.

    method = method name
    rmax = 'MIN' or 'all'
    tol... = (atol, rmstol) for ...
    tolbeta = atol for beta
    weq = compare symbolic and array weights
    """
    # image size
    n = 81  # width
    m = 91  # height
    # origin coordinates
    xc = [0, 30, n // 2, 50, n - 1]
    yc = [0, 25, m // 2, 65, m - 1]
    # peak SD
    sigma = 2.0

    # Gaussian peak.
    def peak(i, r):
        return np.exp(-(r - i * step)**2 / (2 * sigma**2))

    # Test image.
    def image():
        # coordinates:
        x = np.arange(float(n)) - x0
        y = y0 - np.arange(float(m))[:, None]
        # radius:
        r = np.sqrt(x**2 + y**2)
        # cos, |sin|
        r[y0, x0] = np.inf
        c = y / r
        s = np.abs(x) / r
        s[y0, x0] = 1
        r[y0, x0] = 0
        # image: 4 peaks with different anisotropies
        IM = s**2      * peak(1, r) + \
             c**2      * peak(2, r) + \
             (1/2 + c) * peak(3, r) + \
                         peak(4, r)
        return IM, s  # image, sin theta

    # Reference distribution for test image.
    def ref_distr():
        r = np.arange(R + 1)
        P0 = 2/3 * peak(1, r) + \
             1/3 * peak(2, r) + \
             1/2 * peak(3, r) + \
                   peak(4, r)
        P1 = peak(3, r)
        P2 = -2/3 * peak(1, r) + \
              2/3 * peak(2, r)
        I = 4 * np.pi * r**2 * P0
        beta1 = P1 / P0
        beta2 = P2 / P0
        return r, P0, P1, P2, I, beta1, beta2

    for y0, x0 in itertools.product(yc, xc):
        param = ' @ y0 = {}, x0 = {}, rmax = {}, method = {}'.\
                format(y0, x0, rmax, method)

        # determine largest radius extracted from image
        if rmax == 'MIN':
            R = min(max(x0, n - 1 - x0), max(y0, m - 1 - y0))
        elif rmax == 'MAX':
            # exclude situations when the outer ring has insufficient vertical
            # span for reliable even/odd separation
            if y0 in [0, m - 1] and x0 not in [0, n - 1] or \
               x0 == n // 2 and abs(y0 - m // 2) not in [0, m // 2]:
                continue
            R = max(max(x0, n - 1 - x0), max(y0, m - 1 - y0))
        step = (R - 5 * sigma) / 4  # distance between peak centers
        refr, refP0, refP1, refP2, refI, refbeta1, refbeta2 = ref_distr()
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
        P0, P1, P2, r, I, beta1, beta2 = {}, {}, {}, {}, {}, {}, {}
        for use_sin, wname, warray in weights:
            weight_param = param + \
                           ', sin = {}, weights = {}'.format(use_sin, wname)
            key = (use_sin, wname)

            distr = Distributions((y0, x0), rmax, odd=True, use_sin=use_sin,
                                  weights=warray, method=method)
            res = distr(IM)
            P0[key], P1[key], P2[key] = res.harmonics()
            r[key], I[key], beta1[key], beta2[key] = res.rIbeta()

            def assert_cmp(msg, a, ref, tol):
                atol, rmstol = tol
                assert_allclose(a, ref, atol=atol,
                                err_msg=msg + weight_param)
                rms = np.sqrt(np.mean((a - ref)**2))
                assert rms < rmstol, \
                       '\n' + msg + weight_param + \
                       '\nRMS error = {} > {}'.format(rms, rmstol)

            assert_cmp('-> P0', P0[key], refP0, tolP0)
            assert_cmp('-> P1', P1[key], refP1, tolP1)
            assert_cmp('-> P2', P2[key], refP2, tolP2)

            assert_equal(r[key], refr, err_msg='-> r' + weight_param)
            assert_cmp('-> I', f * I[key], f * refI, tolI)
            # beta values at peak centers
            b1 = [round(beta1[key][int(i * step)], 5) for i in (1, 2, 3, 4)]
            assert_allclose(b1, [0, 0, 2, 0], atol=tolbeta1,
                            err_msg='-> beta1' + weight_param)
            b2 = [round(beta2[key][int(i * step)], 5) for i in (1, 2, 3, 4)]
            assert_allclose(b2, [-1, 2, 0, 0], atol=tolbeta2,
                            err_msg='-> beta2' + weight_param)

            assert_equal(IM, IMcopy,
                         err_msg='-> IM corrupted' + weight_param)
            assert_equal(w1, w1copy,
                         err_msg='-> weights corrupted' + weight_param)
            assert_equal(ws, wscopy,
                         err_msg='-> weights corrupted' + weight_param)

        if not weq:
            continue
        # check that results for symbolic and explicit weights match
        for key1, key2 in [((False, '1'), (False, None)),
                           ((False, 'sin'), (True, None)),
                           ((True, '1'), (False, 'sin'))]:
            pair_param = param + ', sin + weights {} != {}'.format(key1, key2)
            assert_allclose(P0[key1], P0[key2],
                            err_msg='-> P0' + pair_param)
            assert_allclose(P1[key1], P1[key2],
                            err_msg='-> P1' + pair_param)
            assert_allclose(P2[key1], P2[key2],
                            err_msg='-> P2' + pair_param)
            assert_allclose(I[key1], I[key2],
                            err_msg='-> I' + pair_param)
            assert_allclose(beta1[key1], beta1[key2],
                            err_msg='-> beta1' + pair_param)
            assert_allclose(beta2[key1], beta2[key2],
                            err_msg='-> beta2' + pair_param)


def test_nearest_odd():
    run_method_odd('nearest', 'MIN',
                   (0.093, 0.030), (0.21, 0.063), (0.17, 0.042),
                   (0.093, 0.030), 0.18, 0.075)
    run_method_odd('nearest', 'MAX',
                   (0.16, 0.029), (0.27, 0.049), (0.12, 0.022),
                   (0.16, 0.029), 0.080, 0.030)


def test_linear_odd():
    run_method_odd('linear', 'MIN',
                   (0.025, 0.0078), (0.072, 0.028), (0.054, 0.019),
                   (0.025, 0.0078), 0.17, 0.11)
    run_method_odd('linear', 'MAX',
                   (0.20, 0.040), (0.33, 0.064), (0.14, 0.027),
                   (0.20, 0.040), 0.11, 0.086)


def test_remap_odd():
    run_method_odd('remap', 'MIN',
                   (0.016, 0.0033), (0.029, 0.0060), (0.015, 0.0030),
                   (0.016, 0.0033), 0.050, 0.012, weq=False)
    run_method_odd('remap', 'MAX',
                   (0.13, 0.025), (0.21, 0.041), (0.087, 0.017),
                   (0.13, 0.025), 0.021, 0.011, weq=False)
    # (resampling of weights array in 'remap' makes "weq" differ)


def run_random(method, odd, use_sin, parts=True):
    """
    Test quadrant flipping and image folding using random image data.

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
            ho = harmonics(IM, (y, x), 'all', odd=odd,
                           use_sin=use_sin, weights=weights,
                           method=method)
            hf = harmonics(IM[::ydir, ::xdir], (0, 0), 'all', odd=odd,
                           use_sin=use_sin, weights=weights[::ydir, ::xdir],
                           method=method)
            if odd:
                ho = ho[:, :-3]
                hf = hf[:, :-3]
                if ydir == -1:
                    hf[1] = -hf[1]
                cmp = assert_allclose
            else:
                cmp = assert_equal
            cmp(ho, hf,
                err_msg='-> flip({}, {}) @ method = {}, odd = {}, sin = {}'.
                        format(ydir, xdir, method, odd, use_sin))

    # parts
    if not parts:
        return
    for y0, x0 in itertools.product([15, m // 2, 25], [10, n // 2, 40]):
        param = ' @ y0 = {}, x0 = {}, method = {}, odd = {}, sin = {}'.\
                format(y0, x0, method, odd, use_sin)

        # trim border
        trim = (slice(1, -1), slice(1, -1))
        IMtrim = IM[trim]
        wtrim = weights[trim]
        ht = harmonics(IMtrim, (y0 - 1, x0 - 1), 'all', odd=odd,
                       use_sin=use_sin, weights=wtrim, method=method)

        wmask = np.zeros_like(IM)
        wmask[trim] = weights[trim]
        hm = harmonics(IM, (y0, x0), 'all', odd=odd, use_sin=use_sin,
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
            hc = harmonics(Q, origin, 'all', odd=odd, use_sin=use_sin,
                           weights=Qw, method=method)

            wmask = np.zeros_like(IM)
            wmask[region] = weights[region]
            hm = harmonics(IM, (y0, x0), 'all', odd=odd, use_sin=use_sin,
                           weights=wmask, method=method)

            assert_allclose(hc, hm[:, :hc.shape[1]],
                            err_msg='-> Q (origin = ' + origin + ')' + param)


def test_nearest_random():
    for odd in [False, True]:
        for use_sin in [False, True]:
            run_random('nearest', odd, use_sin)


def test_linear_random():
    for odd in [False, True]:
        for use_sin in [False, True]:
            run_random('linear', odd, use_sin)


def test_remap_random():
    for use_sin in [False, True]:
        run_random('remap', False, use_sin, parts=False)
    # (interpolation and different sampling in 'remap' make
    #  odd vertical flip and "parts" differ)


if __name__ == '__main__':
    test_origin()

    test_order_nearest()
    test_order_odd_nearest()
    test_nearest()
    test_nearest_odd()
    test_nearest_random()

    test_order_linear()
    test_order_odd_linear()
    test_linear()
    test_linear_odd()
    test_linear_random()

    test_order_remap()
    test_order_odd_remap()
    test_remap()
    test_remap_odd()
    test_remap_random()
