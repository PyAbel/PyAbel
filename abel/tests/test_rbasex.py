from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from abel.rbasex import rbasex_transform, cache_cleanup
from abel.rbasex import get_bs_cached, cache_cleanup
from abel.tools.analytical import GaussianAnalytical
from abel.hansenlaw import hansenlaw_transform
from abel import Transform


DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')


def test_rbasex_shape():
    rmax = 11
    n = 2 * rmax - 1
    im = np.ones((n, n), dtype='float')

    fwd_im, fwd_distr = rbasex_transform(im, direction='forward')
    assert fwd_im.shape == (n, n)
    assert fwd_distr.r.shape == (rmax,)

    inv_im, inv_distr = rbasex_transform(im)
    assert inv_im.shape == (n, n)
    assert inv_distr.r.shape == (rmax,)


def test_rbasex_zeros():
    rmax = 11
    n = 2 * rmax - 1
    im = np.zeros((n, n), dtype='float')

    fwd_im, fwd_distr = rbasex_transform(im, direction='forward')
    assert fwd_im.shape == (n, n)
    assert fwd_distr.r.shape == (rmax,)

    inv_im, inv_distr = rbasex_transform(im)
    assert inv_im.shape == (n, n)
    assert inv_distr.r.shape == (rmax,)


def test_rbasex_gaussian():
    """Check an isotropic gaussian solution for rBasex"""
    rmax = 100
    sigma = 30
    n = 2 * rmax - 1

    ref = GaussianAnalytical(n, rmax, symmetric=True, sigma=sigma)
    # images as direct products
    src = ref.func * ref.func[:, None]
    proj = ref.abel * ref.func[:, None]  # (vertical is not Abel-transformed)

    fwd_im, fwd_distr = rbasex_transform(src, direction='forward')
    # whole image
    assert_allclose(fwd_im, proj, rtol=0.02, atol=0.001)
    # radial intensity profile (without r = 0)
    assert_allclose(fwd_distr.harmonics()[0, 1:], ref.abel[rmax:],
                    rtol=0.02, atol=5e-4)

    inv_im, inv_distr = rbasex_transform(proj)
    # whole image
    assert_allclose(inv_im, src, rtol=0.02, atol=0.02)
    # radial intensity profile (without r = 0)
    assert_allclose(inv_distr.harmonics()[0, 1:], ref.func[rmax:],
                    rtol=0.02, atol=1e-4)


def run_orders(odd=False):
    """
    Test angular orders using Gaussian peaks by comparison with hansenlaw.
    """
    maxorder = 6
    sigma = 5.0  # Gaussian sigma
    step = 6 * sigma  # distance between peak centers

    for order in range(maxorder + 1):
        rmax = int((order + 2) * step)
        if odd:
            if order == 0:
                continue  # 0th order cannot be odd
            height = 2 * rmax + 1
        else:  # even only
            if order % 2:
                continue  # skip odd
            height = rmax + 1
        # coordinates (Q0 or right half):
        x = np.arange(float(rmax + 1))
        y = rmax - np.arange(float(height))[:, None]
        # radius
        r = np.sqrt(x**2 + y**2)
        # cos, sin
        r[rmax, 0] = np.inf
        c = y / r
        s = x / r
        r[rmax, 0] = 0

        # Gaussian peak with one cossin angular term
        def peak(i):
            m = i  # cos power
            k = (order - m) & ~1  # sin power (round down to even)
            return c ** m * s ** k * \
                   np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))

        # create source distribution
        src = peak(0)
        for i in range(1, order + 1):
            if not odd and i % 2:
                continue  # skip odd
            src += peak(i)

        # reference forward transform
        abel = hansenlaw_transform(src, direction='forward', hold_order=1)

        param = ', order = {}, odd = {}, '.format(order, odd)

        # test forward transform
        for mode in ['clean', 'cached']:
            if mode == 'clean':
                cache_cleanup()
            proj, _ = rbasex_transform(src, origin=(rmax, 0),
                                       order=order, odd=odd,
                                       direction='forward', out='fold')
            assert_allclose(proj, abel, rtol=0.003, atol=0.4,
                            err_msg='-> forward' + param + mode)

        # test inverse transforms
        for reg in [None, ('L2', 1), ('diff', 1), ('SVD', 1 / rmax)]:
            for mode in ['clean', 'cached']:
                if mode == 'clean':
                    cache_cleanup()
                recon, _ = rbasex_transform(abel, origin=(rmax, 0),
                                            order=order, odd=odd,
                                            reg=reg, out='fold')
                recon[rmax-2:rmax+3, :2] = 0  # exclude pixels near center
                assert_allclose(recon, src, atol=0.03,
                                err_msg='-> reg = ' + str(reg) + param + mode)


def test_rbasex_orders():
    run_orders()


def test_rbasex_orders_odd():
    run_orders(odd=True)


def test_rbasex_pos():
    """
    Test positive regularization as in run_orders().
    """
    sigma = 5.0  # Gaussian sigma
    step = 6 * sigma  # distance between peak centers

    for order in [0, 1, 2, 4, 6]:
        rmax = int((order + 2) * step)
        if order == 1:  # odd
            rmax += int(step)  # 3 peaks instead of 2
            height = 2 * rmax + 1
        else:  # even only
            height = rmax + 1
        # coordinates (Q0 or right half):
        x = np.arange(float(rmax + 1))
        y = rmax - np.arange(float(height))[:, None]
        # radius
        r = np.sqrt(x**2 + y**2)
        # cos, sin
        r[rmax, 0] = np.inf
        c = y / r
        s = x / r
        r[rmax, 0] = 0

        # Gaussian peak with one cossin angular term
        def peak(i, isotropic=False):
            if isotropic:
                return np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))
            m = i  # cos power
            k = (order - m) & ~1  # sin power (round down to even)
            return c ** m * s ** k * \
                   np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))

        # create source distribution
        src = peak(0)
        if order == 1:  # special case
            src += (1 + c) * peak(1, True)  # 1 + cos >= 0
            src += (1 - c) * peak(2, True)  # 1 - cos >= 0
        else:  # other even orders
            for i in range(2, order + 1, 2):
                src += peak(i)
        # array for nonnegativity test (with some tolerance)
        zero = np.full_like(src, -1e-15)

        # reference forward transform
        abel = hansenlaw_transform(src, direction='forward', hold_order=1)
        # with some noise
        abel += 0.05 * np.random.RandomState(0).rand(*abel.shape)

        param = '-> order = {}, '.format(order)

        # test inverse transform
        for mode in ['clean', 'cached']:
            if mode == 'clean':
                cache_cleanup()
            recon, _ = rbasex_transform(abel, origin=(rmax, 0),
                                        order=order,
                                        reg='pos', out='fold')
            recon[rmax-3:rmax+4, :2] = 0  # exclude pixels near center
            assert_allclose(recon, src, atol=0.05,
                            err_msg=param + mode)
            # check nonnegativity
            assert_array_less(zero, recon)


def run_out(odd=False):
    """
    Test some output shapes.
    """
    sigma = 5.0  # Gaussian sigma
    step = int(6 * sigma)  # distance between peak centers

    rmax = (2 + 2) * step
    size = 2 * rmax + 1
    # coordinates (full image):
    x = np.arange(float(size)) - rmax
    y = rmax - np.arange(float(size))[:, None]
    # radius
    r = np.sqrt(x**2 + y**2)
    # cos
    r[rmax, rmax] = np.inf
    c = y / r
    r[rmax, rmax] = 0

    # Gaussian peak with one cos^n angular term
    def peak(i):
        n = i if odd else 2 * i
        return c ** n * \
               np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))

    # create source distribution
    src = peak(0) + peak(1)

    # reference forward transform
    abel = Transform(src, direction='forward', method='hansenlaw',
                     transform_options={'hold_order': 1}).transform

    param = '-> odd = {}, '.format(odd)

    # Test forward transform:

    # rmax < MIN => larger 'same'
    proj, _ = rbasex_transform(src, rmax=rmax - step, odd=odd,
                               direction='forward', out='same')
    assert_allclose(proj, abel, rtol=0.005, atol=0.2,
                    err_msg=param + 'rmax < MIN, out = same')

    # cropped, rmax = HOR < VER -> same
    crop = (slice(3 * step, -step // 2),
            slice(3 * step, -step))
    proj, _ = rbasex_transform(src[crop], origin=(step, step),
                               odd=odd, direction='forward', out='same')
    assert_allclose(proj, abel[crop], rtol=0.005, atol=0.2,
                    err_msg=param + 'rmax = HOR < VER, out = same')

    # cropped, rmax = VER < HOR -> same
    crop = (slice(step, -3 * step),
            slice(3 * step, -step // 2))
    proj, _ = rbasex_transform(src[crop], origin=(3 * step, step),
                               odd=odd, direction='forward', out='same')
    assert_allclose(proj, abel[crop], rtol=0.0003, atol=0.3,
                    err_msg=param + 'rmax = VER < HOR, out = same')

    # cropped, rmax = VER > HOR -> same
    crop = (slice(3 * step, -step // 2),
            slice(3 * step, -step))
    proj, _ = rbasex_transform(src[crop], origin=(step, step), rmax='MAX',
                               odd=odd, direction='forward', out='same')
    assert_allclose(proj, abel[crop], rtol=0.005, atol=0.2,
                    err_msg=param + 'rmax = VER > HOR, out = same')

    # cropped, rmax = HOR > VER -> same
    crop = (slice(step, -3 * step),
            slice(3 * step, -step // 2))
    proj, _ = rbasex_transform(src[crop], origin=(3 * step, step), rmax='MAX',
                               odd=odd, direction='forward', out='same')
    assert_allclose(proj, abel[crop], rtol=0.0003, atol=0.3,
                    err_msg=param + 'rmax = HOR > VER, out = same')

    # cropped, rmax = rmax -> full
    crop = (slice(3 * step, -step),
            slice(3 * step, -step))
    # (in multiples of step: VER^2 + HOR^2 = 2 * 3^2 > rmax^2 = 4^2)
    proj, _ = rbasex_transform(src[crop], origin=(step, step), rmax=rmax,
                               odd=odd, direction='forward', out='full')
    assert_allclose(proj, abel, rtol=0.002, atol=0.3,
                    err_msg=param + 'rmax = rmax, out = full')


def test_rbasex_out():
    run_out()


def test_rbasex_out_odd():
    run_out(odd=True)


def get_basis_file_name(rmax, order, odd, inv):
    return os.path.join(DATA_DIR,
                        'rbasex_basis_{}_{}{}{}.npy'.format(rmax, order,
                                                            'o' if odd else '',
                                                            'i' if inv else ''))


def test_rbasex_bs_cache():
    rmax, order, odd, inv = 50, 2, False, True
    file_name = get_basis_file_name(rmax, order, odd, inv)
    if os.path.exists(file_name):
        os.remove(file_name)
    # 1st call generate and save
    bs1 = get_bs_cached(rmax, order, odd, basis_dir=DATA_DIR, verbose=True)
    assert os.path.exists(file_name), 'Basis set was not saved!'
    # 2nd call load from file
    bs2 = get_bs_cached(rmax, order, odd, basis_dir=DATA_DIR, verbose=True)
    assert_allclose(bs1, bs2, err_msg='Loaded basis set differs from saved!')
    if os.path.exists(file_name):
        os.remove(file_name)


def rbasex_bs_resize(old, new, new_saved=False):
    """
    Compare basis set loaded and modified to cleanly computed.
    old, new: basis parameters (rmax, order, odd, inv)
    new_saved: check that new basis is (True) or is not (False) saved
    """
    file_name_old = get_basis_file_name(*old)
    file_name_new = get_basis_file_name(*new)

    # (remove both basis files)
    def remove_files():
        if os.path.exists(file_name_old):
            os.remove(file_name_old)
        if os.path.exists(file_name_new):
            os.remove(file_name_new)

    # (convert basis-set parameters to get_bs_cached() arguments)
    def arg(prm):
        return prm[:-1] + ('inverse' if prm[3] else 'forward',)

    # make sure that basis files do not exist and are not cached
    remove_files()
    cache_cleanup()
    # generate "old" basis and save
    get_bs_cached(*arg(old), basis_dir=DATA_DIR, verbose=False)
    cache_cleanup()
    # generate "new" basis without saving
    bs_new = get_bs_cached(*arg(new), basis_dir=None, verbose=False)
    cache_cleanup()

    # load "new" basis from "old" file, cropped
    bs = get_bs_cached(*arg(new), basis_dir=DATA_DIR, verbose=True)
    # check that "new" file is (not) be saved
    if new_saved:
        assert os.path.exists(file_name_new), "New basis set was not saved!"
    else:
        assert not os.path.exists(file_name_new), "New basis set was saved!"
    # clean-up all caches
    cache_cleanup()
    remove_files()

    # compare cropped to clean
    assert_allclose(bs, bs_new)


def test_rbasex_bs_crop_rmax():
    rbasex_bs_resize((60, 2, False, True),
                     (50, 2, False, True))


def test_rbasex_bs_crop_order():
    rbasex_bs_resize((50, 2, False, True),
                     (50, 0, False, True))


def test_rbasex_bs_crop_odd():
    rbasex_bs_resize((50, 2, True, True),
                     (50, 2, False, True))


def test_rbasex_bs_crop_order_odd():
    rbasex_bs_resize((50, 3, True, True),
                     (50, 2, False, True))


def test_rbasex_bs_crop_inv():
    rbasex_bs_resize((50, 2, False, True),
                     (50, 2, False, False))


def test_rbasex_bs_add_inv():
    rbasex_bs_resize((50, 2, False, False),
                     (50, 2, False, True),
                     new_saved=True)


if __name__ == '__main__':
    test_rbasex_shape()
    test_rbasex_zeros()
    test_rbasex_gaussian()
    test_rbasex_orders()
    test_rbasex_orders_odd()
    test_rbasex_pos()
    test_rbasex_out()
    test_rbasex_out_odd()
    test_rbasex_bs_cache()
    test_rbasex_bs_crop_rmax()
    test_rbasex_bs_crop_order()
    test_rbasex_bs_crop_odd()
    test_rbasex_bs_crop_order_odd()
    test_rbasex_bs_crop_inv()
    test_rbasex_bs_add_inv()
