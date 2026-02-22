import os.path

import numpy as np
from numpy.testing import assert_allclose

import abel
from abel.nestorolsen import get_bs_cached, cache_cleanup
from abel.tools.analytical import GaussianAnalytical


DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')


def get_basis_file_name(n):
    return os.path.join(DATA_DIR, f'nestorolsen_basis_{n}.npy')


def test_nestorolsen_basis_sets_cache():
    # n = 61  (121 full width)
    n = 61
    file_name = get_basis_file_name(n)
    if os.path.exists(file_name):
        os.remove(file_name)
    # 1st call generate and save
    get_bs_cached(n, basis_dir=DATA_DIR, verbose=False)
    # 2nd call load from file
    get_bs_cached(n, basis_dir=DATA_DIR, verbose=False)
    if os.path.exists(file_name):
        os.remove(file_name)


def test_nestorolsen_basis_sets_resize():
    """Test basis resize"""
    n_s = 50
    n_l = 100
    file_name_s = get_basis_file_name(n_s)
    file_name_l = get_basis_file_name(n_l)

    # (remove both basis files)
    def remove_files():
        if os.path.exists(file_name_s):
            os.remove(file_name_s)
        if os.path.exists(file_name_l):
            os.remove(file_name_l)

    # make sure that basis files do not exist and are not cached
    remove_files()
    cache_cleanup()
    # generate small basis and save
    Ai_s = get_bs_cached(n_s, basis_dir=DATA_DIR, verbose=False)
    cache_cleanup()
    # extend to large basis and save
    Ai_s_l = get_bs_cached(n_l, basis_dir=DATA_DIR, verbose=False)
    cache_cleanup()
    # delete basis files
    remove_files()
    # generate large basis and save
    Ai_l = get_bs_cached(n_l, basis_dir=DATA_DIR, verbose=False)
    cache_cleanup()
    # crop large basis to small
    Ai_l_s = get_bs_cached(n_s, basis_dir=DATA_DIR, verbose=False)
    cache_cleanup()
    # delete basis files (clean-up)
    remove_files()

    assert_allclose(Ai_s, Ai_l_s, atol=1e-15, rtol=1e-15)
    assert_allclose(Ai_l, Ai_s_l, atol=1e-15, rtol=1e-15)


def test_nestorolsen_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = abel.nestorolsen.nestorolsen_transform(x, basis_dir=None, verbose=False)

    assert recon.shape == (n, n)


def test_nestorolsen_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = abel.nestorolsen.nestorolsen_transform(x, basis_dir=None, verbose=False)

    assert_allclose(recon, 0)


def test_nestorolsen_gaussian():
    """Check a gaussian solution for 'nestorolsen' inverse transform"""
    atol = 1e-3
    rtol = 1e-2
    n = 100
    r_max = n - 1

    ref = GaussianAnalytical(n, r_max, symmetric=False, sigma=30)
    tr = np.tile(ref.abel[None, :], (n, 1))  # make a 2D array from 1D

    recon = abel.nestorolsen.nestorolsen_transform(tr, basis_dir=None, verbose=False)
    recon = recon[n // 2 + n % 2]

    assert_allclose(recon, ref.func, atol=atol, rtol=rtol)


def test_nestorolsen_forward_gaussian():
    """Check a gaussian solution for 'nestorolsen' forward transform"""
    atol = 1e-3
    rtol = 1e-2
    n = 100
    r_max = n - 1

    ref = GaussianAnalytical(n, r_max, symmetric=False, sigma=30)
    tr = np.tile(ref.func[None, :], (n, 1))  # make a 2D array from 1D

    recon = abel.nestorolsen.nestorolsen_transform(tr, basis_dir=None, verbose=False,
                                                   direction='forward')
    recon = recon[n // 2 + n % 2]

    assert_allclose(recon, ref.abel, atol=atol, rtol=rtol)


if __name__ == '__main__':
    test_nestorolsen_basis_sets_cache()
    test_nestorolsen_basis_sets_resize()
    test_nestorolsen_shape()
    test_nestorolsen_zeros()
    test_nestorolsen_gaussian()
    test_nestorolsen_forward_gaussian()
