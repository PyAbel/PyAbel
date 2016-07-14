from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal

import abel
from abel.benchmark import is_symmetric



DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def assert_allclose_msg(x, y, message, rtol=1e-5):
    assert np.allclose(x, y, rtol=1e-5), message


def test_speeds():
    # This very superficial test checks that angular_integration is able to
    # execute (no syntax errors)

    n = 101

    IM = np.random.randn(n, n)

    abel.tools.vmi.angular_integration(IM)


def test_centering_function_shape():
    # ni -> original shape
    # n  -> result of the centering function
    for (y, x) in  [(20, 11), # crop image
                    (21, 11),
                    (5, 11),  # pad image
                    (4, 11)]:
        data = np.zeros((y, x))
        res = abel.tools.center.center_image(data, (y//2, x//2))
        assert_equal( res.shape, (y, x),
                    'Centering preserves shapes for ni={}, n={}'.format(y, x))


def test_centering_function():
    # ni -> original shape of the data is (ni, ni)
    # n_c  -> the image center is (n_c, n_c)

    for (ni, n_c) in [(10, 5),
                         (10,  5),
                         ]:
        arr = np.zeros((ni, ni))

        # arr[n_c-1:n_c+2,n_c-1:n_c+2] = 1
        # # else:
        arr[n_c-1:n_c+1,n_c-1:n_c+1] = 1.0

        res = abel.tools.center.center_image(arr, (n_c, n_c), odd_size=False)
        # The print statements  below can be commented after we fix the centering issue
        # print('Original array')
        # print(arr)
        # print('Centered array')
        # print(res)

        assert_equal( is_symmetric(res), True,\
            'Validating the centering function for ni={}, n_c={}'.format(ni, n_c))

def test_speeds_non_integer_center():  
    # ensures that the rest speeds function can work with a non-integer center
    n  = 101
    IM = np.random.randn(n, n)
    abel.tools.vmi.angular_integration(IM, origin=(50.5, 50.5))

def test_anisotropy_parameter():
    # anisotropy parameter from test image (not transformed)
    IM = abel.tools.analytical.sample_image(name='dribinski')
    
    Beta, Amp, Rmid, Ivstheta, theta = abel.tools.vmi.radial_integration(IM,
                                         radial_ranges=([(0, 33), (92, 108)]))

    assert_almost_equal(-0.13, Beta[0][0], decimal=2)


if __name__ == "__main__":
    test_anisotropy_parameter()
