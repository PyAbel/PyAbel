from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from abel.tools import calculate_speeds, center_image
from abel.benchmark import is_symmetric



DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def assert_allclose_msg(x, y, message, rtol=1e-5):
    assert np.allclose(x, y, rtol=1e-5), message


def test_speeds():
    # This very superficial test checks that calculate_speeds is able to
    # execute (no syntax errors)

    n = 101

    IM = np.random.randn(n, n)

    calculate_speeds(IM, n)


def test_centering_function_shape():
    # ni -> original shape
    # n  -> result of the centering function
    for (ni, n) in [(20, 10), # crop image
                    (20, 11),
                    (4, 10),  # pad image
                    (4, 11)]:
        data = np.zeros((ni, ni))
        res = center_image(data, (ni//2, ni//2), n)
        yield assert_equal, res.shape, (n, n),\
                    'Centering preserves shapes for ni={}, n={}'.format(ni, n)


def test_centering_function():
    # ni -> original shape of the data is (ni, ni)
    # n_c  -> the image center is (n_c, n_c)
    # n  -> after the centering function, the array has a shape (n, n)

    for (ni, n_c, n) in [(9, 5, 3),
                         (9, 5, 4),
                         ]:
        arr = np.zeros((ni, ni))

        if n % 2 == 1:
            arr[n_c-1:n_c+2,n_c-1:n_c+2] = 1
        else:
            arr[n_c-1:n_c+1,n_c-1:n_c+1] = 1

        res = center_image(arr, (n_c, n_c), n)
        # The print statements  below can be commented after we fix the centering issue
        print('Original array')
        print(arr)
        print('Centered array')
        print(res)

        yield assert_equal, is_symmetric(res), True,\
            'Validating the centering function for ni={}, n_c={}, n={}'.format(ni, n_c, n)

