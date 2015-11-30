from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from abel.tools import calculate_speeds, center_image



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
