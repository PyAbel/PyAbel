# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from numpy.testing import assert_allclose
from abel.tools.math import gradient


def test_gradient():
    x = np.random.randn(10,10)

    dx = np.gradient(x)[1]
    dx2 = gradient(x)

    assert_allclose(dx, dx2)

