# -*- coding: utf-8 -*-
from __future__ import absolute_import

from numpy.testing import assert_allclose

from abel.tools.analytical import TransformPair
from abel.daun import daun_transform


def test_transform_pairs():
    """
    Test analytical Abel-transform pairs.
    """
    n = 101  # dr = 0.01

    for i in range(1, 8):
        pair = TransformPair(n, profile=i)
        proj = daun_transform(pair.func, degree=3, direction='forward',
                              dr=pair.dr, verbose=False)

        clip = None
        tol = 3e-6
        if i == 4:  # has small discontinuity
            tol = 3e-4
        elif i == 5:  # harsh
            tol = 0.05
            clip = -2

        assert_allclose(pair.abel[:clip], proj[:clip], atol=tol,
                        err_msg='-> ' + pair.label)


if __name__ == "__main__":
    test_transform_pairs()
