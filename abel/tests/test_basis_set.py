from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

from abel.core import get_left_right_matrices
from abel.io import parse_matlab
from abel.basis import generate_basis_sets

DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def test_generation_basis():
    """
    Check the that the basis.py returns the same result as the BASIS1.m script
    """
    size = 100
    M_ref, Mc_ref = parse_matlab(os.path.join(DATA_DIR, 'dan_basis100{}_1.bst.gz'))

    M, Mc = generate_basis_sets(size+1, size//2, verbose=False)

    yield assert_allclose, Mc_ref, Mc, 1e-7, 1e-100
    yield assert_allclose, M_ref, M, 1e-7, 1e-100
