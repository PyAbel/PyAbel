from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import itertools

import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.ndimage.interpolation import shift

import abel
from abel.tools.center import find_origin, center_image, set_center


def test_find_origin():
    """
    Test find_origin methods.
    """
    size = [12, 13]
    row, col = 5.4, 6.6  # origin
    w = 3.0  # gaussian width parameter (sqrt(2) * sigma)
    for rows in size:
        y2 = ((np.arange(rows) - row) / w)**2
        for cols in size:
            x2 = ((np.arange(cols) - col) / w)**2
            data = np.exp(-(x2 + y2[:, None]))
            axes = (1, 0)
            # (not testing trivial 'image_center', which does not find origin)
            for method in ['com', 'convolution', 'gaussian', 'slice']:
                origin = find_origin(data, method, axes)
                ref = (row if 0 in axes else rows // 2,
                       col if 1 in axes else cols // 2)
                tol = 0.2  # 'convolution' rounds to 0.5 pixels
                assert_allclose(origin, ref, atol=tol, verbose=False,
                                err_msg='-> {} x {}, method = {}, axes = {}: '
                                        'origin = {} not equal {}'.
                                        format(rows, cols, method, axes,
                                               origin, ref))


def test_set_center_int():
    """
    Test whole-pixel shifts.
    """
    # input sizes
    size = [4, 5]
    # input size, crop, origin -> output elements
    param = {4: {'maintain_size': [[None, '1234'],
                                   [0,    '0012'],
                                   [1,    '0123'],
                                   [2,    '1234'],
                                   [3,    '2340']],
                 'valid_region':  [[None, '1234'],
                                   [0,    '1'],
                                   [1,    '123'],
                                   [2,    '234'],
                                   [3,    '4']],
                 'maintain_data': [[None, '1234'],
                                   [0,    '0001234'],
                                   [1,    '01234'],
                                   [2,    '12340'],
                                   [3,    '1234000']]},
             5: {'maintain_size': [[None, '12345'],
                                   [0,    '00123'],
                                   [1,    '01234'],
                                   [2,    '12345'],
                                   [3,    '23450'],
                                   [4,    '34500']],
                 'valid_region':  [[None, '12345'],
                                   [0,    '1'],
                                   [1,    '123'],
                                   [2,    '12345'],
                                   [3,    '345'],
                                   [4,    '5']],
                 'maintain_data': [[None, '12345'],
                                   [0,    '000012345'],
                                   [1,    '0012345'],
                                   [2,    '12345'],
                                   [3,    '1234500'],
                                   [4,    '123450000']]}}
    # all size combinations
    for rows, cols in itertools.product(size, repeat=2):
        # test data: consecutive numbers from 1, row by row
        data = (np.arange(rows * cols) + 1).reshape((rows, cols))
        # all crop options
        for crop in ['maintain_size', 'valid_region', 'maintain_data']:
            # all origin rows
            for row, rref in param[rows][crop]:
                # vector or reference rows
                rref = np.array([int(n) for n in rref])
                # all origin columns
                for col, cref in param[cols][crop]:
                    # vector of reference columns
                    cref = np.array([int(n) for n in cref])
                    # reference array
                    ref = (rref[:, None] - 1) * cols + cref
                    ref[rref == 0] = 0
                    ref[:, cref == 0] = 0
                    # check set_center() result
                    result = set_center(data, (row, col), crop=crop)
                    assert_equal(result, ref, verbose=False,
                                 err_msg='-> {} x {}, origin = {}, crop = {}\n'
                                         'result =\n{}\n'
                                         'must be =\n{}'.
                                         format(rows, cols, (row, col), crop,
                                                result, ref))


def test_set_center_float():
    """
    Test fractional shifts.
    """
    # input sizes
    size = [10, 11]
    # default origin coordinate (substituting None)
    default = 5.0
    # input size, origin, crop -> output size, non-zero range
    param = {10: [(None, {'maintain_size': [10, (0, 10)],
                          'valid_region':  [10, (0, 10)],
                          'maintain_data': [10, (0, 10)]}),
                  (2.5,  {'maintain_size': [10, (2, 10)],
                          'valid_region':  [5,  (0,  5)],
                          'maintain_data': [15, (4, 15)]}),
                  (3.5,  {'maintain_size': [10, (1, 10)],
                          'valid_region':  [7,  (0,  7)],
                          'maintain_data': [13, (2, 13)]}),
                  (4.5,  {'maintain_size': [10, (0, 10)],
                          'valid_region':  [9,  (0,  9)],
                          'maintain_data': [11, (0, 11)]}),
                  (5.5,  {'maintain_size': [10, (0, 10)],
                          'valid_region':  [7,  (0,  7)],
                          'maintain_data': [13, (0, 11)]}),
                  (6.5,  {'maintain_size': [10, (0,  9)],
                          'valid_region':  [5,  (0,  5)],
                          'maintain_data': [15, (0, 11)]})],
             11: [(None, {'maintain_size': [11, (0, 11)],
                          'valid_region':  [11, (0, 11)],
                          'maintain_data': [11, (0, 11)]}),
                  (3.5,  {'maintain_size': [11, (1, 11)],
                          'valid_region':  [7,  (0,  7)],
                          'maintain_data': [15, (3, 15)]}),
                  (4.5,  {'maintain_size': [11, (0, 11)],
                          'valid_region':  [9,  (0,  9)],
                          'maintain_data': [13, (1, 13)]}),
                  (5.5,  {'maintain_size': [11, (0, 11)],
                          'valid_region':  [9,  (0,  9)],
                          'maintain_data': [13, (0, 12)]}),
                  (6.5,  {'maintain_size': [11, (0, 10)],
                          'valid_region':  [7,  (0,  7)],
                          'maintain_data': [15, (0, 12)]})]}
    w = 2.0  # gaussian width parameter (sqrt(2) * sigma)
    # all size combinations
    for rows, cols in itertools.product(size, repeat=2):
        # all origin "rows"
        for row, rparam in param[rows]:
            y2 = ((np.arange(rows) - (row or default)) / w)**2
            # all origin "columns"
            for col, cparam in param[cols]:
                x2 = ((np.arange(cols) - (col or default)) / w)**2
                # test data: gaussian centered at (row, col)
                data = np.exp(-(x2 + y2[:, None]))
                # all crop options
                for crop in ['maintain_size', 'valid_region', 'maintain_data']:
                    # check set_center() result
                    result = set_center(data, (row, col), crop=crop)
                    refrows, rrange = rparam[crop]
                    refcols, crange = cparam[crop]
                    refshape = (refrows, refcols)
                    refrange = (slice(*rrange), slice(*crange))
                    reforigin = (refrows // 2 if row else default,
                                 refcols // 2 if col else default)
                    msg = '-> {} x {}, origin = {}, crop = {}: '.\
                          format(rows, cols, (row, col), crop)
                    # shape
                    assert_equal(result.shape, refshape, verbose=False,
                                 err_msg=msg + 'shape {} not equal {}'.
                                               format(result.shape, refshape))
                    # non-zero data
                    assert_equal(result[refrange] != 0, True,
                                 err_msg=msg + 'zeros in non-zero range')
                    # zero padding
                    tmp = result.copy()
                    tmp[refrange] = 0
                    assert_equal(tmp, 0, err_msg=msg +
                                 'non-zeros outside non-zero range')
                    # gaussian center
                    origin = find_origin(result, 'gaussian')
                    assert_allclose(origin, reforigin, atol=0.01,
                                    verbose=False, err_msg=msg +
                                    'shifted center {} not equal {}'.
                                    format(origin, reforigin))


def test_set_center_axes():
    """
    Test "None" origin components and axes selection.
    """
    for N in [4, 5]:
        data = np.arange(N**2).reshape((N, N))
        c = N // 2
        msg = '-> N = {}, '.format(N)
        assert_equal(set_center(data, (None, None)),
                     data,
                     err_msg=msg + '(None, None)')
        assert_equal(set_center(data, (0, 0), axes=[]),
                     data,
                     err_msg=msg + '(0, 0), axes=[]')
        assert_equal(set_center(data, (0, None)),
                     set_center(data, (0, c)),
                     err_msg=msg + '(0, None)')
        assert_equal(set_center(data, (None, 0)),
                     set_center(data, (c, 0)),
                     err_msg=msg + '(None, 0)')
        assert_equal(set_center(data, (0, 0), axes=0),
                     set_center(data, (0, c)),
                     err_msg=msg + '(0, 0), axes=0')
        assert_equal(set_center(data, (0, 0), axes=1),
                     set_center(data, (c, 0)),
                     err_msg=msg + '(0, 0), axes=1')


def test_set_center_order():
    """
    Test rounding for order = 0 and exact output for order = 1.
    """
    data = data = np.ones((5, 5))
    origin = np.array([1.9, 2.2])
    # check origin rounding for order = 0
    assert_equal(set_center(data, origin, order=0),
                 set_center(data, origin.round()),
                 err_msg='-> order = 0 not equal round(origin)')
    # check output for order = 1:
    # maintain_size
    result = set_center(data, origin, 'maintain_size', order=1)
    ref = np.outer([0.9, 1, 1, 1, 1],
                   [1, 1, 1, 1, 0.8])
    assert_allclose(result, ref,
                    err_msg='-> crop = maintain_size, order = 1')
    # valid_region
    result = set_center(data, origin, 'valid_region', order=1)
    ref = np.ones((3, 3))
    assert_allclose(result, ref,
                    err_msg='-> crop = valid_region, order = 1')
    # maintain_data
    result = set_center(data, origin, 'maintain_data', order=1)
    ref = np.outer([0, 0.9, 1, 1, 1, 1, 0.1],
                   [0.2, 1, 1, 1, 1, 0.8, 0])
    assert_allclose(result, ref,
                    err_msg='-> crop = maintain_data, order = 1')


def test_center_image():

    # BASEX sample image, Gaussians at 10, 15, 20, 70,85, 100, 145, 150, 155
    # image width, height n = 361, origin = (180, 180)
    IM = abel.tools.analytical.SampleImage(n=361, name="dribinski").image

    # artificially displace origin, now at (179, 182)
    IMx = shift(IM, (-1, 2))
    true_origin = (179, 182)

    # find_origin using 'slice' method
    origin = find_origin(IMx, method="slice")

    assert_allclose(origin, true_origin, atol=1)

    # find_origin using 'com' method
    origin = find_origin(IMx, method="com")

    assert_allclose(origin, true_origin, atol=1)

    # check single axis - vertical
    # center shifted image IMx in the vertical direction only
    IMc = center_image(IMx, method="com", axes=1)
    # determine the origin
    origin = find_origin(IMc, method="com")

    assert_allclose(origin, (179, 180), atol=1)

    # check single axis - horizontal
    # center shifted image IMx in the horizontal direction only
    IMc = center_image(IMx, method="com", axes=0)
    origin = find_origin(IMc, method="com")

    assert_allclose(origin, (180, 182), atol=1)

    # check even image size returns odd
    # drop off one column, to make an even column image
    IM = IM[:, :-1]
    m, n = IM.shape

    IMy = center_image(IM, method="slice", odd_size=True)

    assert_allclose(IMy.shape, (m, n-1))


if __name__ == "__main__":
    test_find_origin()
    test_set_center_axes()
    test_set_center_int()
    test_set_center_float()
    test_set_center_order()
    test_center_image()
