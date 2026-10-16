"""Operand-order regressions and a Cartesian projection oracle."""

from itertools import zip_longest

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from abel.tools.polynomial import Angular, SPolynomial, rcos


@pytest.mark.parametrize('left,right', [
    ([2], [1]),
    ([1], [0, 0, 1]),
    ([0, 0, 1], [1]),
    ([1, 2, 3], [4, 5, 6]),
    ([1, 2], [1, 2]),
    ([1, 0, 0], [0, 1]),
    ([], [1, 2]),
    ([1, 2], []),
    ([], []),
])
@pytest.mark.parametrize('scale', [2.0**-900, 1.0, 2.0**900])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64])
def test_angular_subtraction_order(left, right, scale, dtype):
    """Subtract in operand order, not coefficient-array length order."""
    a = Angular(np.array(left, dtype=dtype)) * scale
    b = Angular(np.array(right, dtype=dtype)) * scale
    before_a, before_b = a.c.copy(), b.c.copy()
    result = a - b
    # Python integer arithmetic, with explicit zero extension.
    expected = [int(x) - int(y) for x, y in
                zip_longest(left, right, fillvalue=0)]
    assert_array_equal(result.c, np.array(expected) * scale)
    assert_array_equal(a.c, before_a)
    assert_array_equal(b.c, before_b)
    assert not np.shares_memory(result.c, a.c)
    assert not np.shares_memory(result.c, b.c)


@pytest.mark.parametrize('radius', [1e-3, 1.0, 1e3])
@pytest.mark.parametrize('inner_fraction', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('quantity', ['func', 'abel'])
def test_angular_subtraction_projection(radius, inner_fraction, dtype, quantity):
    """rho² sin²(theta) is x²+t² in Cartesian coordinates (t is LOS)."""
    axis = np.linspace(-1.25, 1.25, 11) * radius
    rows, cols = np.meshgrid(axis, axis, indexing='ij')
    r, cos = rcos(rows, cols)
    inner = inner_fraction * radius
    angular = Angular(np.array([1], dtype=dtype)) - Angular.cos(2)
    p = SPolynomial(r, cos, inner, radius, [0, 0, 1] * angular)

    # At t=0 the density is x². The radial support is half-open.
    density = np.where((inner <= r) & (r < radius), cols**2, 0)
    # Integrate x²+t² directly from -L to L, subtracting the inner sphere.
    def sphere_projection(bound):
        length = np.sqrt(np.maximum(bound**2 - rows**2 - cols**2, 0))
        return 2 * cols**2 * length + (2 / 3) * length**3

    projection = sphere_projection(radius) - sphere_projection(inner)
    if quantity == 'func':
        assert_allclose(p.func, density, rtol=1e-13,
                        atol=radius**2 * 1e-14)
        assert np.min(p.func) >= -radius**2 * 1e-14
    else:
        assert_allclose(p.abel, projection, rtol=1e-13,
                        atol=radius**3 * 1e-14)
        assert np.min(p.abel) >= -radius**3 * 1e-14
