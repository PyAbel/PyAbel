:orphan:

.. _Polynomials:

Polynomials
===========

Implemented in :py:class:`abel.tools.polynomial`.

Abel transform
--------------

The Abel transform of a polynomial

.. math::
    \text{func}(r) = \sum_{k=0}^K c_k r^k

defined on a domain :math:`[r_\text{min}, r_\text{max}]` (and zero elsewhere)
is calculated as

.. math::
    \text{abel}(x) = \sum_{k=0}^K c_k \int r^k \,dy,

where :math:`r = \sqrt{x^2 + y^2}`, and the Abel integral is taken over the
domain where :math:`r_\text{min} \le r \le r_\text{max}`. Namely,

.. math::
    \int r^k \,dy = 2 \int_{y_\text{min}}^{y_\text{max}} r^k \,dy,

.. math::
    y_\text{min,max} = \begin{cases}
        \sqrt{r_\text{min,max}^2 - x^2}, &  x < r_\text{min,max}, \\
        0 & \text{otherwise},
    \end{cases}

These integrals for any power :math:`k` are easily obtained from the recursive
relation

.. math::
    \int r^k \,dy = \frac1{k + 1} \left(
        y r^k + k x^2 \int r^{k-2} \,dy
    \right).

For **even** :math:`k` this yields a polynomial in :math:`y` and powers of
:math:`x` and :math:`r`:

.. math::
    \int r^k \,dy = y \sum_{m=0}^k C_m r^m x^{k-m},
    \qquad (\text{summing over even}\ m)

.. math::
    C_k = \frac1{k + 1}, \quad
    C_{m-2} = \frac m{m - 1} C_m.

For **odd** :math:`k`, the recursion terminates at

.. math::
    \int r^{-1} \,dy = \ln (y + r),

so

.. math::
    \int r^k \,dy = y \sum_{m=1}^k C_m r^m x^{k-m} + C_1 x^{k+1} \ln (y + r),
    \qquad (\text{summing over odd}\ m)

with the same expressions for :math:`C_m`.

These sums are computed using Horner's method in :math:`x`, which requires only
:math:`x^2`, :math:`y` (see above), :math:`\ln (y + r)` (for polynomials with
odd degrees), and powers of :math:`r` up to :math:`K`.

The sum of the integrals, however, is computed by direct addition. In
particular, this means that an attempt to use this method for high-degree
polynomials (for example, approximating some function with a 100-degree Taylor
polynomial) will most likely fail due to `loss of significance
<https://en.wikipedia.org/wiki/Loss_of_significance>`_ in floating-point
operations. Splines are a much better choice in this respect, although at
sufficiently large :math:`r` and :math:`x` (≳10 000) these numerical problems
might become significant even for cubic polynomials.


Affine transformation
---------------------

It is sometimes convenient to define a polynomial in some canonical form and
adapt it to the particular case by an affine transformation (translation and
scaling) of the independent variable, like in the `example`_ below.

The scaling around :math:`r = 0` is

.. math::
    P'(r) = P(r/s) = \sum_{k=0}^K c_k (r/s)^k,

which applies an :math:`s`-fold stretching to the function. The coefficients
of the transformed polynomial are thus

.. math::
    c'_k = c_k / s^k.

The translation is

.. math::
    P'(r) = P(r - r_0) = \sum_{k=0}^K c_k (r - r_0)^k,

which shifts the origin to :math:`r_0`. The coefficients of the transformed
polynomial can be obtained by expanding all powers of the binomial :math:`r -
r_0` and collecting the powers of :math:`r`. This is implemented in a matrix
form

.. math::
    \mathbf{c}' = \mathrm{M} \mathbf{c},

where the coefficients are represented by a column vector :math:`\mathbf{c} =
(c_0, c_1, \dots, c_K)^\mathrm{T}`, and the matrix :math:`\mathrm{M}` is the
`Hadamard product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`_
of the upper-triangular `Pascal matrix
<https://en.wikipedia.org/wiki/Pascal_matrix>`_ and the `Toeplitz matrix
<https://en.wikipedia.org/wiki/Toeplitz_matrix>`_ of :math:`r_0^k`:

.. math::
    \mathrm{M} =
    \begin{pmatrix}
        1      & 1      & 1      & 1      & 1      & \cdots \\
        0      & 1      & 2      & 3      & 4      & \cdots \\
        0      & 0      & 1      & 3      & 6      & \cdots \\
        0      & 0      & 0      & 1      & 4      & \cdots \\
        0      & 0      & 0      & 0      & 1      & \cdots \\
        \vdots & \vdots & \vdots & \vdots & \vdots & \ddots  \\
    \end{pmatrix}
    \circ
    \begin{pmatrix}
        r_0^0  & r_0^1  & r_0^2  & \ddots & r_0^K     \\
        0      & r_0^0  & r_0^1  & \ddots & r_0^{K-1} \\
        0      & 0      & r_0^0  & \ddots & r_0^{K-2} \\
        \ddots & \ddots & \ddots & \ddots & \ddots    \\
        0      & 0      & 0      & \ddots & r_0^0
    \end{pmatrix}.


Example
-------

Consider a two-sided step function with soft edges:

.. plot:: tools/smoothstep.py

The edges can be represented by the cubic `smoothstep
<https://en.wikipedia.org/wiki/Smoothstep>`_ function

.. math::
    S(r) = 3r^2 - 2r^3,

which smoothly rises from :math:`0` at :math:`r = 0` to :math:`1` at :math:`r =
1`. The left edge requires stretching it by :math:`2w` and shifting the origin
to :math:`r_\text{min} - w`. The right edge is :math:`S(r)` stretched by
:math:`-2w` (the negative sign mirrors it horizontally) and shifted to
:math:`r_\text{max} + w`. The shelf is just a constant (zeroth-degree
polynomial). It can be set to :math:`1`, and then the desired function with the
amplitude :math:`A` is obtained by multiplying the resulting piecewise
polynomial by :math:`A`:

::

    import matplotlib.pyplot as plt
    import numpy as np

    from abel.tools.polynomial import PiecewisePolynomial as PP

    r = np.arange(51.0)

    rmin = 10
    rmax = 40
    w = 5
    A = 3

    c = [0, 0, 3, -2]
    smoothstep = A * PP(r, [(rmin - w, rmin + w, c, rmin - w, 2 * w),
                            (rmin + w, rmax - w, [1]),
                            (rmax - w, rmax + w, c, rmax + w, -2 * w)])

    fig, axs = plt.subplots(2, 1)

    axs[0].set_title('func')
    axs[0].set_xlabel('$r$')
    axs[0].plot(r, smoothstep.func)

    axs[1].set_title('abel')
    axs[1].set_xlabel('$x$')
    axs[1].plot(r, smoothstep.abel)

    plt.tight_layout()
    plt.show()

``Polynomial`` and ``PiecewisePolynomial`` are also accessible through the
:py:class:`abel.tools.analytical` module. Amplitude scaling by multiplying the
“function” (a Python object actually) is not supported there, but it can be
achieved simply by scaling all the coefficients::

    from abel.tools.analytical import PiecewisePolynomial as PP
    c = A * np.array([0, 0, 3, -2])
    smoothstep = PP(..., [(rmin - w, rmin + w, c, rmin - w, 2 * w),
                          (rmin + w, rmax - w, [A]),
                          (rmax - w, rmax + w, c, rmax + w, -2 * w)], ...)

.. |:ref:`abeltoolsanalytical`| replace:: ``abel.tools.analytical``
