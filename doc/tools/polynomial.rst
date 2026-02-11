.. _Polynomials:

Polynomials
===========

Implemented in :mod:`abel.tools.polynomial`.

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
domain where :math:`r_\text{min} \leqslant r \leqslant r_\text{max}`. Namely,

.. math::
    \int r^k \,dy = 2 \int_{y_\text{min}}^{y_\text{max}} r^k \,dy,

.. math::
    y_\text{min,max} = \begin{cases}
        \sqrt{r_\text{min,max}^2 - x^2}, &  x < r_\text{min,max}, \\
        0 & \text{otherwise}.
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

with the same expressions for :math:`C_m`. For example, here are explicit
formulas for several low degrees:

    ========= =====================
    :math:`k` :math:`\int r^k \,dy`
    ========= =====================
    0         :math:`y`
    1         :math:`\frac12 r y + \frac12 x^2 \ln(y + r)`
    2         :math:`\left(\frac13 r^2 + \frac23 x^2\right) y`
    3         :math:`\left(\frac14 r^3 + \frac38 r x^2\right) y +
              \frac38 x^4 \ln(y + r)`
    4         :math:`\left(\frac15 r^4 + \frac4{15} r^2 x^2 +
              \frac8{15} x^4\right) y`
    5         :math:`\left(\frac16 r^5 + \frac5{24} r^3 x^2 +
              \frac5{16} r x^4\right) y + \frac5{16} x^6 \ln(y + r)`
    ...       :math:`\dots`
    ========= =====================

The sums over :math:`m` are computed using `Horner's method
<https://en.wikipedia.org/wiki/Horner's_method>`__ in :math:`x`, which requires
only :math:`x^2`, :math:`y` (see above), :math:`\ln (y + r)` (for polynomials
with odd degrees), and powers of :math:`r` up to :math:`K`.

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
:mod:`abel.tools.analytical` module. Amplitude scaling by multiplying the
“function” (a Python object actually) is not supported there, but it can be
achieved simply by scaling all the coefficients::

    from abel.tools.analytical import PiecewisePolynomial as PP
    c = A * np.array([0, 0, 3, -2])
    smoothstep = PP(..., [(rmin - w, rmin + w, c, rmin - w, 2 * w),
                          (rmin + w, rmax - w, [A]),
                          (rmax - w, rmax + w, c, rmax + w, -2 * w)], ...)


In spherical coordinates
========================

Implemented as :class:`.SPolynomial`.

Axially symmetric bivariate polynomials in spherical coordinates have the
general form

.. math::
    \text{func}(\rho, \theta') = \sum_{m,n=0}^{M,N} c_{mn} \rho^m \cos^n\theta'

(see :ref:`rBasexmath` for definitions of the coordinate systems).

Abel transform
--------------

The forward Abel transform of this function defined on a radial domain
:math:`[\rho_\text{min}, \rho_\text{max}]` (and zero elsewhere) is calculated
as

.. math::
    \text{abel}(r, \theta) = \sum_{m,n=0}^{M,N} c_{mn}
        \int \rho^m \cos^n\theta' \,dz,

where

.. math::
    \rho = \sqrt{r^2 + z^2}, \quad \cos\theta' = \frac{r}{\rho} \cos\theta,

and the Abel integral is taken over the domain where :math:`\rho_\text{min}
\leqslant \rho \leqslant \rho_\text{max}`. That is, for each term we have

.. math::
    \int \rho^m \cos^n\theta' \,dz =
    2 \int\limits_{z_\text{min}}^{z_\text{max}}
        \rho^m \left(\frac{r}{\rho}\right)^n \,dz \cdot \cos^n\theta =
    2 r^m \int\limits_{z_\text{min}}^{z_\text{max}}
        \left(\frac{r}{\rho}\right)^{(n-m)} \,dz \cdot \cos^n\theta,

where

.. math::
    z_\text{min,max} = \begin{cases}
        \sqrt{\rho_\text{min,max}^2 - r^2}, & r < \rho_\text{min,max}, \\
        0 & \text{otherwise}.
    \end{cases}

The antiderivatives

.. math::
    F_{n-m}(r, z) = \int \left(\frac{r}{\rho}\right)^{n-m} \,dz

are given in :ref:`rBasexmath`, with the only difference that besides the
recurrence relation

.. math::
    F_{k+2}(r, z) = \frac{1}{k} \left[z \left(\frac{r}{\rho}\right)^k +
                                      (k - 1) F_k(r, z)\right]

for calculating the terms with positive :math:`k = n - m`, the reverse
recurrence relation

.. math::
    F_k(r, z) = \frac{1}{1 - k} \left[z \left(\frac{r}{\rho}\right)^k -
                                      k F_{k+2}(r, z)\right]

is also used for negative :math:`k`, requred for the terms with :math:`m > n`.

The overall Abel transform thus has the form

.. math::
    \text{abel}(r, \theta) = \sum_{m,n=0}^{M,N} c_{mn}\,
        2 r^m [F_{n-m}(r, z_\text{max}) - F_{n-m}(r, z_\text{min})]
        \cos^n\theta

and is calculated using Horner’s method in :math:`r` and :math:`\cos\theta`
after precomputing the :math:`F_{n-m}(r, z_\text{min,max})` pairs for each
needed value of :math:`n - m` (there are at most :math:`M + N + 1` of them, if
all :math:`M \times N` coefficients :math:`c_{mn} \ne 0`).

Notice that these calculations are relatively expensive, since they are done
for all pixels with :math:`\rho \leqslant \rho_\text{max}`, for each of them
involving summation and multiplication of up to :math:`2MN` terms in the above
expression and evaluating transcendent functions present in :math:`F_k(r, z)`.
Moreover, the numerical problems for high-degree polynomials thus can be even
more severe than for `univariate polynomials <#polynomials>`_.


Approximate Gaussian
====================

Implemented as :class:`.ApproxGaussian`.

The Gaussian function

.. math::
    A \exp\left(-\frac{(r - r_0)^2}{2\sigma^2}\right)

is useful for representing peaks in simulated data but does not have an
analytical Abel transform unless :math:`r_0 = 0`. However, it can be
approximated by piecewise polynomials to any accuracy, and these polynomials
can be Abel-transformed analytically, as shown above, thus providing an
arbitrarily accurate approximation to the Abel transform of the initial
Gaussian function.

In practice, it is sufficient to find the approximating piecewise polynomial
for the Gaussian function :math:`g(r) = \exp(-r^2/2)` with unit amplitude and
unit standard deviation, and the polynomial coefficients can then be scaled as
described `above <#affine-transformation>`_ to account for any :math:`A`,
:math:`r_0` and :math:`\sigma`.

The goal is therefore to find :math:`f(r)` such that :math:`|f(r) - g(r)|
\leqslant \varepsilon` for a given tolerance :math:`\varepsilon`. The
approximation implemented here uses a piecewise quadratic polynomial:

.. math::
    f(r) = \begin{cases}
        f_n(r) = c_{0,n} + c_{1,n} r + c_{2,n} r^2, & r \in [R_n, R_{n+1}], \\
        0, & r \notin [R_0, R_N],
    \end{cases}

where the domain is split into :math:`N` intervals :math:`[R_n, R_{n+1}]`,
:math:`n = 0, \dots, N - 1`. The strategy used for minimizing the number of
intervals is to find the splitting points such that

.. math::
    f_n(r) = g(r) \quad \text{for} \quad
    r = R_n, R_{n+\frac12}, R_{n+1}, ~\text{where}~
    R_{n+\frac12} \equiv \frac{R_n + R_{n+1}}{2},
.. math::
    \max_{r \in [R_n, R_{n+1}]} \big|f_n(r) - g(r)\big| = \varepsilon,

in other words, each parabolic segment matches the :math:`g(r)` values at the
endpoints and midpoint of its interval, and its maximal deviation from
:math:`g(r)` equals :math:`\varepsilon`. The process starts from :math:`R_0 =
\sqrt{-2 \ln(\varepsilon/2)}`, such that :math:`g(R_0) = \varepsilon/2`, but
setting :math:`f_0(R_0) = 0` for continuity. Then subsequent points :math:`R_1,
R_2, \dots` are found by solving :math:`\max_{r \in [R_n, R_{n+1}]} \big|f_n(r)
- g(r)\big| \approx \varepsilon` for :math:`R_{n+1}` numerically, using the
following approximation obtained from the 3rd-order term of the :math:`g(r)`
Taylor series (by construction, :math:`f_n(r)` reproduces the lower-order terms
exactly, and the magnitudes of higher-order terms are much smaller):

.. math::
    & \max_{r \in [R_n, R_{n+1}]} \big|f_n(r) - g(r)\big| \approx \\
    & \qquad
    \approx \max_{r = R_n, R_{n+\frac12}, R_{n+1}}
        \left|\frac{g'''(r)}{3!}\right| \cdot
        \max_{r \in [R_n, R_{n+1}]}
            \big|(r - R_n)(r - R_{n+\frac12})(r - R_{n+1})\big| = \\
    & \qquad
    = \max_{r = R_n, R_{n+\frac12}, R_{n+1}} \big|(3 - r^2) r g(r)\big|
        \cdot \frac{|R_n - R_{n+1}|^3}{72\sqrt{3}}.

This process is repeated until :math:`r = 0` is reached, after which the found
splitting is symmetrically extended to :math:`-R_0 \leqslant r < 0`, and the
polynomial coefficients for each segment are trivially calculated from the
equations :math:`f_n(r) = g(r)` for :math:`r = R_n, R_{n+\frac12}, R_{n+1}`.

As an example, here is the outcome for the default approximation accuracy
≲0.5 %, resulting in just 7 segments::

    from abel.tools.polynomial import ApproxGaussian, PiecewisePolynomial
    r = np.arange(201)
    r0 = 100
    sigma = 20
    # actual Gaussian function
    gauss = np.exp(-((r - r0) / sigma)**2 / 2)
    # approximation with default tolerance (~0.5%)
    approx = PiecewisePolynomial(r, ApproxGaussian().scaled(1, r0, sigma))

.. plot::

    from approx_gaussian import plot, approx, gauss
    plot('func', approx.func, gauss, 0.005)

The Abel transform of this approximation is even more accurate, having the
maximal relative deviation ~0.35/170 ≈ 0.2 %:

.. plot::

    from approx_gaussian import plot, approx, ref
    plot('abel', approx.abel, ref.abel, 0.35)

A practical example of using :class:`.ApproxGaussian` with
:class:`.PiecewiseSPolynomial` can be found in the :class:`.SampleImage` source
code.
