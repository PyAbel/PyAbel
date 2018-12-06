:orphan:

.. _BASEXcomp:

BASEX: computational details
============================

The BASEX article does not provide the derivation of the basis projections and
is very terse regarding their computation, so here we provide the missing
explanations. The differences in the PyAbel implementation of the method are
also discussed below.


Basis projections
-----------------

The basis functions are

.. math::
    \rho_k(r) = (e/k^2)^{k^2} (r/\sigma)^{2k^2} e^{-(r/\sigma)^2},

or in a reduced form,

.. math::
    \rho_k(u) = A_k \, u^{2k^2} e^{-u^2},

.. math::
    A_k = (e/k^2)^{k^2}, \quad u = r/\sigma.

Their Abel transform is most easily obtained by considering the projection in
rectangular coordinates:

.. math::
    \chi_k(x) =
    \int_{-\infty}^\infty \rho_k(r) \,dy =
    2 \int_0^\infty \rho_k(r) \,dy,

.. math::
    r = \sqrt{x^2 + y^2}.

Then

.. math::
    \int_0^\infty \left(\sqrt{x^2 + y^2}\right)^{2k^2}
        e^{-\left(x^2 + y^2\right)} \,dy =
    \int_0^\infty \left(x^2 + y^2\right)^{k^2}
        e^{-x^2} e^{-y^2} \,dy.

After expanding the binomial :math:`\left(x^2 + y^2\right)^{k^2}`, this
integral becomes

.. math::
    e^{-x^2} \sum_{l=0}^{k^2} \binom{k^2}l x^{2l}
        \int_0^\infty y^{2\left(k^2-l\right)} e^{-y^2} \,dy,

where the binomial coefficients

.. math::
    \binom{k^2}l = \frac{k^2!}{l! \, (k^2-l)!} =
    \frac{\Gamma(k^2 + 1)}{\Gamma(l + 1) \, \Gamma(k^2 - l + 1)},

and the integrals are also expressed through the `gamma function
<https://en.wikipedia.org/wiki/Gamma_function>`_:

.. math::
    \int_0^\infty y^{2\left(k^2-l\right)} e^{-y^2} \,dy \stackrel{t=y^2}{=}
    \int_0^\infty t^{k^2-l} e^{-t} \frac1{2\sqrt{t}} \,dt =
    \frac12 \Gamma\left(k^2 - l + \frac12\right).

The complete expression for the projections (in a reduced form, :math:`u =
x/\sigma`) is thus

.. math::
    \chi_k(u) = A_k \sigma e^{-u^2} \sum_{l=0}^{k^2}
        \frac{\Gamma(k^2 + 1) \, \Gamma\left(k^2 - l + \frac12\right)}
             {\Gamma(l + 1) \, \Gamma(k^2 - l + 1)}
        u^{2l}.

The case :math:`k = 0` is special, since formally :math:`A_0 = (e/0)^{0}`,
which is undefined. However, taking the limit :math:`k \to 0`, we obtain

.. math::
    \rho_0(u) = e^{-u^2},

the Abel transform of which is simply

.. math::
    \chi_0(u) = \sqrt{\pi}\,\sigma e^{-u^2}.

.. note::
    The original MATLAB implementation by Dribinski used an incorrect prefactor
    “2” instead if “:math:`\sqrt{\pi}`” in calculations of the basis
    projections :math:`\chi_k` (in the above expression the :math:`\sqrt{\pi}`
    factor for :math:`k > 0` is invisibly present in the :math:`\Gamma(\ldots +
    1/2)` terms). The `BASEX.exe` program by Karpichev also uses these
    MATLAB-generated basis sets and has the same problem, producing intensities
    off by a factor of :math:`\sqrt{\pi}/2` and applying regularization with a
    strength off by a square of that factor.

    We use the correct expressions for all calculations.

Computations
------------

The above expressions for :math:`\rho_k(u)` and :math:`\chi_k(u)` involve very
small (:math:`e^{-u^2}`) and very large (:math:`u^{2k^2}`) numbers and thus
will cause floating-point underflow/overflow if computed directly. However,
they can be recast as

.. math::
    \rho_k(u) = \exp\left[
        \left(1 - \ln k^2\right) k^2 + \ln u \cdot 2k^2 - u^2
    \right],

.. math::
    \begin{aligned}
    \chi_k(u) = \sigma \smash{\sum_{l=0}^{k^2} \exp\Big[}
        & \left(1 - \ln k^2\right) k^2 - u^2 + {} \\
        &+ \ln\Gamma(k^2 + 1) + \ln\Gamma\left(k^2 - l + \frac12\right) - {} \\
        &- \ln\Gamma(l + 1) - \ln\Gamma(k^2 - l + 1) + {} \\
        &+ \ln u \cdot 2l
    \Big],
    \end{aligned}

in which all terms are comparable to :math:`k^2` and :math:`u^2`. In
particular, :math:`\ln \Gamma(z) \sim (\ln z - 1) z` and is available directly
as :py:func:`scipy.special.gammaln`.

The :math:`\ln \Gamma(z)` functions are relatively computationally expensive,
but as can be seen, computing the projections :math:`\chi_k(u)` for all
:math:`k` up to :math:`K` requires only the values of :math:`\ln \Gamma(n)` and
:math:`\Delta \ln \Gamma(n) = \ln \Gamma(n) - \ln \Gamma(n - 1/2)` for integers
:math:`n = 1, \dots, K^2 + 1`. They are precomputed and cached before the basis
generation. This requires :math:`O(K^2)` extra memory (comparable to
:math:`O(NK)` for the basis matrices themselves), but saves :math:`O(NK^2)`
evaluations (see below) of these special functions.

The BASEX article mentions that actually “only a few terms contribute to the
sum”, but does not give any quantitative estimations. In order to obtain the
practical constraints on the summation index, consider how the exponential
terms change with :math:`l` at fixed :math:`k` and :math:`u`:

.. math::
    \exp[\dots] = \exp f_{k,u} \cdot \exp g_{k,u}(l),

where

.. math::
    f_{k,u} = \left(1 - \ln k^2\right) k^2 - u^2 + \ln\Gamma(k^2 + 1)

does not depend on :math:`l`, and

.. math::
    \begin{aligned}
        g_{k,u}(l) &= -\underbrace{\ln\Gamma(l + 1)}_{\approx (\ln l - 1)l} -
            \underbrace{\Delta\ln\Gamma(k^2 - l + 1)}_{\approx \ln(k^2 - l)/2} +
            \ln u \cdot 2l = \\
        &= (1 + \ln u^2 - \ln l) l + o(l).
    \end{aligned}

The last expression (:math:`g` without sublinear terms) reaches its maximum at
:math:`l_\text{max} = u^2` and behaves near it as

.. math::
    g_{k,u}(l_\text{max} + \delta) = u^2 - \frac{\delta^2}{2u^2} + o(\delta^2).

From the practical perspective, the terms

.. math::
    \exp g_{k,u}(l) < \varepsilon_\text{FP} \cdot \exp g_{k,u}(l_\text{max}),

where :math:`\varepsilon_\text{FP} \sim 10^{-16}` is the floating-point
precision, will be lost in rounding errors and thus do not need to be computed.
This inequality can be transformed into

.. math::
    g_{k,u}(l) - g_{k,u}(l_\text{max}) = -\frac{\delta^2}{2u^2} <
        \ln \varepsilon_\text{FP},

from which

.. math::
    \delta > \sqrt{-2 \ln\varepsilon_\text{FP}} \, u \approx 8.6 \, u.

That is, the projections :math:`\chi_k(u)` can be computed to within the
floating-point precision by summing only the terms with :math:`l \in
[l_\text{max} - \delta, l_\text{max} + \delta]`, where :math:`l_\text{max} =
u^2` and :math:`\delta = 9\,u`.

Since :math:`\max u = K`, the total time complexity of computing :math:`K`
basis projections at :math:`N` points is :math:`O(NK^2)`.


----


Intensity correction
--------------------

The Gaussian-like BASEX basis functions do not sum to unity:

.. plot:: transform_methods/basex-basis.py

so they cannot describe a flat distribution, and for :math:`\sigma \ne 1` these
intensity oscillations are visible in the reconstructed distributions. In
addition, the basis projections are sampled only at pixel centers, which does
not satisfy the requirements of the `sampling theorem
<https://en.wikipedia.org/wiki/Nyquist–Shannon_sampling_theorem>`_ for their
adequate representation. In particular, this leads to a reconstructed-intensity
bias in the most useful :math:`\sigma = 1` case.

Moreover, the :math:`k = 0` basis function is broader than the :math:`k > 0`
functions, and :math:`\rho_k(r = 0) = 0` for all :math:`k > 0`, whereas
:math:`\rho_k(r \ne 0) \ne 0`. In other words, the region near the symmetry
axis is treated quite differently from the rest of the image, which leads to an
artifact near :math:`r = 0` in the reconstructed distributions.

Another problem arises when `Tikhonov regularization
<https://en.wikipedia.org/wiki/Tikhonov_regularization>`_ is applied. Since it
includes the norm of the solution in its minimization criterion, this generally
leads to some intensity drop in the reconstructed distributions, especially
near the symmetry axis.

In order to reduce these problems, PyAbel can use an automatic “intensity
correction”. It is based on the linearity of the transform and uses a
“calibration” distribution with a known analytical Abel transform.

Specifically, a flat distribution (with a soft edge, to avoid ringing artifacts
near the image boundary) and its analytical Abel transform are generated. Then
the BASEX transform with the desired parameters is applied to that Abel
transform, what should reconstruct the initial flat distribution, but actually
includes the artifacts described above. The ratio of the desired flat
distribution to this BASEX result is then taken as the intensity correction
profile and is applied to the BASEX transform of the actual data.

Although this correction procedure does not reproduce analytical results for
*all* distributions (except the calibration distribution itself), it greatly
reduces the method artifacts in most cases.


Vertical transform
------------------

(See `this discussion
<https://github.com/PyAbel/PyAbel/issues/225#issuecomment-421698132>`_ about
notation and details of the original implementation.)

Besides the horizontal transform that realizes the inverse Abel transform, the
BASEX article and the `BASEX.exe` program also apply a vertical transform to
the data. It is performed by multiplying the data by :math:`\mathbf B` in
equation (13) to obtain the expansion coefficients and then multiplying these
coefficients by :math:`\mathbf Z` in equation (9) to obtain the reconstructed
image.

However, regularization is never applied to the vertical transform
(:math:`q_2^2 = 0`), so when :math:`\mathbf Z` has full rank (:math:`\sigma =
1`, the “narrow” basis set in `BASEX.exe`), the overall vertical transform is

.. math::
    \mathbf{BZ} =
    \mathbf Z^{\mathrm T}\left(\mathbf{ZZ}^{\mathrm T}\right)^{-1} \mathbf Z =
    \mathbf I,

that is, an identity transform, having no effect on the final results.

When :math:`\mathbf Z` is not of full rank, for example, for the “broad” basis
set (:math:`\sigma = 2`), the transform is no longer an identity, but actually
has some undesirable properties.

First, it is not strictly translationally invariant (see the plot of the basis
functions above) and thus is in fact not applied by the `BASEX.exe` program
when “Line-by-line reconstruction” is chosen.

Second, far from the edges this transform is close to a convolution with the
following functions:

.. plot:: transform_methods/basex-vert.py

so, in addition to the possibly useful vertical smoothing, it also introduces
noticeable ringing artifacts.

Therefore in the PyAbel BASEX implementation we never apply the vertical
transform. If the vertical smoothing for :math:`\sigma > 1` is desirable, it
can be achieved by applying a vertical Gaussian blur to the transformed image.

The behavior of the original `BASEX.exe` program with top–bottom symmetry and
the “broad” basis set can be reproduced by replacing the line ::

    return rawdata.dot(A)

in :py:func:`abel.basex.basex_core_transform` with the following code::

    Mc = (_bs[1])[::-1]  # PyAbel and BASEX.exe use different coordinates
    V = Mc.dot(inv((Mc.T).dot(Mc))).dot(Mc.T)
    return V.dot(rawdata).dot(A)

and using the code example from BASEX/:ref:`BASEXhowto` with a additional
``sigma=2`` parameter in ``transform_options``.
