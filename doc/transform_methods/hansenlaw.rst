.. |nbsp| unicode:: 0xA0
   :trim:

Hansen–Law
==========


Introduction
------------

The Hansen and Law transform [1]_ [2]_ is a fast (linear time) Abel transform.

In their words, Hansen and Law [1]_ present:

*"... new family of algorithms, principally for Abel inversion, that are
recursive and hence computationally efficient. The methods are based on a
linear, space-variant, state-variable model of the Abel transform. The model
is the basis for deterministic algorithms."*

and [2]_:

*"... Abel transform, which maps an axisymmetric two-dimensional function into a line integral projection."*


The algorithm is efficient, one of the few methods to provide both the **forward** Abel and **inverse** Abel transform.


How it works
------------

.. comment:
   For ":align: right" figures, Sphinx uses LaTeX's wrapfig wrongly (not
   allowing floating), so the PDF results are horrible. Thus the figures here
   are inserted differently for "html" and "latex".
   (In "html", it also doesn't properly break long lines in captions, so this
   is done manually below...)

.. plot:: transform_methods/hansenlaw-proj.py
    :nofigs:

.. plot:: transform_methods/hansenlaw-recur.py
    :nofigs:

.. only:: html

   .. figure:: hansenlaw-proj.*
      :align: right

      Projection geometry (Fig. 1 [1]_)

For an axis-symmetric source image the projection of a source image,
:math:`g(R)`, is given by the forward Abel transform:

.. math:: g(R) = 2 \int_R^\infty \frac{f(r) r}{\sqrt{r^2 - R^2}} dr

The corresponding inverse Abel transform is

.. math:: f(r) = -\frac{1}{\pi}  \int_r^\infty \frac{g^\prime(R)}{\sqrt{R^2 - r^2}} dR

.. only:: latex

   .. list-table::

      * - .. figure:: hansenlaw-proj.*

              Projection geometry (Fig. 1 [1]_)

        - .. figure:: hansenlaw-recur.*

              Recursion: pixel value from adjacent outer-pixel

The Hansen and Law method makes a coordinate transformation to model the Abel transform as a set of linear differential equation, with the driving function
either the source image :math:`f(r)`,  for the forward transform, or the
projection image gradient :math:`g^\prime(R)`, for the inverse transform.
More detail is given :ref:`below <themath>`.

.. |br| raw:: html

   <br>

.. only:: html

   .. figure:: hansenlaw-recur.*
      :align: right

      Recursion: pixel value from |br| adjacent outer-pixel

Forward transform is

.. math::

  x_{n-1} &= \Phi_n x_n + B_{0n} f_n + B_{1n} f_{n-1}

  g_n &= \tilde{C} x_n,

where :math:`B_{1n}=0` for the zero-order hold approximation.

Inverse transform:

.. math::

  x_{n-1} &= \Phi_n x_n + B_{0n} g^\prime_n + B_{1n} g^\prime_{n-1}

  f_n &= \tilde{C} x_n


Note the only difference between the *forward* and *inverse* algorithms is
the exchange of :math:`f_n` with :math:`g^\prime_n` (or :math:`g_n`).

Details on the evaluation of :math:`\Phi, B_{0n},` and :math:`B_{1n}` are given :ref:`below <themath>`.

The algorithm iterates along each individual row of the image, starting at
the out edge, ending at the center-line. Since all rows in an image can be
processed simultaneously, the operation can be easily vectorized and is
therefore numerically efficient.


When to use it
--------------

The Hansen-Law algorithm offers one of the fastest, most robust methods for
both the forward and inverse transforms. It requires reasonably fine sampling
of the data to provide exact agreement with the analytical result, but otherwise
this method is a hidden gem of the field.


How to use it
-------------

To complete the forward or inverse transform of a full image with the
``hansenlaw method``, simply use the :class:`abel.Transform
<abel.transform.Transform>` class::

    abel.Transform(myImage, method='hansenlaw', direction='forward').transform
    abel.Transform(myImage, method='hansenlaw', direction='inverse').transform


If you would like to access the Hansen-Law algorithm directly (to transform a
right-side half-image), you can use :func:`abel.hansenlaw.hansenlaw_transform`.


Tips
----

`hansenlaw` tends to perform better with images of large size :math:`n > 1001` pixel width. For smaller images the angular_integration (speed) profile may look better if sub-pixel sampling is used::

    angular_integration_options=dict(dr=0.5)


Example
-------

.. plot:: ../examples/example_O2_PES_PAD.py

:doc:`Source code </example_O2_PES_PAD>`


Historical Note
---------------

The Hansen and Law algorithm was almost lost to the scientific community. It was
rediscovered by Jason Gascooke (Flinders University, South Australia) for use in
his velocity-map image analysis, and written up in his PhD thesis [3]_.

Eric Hansen provided guidance, algebra, and explanations, to aid the implementation of his first-order hold algorithm, described in Ref. [2]_ (April 2018).

.. _themath:

The Math
--------

The resulting state equations are, for the forward transform:

 .. math::

  x^\prime(r) = -\frac{1}{r} \tilde{A} x(r) + \frac{1}{\pi r} \tilde{B} f(R),

with inverse:

 .. math::

   x^\prime(R) = -\frac{1}{R} \tilde{A} x(R) - 2\tilde{B} f(R),

where :math:`[\tilde{A}, \tilde{B}, \tilde{C}]` realize the impulse response: :math:`\tilde{h}(t) = \tilde{C} \exp{(\tilde{A} t)}\tilde{B} = \left[1-e^{-2t}\right]^{-\frac{1}{2}}`, with

  .. math::

    \tilde{A} &= \rm{diag}[\lambda_1, \lambda_2, ..., \lambda_K]

    \tilde{B} &= [h_1, h_2, ..., h_K]^T

    \tilde{C} &= [1, 1, ..., 1]

The differential equations have the transform solutions, forward:

 .. math:: x(r) = \Phi(r, r_0) x(r_0) + 2 \int_{r_0}^{r} \Phi(r, \epsilon) \tilde{B} f(\epsilon) d\epsilon.

and inverse:

 .. math:: x(r) = \Phi(r, r_0) x(r_0) - \frac{1}{\pi} \int_{r_0}^{r} \frac{\Phi(r, \epsilon)}{r} \tilde{B} g^\prime(\epsilon) d\epsilon,


with :math:`\Phi(r, r_0) = \rm{diag}[(\frac{r_0}{r})^{\lambda_1}, ..., (\frac{r_0}{r})^{\lambda_K}] \equiv \rm{diag}[(\frac{n}{n-1})^{\lambda_1}, ..., (\frac{n}{n-1})^{\lambda_K}]`, where the integration limits :math:`(r, r_0)` extend across one grid interval or a pixel, so :math:`r_0 = n\Delta`, :math:`r = (n-1)\Delta`.

To evaluate the (superposition) integral, the driven part of the solution, the
driving function :math:`f(\epsilon)` or :math:`g^\prime(\epsilon)` is assumed to
either be constant across each grid interval, the **zero-order hold** approximation, :math:`f(\epsilon) \sim f(r_0)`, or linear, a **first-order hold** approximation, :math:`f(\epsilon) \sim p + q\epsilon = (r_0f(r) - rf(r_0))/\Delta + (f(r_0) - f(r))\epsilon/\Delta`. The integrand then separates into a sum over terms multiplied by :math:`h_k`,

 .. math::

    \sum_k h_k f(r_0) \int_{r_0}^{r} \Phi_k(r, \epsilon) d\epsilon

with each integral

 .. math::

  \int_{r_0}^{r} \left(\frac{\epsilon}{r}\right)^\lambda_k d\epsilon = \frac{r}{r_0}\left[ 1 - \left(\frac{r}{r_0}\right)^{\lambda_k + 1}\right] = \frac{(n-1)^a}{\lambda_k + a} \left[ 1 - \left(\frac{n}{n-1}\right)^{\lambda_k+a} \right],

where, the right-most-side of the equation has an additional parameter, :math:`a` to generalize the power of :math:`\lambda_k`.  For the inverse transform, there is an additional factor :math:`\frac{1}{\pi r}` in the state equation, and hence the integrand has :math:`\lambda_k` power, reduced by −1. While, for the
first-order hold approximation, the linear :math:`\epsilon` term increases :math:`\lambda_k` by +1.


Citation
--------

.. |ref1| replace:: \ E. W. Hansen, P.-L. Law, "Recursive methods for computing the Abel transform and its inverse", `J. Opt. Soc. Am. A 2, 510–520 (1985) <https://dx.doi.org/10.1364/JOSAA.2.000510>`__.

.. |ref2| replace:: \ E. W. Hansen, "Fast Hankel transform algorithm", `IEEE Trans. Acoust. Speech Signal Proc. 33, 666–671 (1985) <https://dx.doi.org/10.1109/TASSP.1985.1164579>`__

.. |ref3| replace:: \ J. R. Gascooke, PhD Thesis: "Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals Molecule Dissociation", Flinders University (2000), (`record <https://trove.nla.gov.au/version/41486301>`__, `PDF <https://raw.githubusercontent.com/PyAbel/abel_info/master/Gascooke_Thesis.pdf>`__).

.. [1] |ref1|

.. [2] |ref2|

.. [3] |ref3|

.. only:: latex

    * |ref1|
    * |ref2|
    * |ref3|
