.. _Daun:

Daun
====


Introduction
------------

This suite of methods is based on the deconvolution procedure with Tikhonov
regularization described by Daun at al. [1]_ and extends it with additional,
smoother, approximations and regularization types.


How it works
------------

The original method formulates the numerical Abel transform in a form
equivalent to the “onion peeling” method, where the original distribution is
approximated with a `step function
<https://en.wikipedia.org/wiki/Step_function>`__ (piecewise constant function
with 1-pixel-wide radial intervals). The forward transform thus can be
described by a system on linear equations in a matrix form

.. math::
    \mathbf A_\text{OP} \mathbf x = \mathbf b,
    :label: forward

where the vector :math:`\mathbf x` consists of the original distribution values
sampled at a uniform radial grid :math:`r_i = i \Delta r`, the vector
:math:`\mathbf b` consists of the projection values sampled at the
corresponding uniform grid :math:`y_i = i \Delta r`, and the matrix
:math:`\mathbf A_\text{OP}` corresponds to the “onion peeling” forward Abel
transform. Its elements are

.. math::
    A_{\text{OP}, ij} = \begin{cases}
        0, & j < i, \\
        2 \Delta r \big[(j + 1/2)^2 - i^2\big]^{1/2}, & j = i, \\
        2 \Delta r \Big(\big[(j + 1/2)^2 - i^2\big]^{1/2} -
                        \big[(j - 1/2)^2 - i^2\big]^{1/2}\Big), & j > i
    \end{cases}

and represent contributions of each distribution interval to each projection
interval.

However, instead of performing the inverse transform by using the inverse
matrix directly:

.. math::
    \mathbf x = \mathbf A_\text{OP}^{-1} \mathbf b,

as it is done in the onion peeling method, the equation :eq:`forward` is solved
by applying `Tikhonov regularization
<https://en.wikipedia.org/wiki/Tikhonov_regularization>`__:

.. math::
    \tilde{\mathbf x} = \operatorname{arg\,min}_{\mathbf x} \left(
        \|\mathbf A_\text{OP} \mathbf x - \mathbf b\|^2 +
        \alpha \|\mathbf L \mathbf x\|^2
    \right),
    :label: minimization

where :math:`\alpha` is the regularization parameter, and the finite-difference
matrix

.. math::
    \mathbf L = \begin{bmatrix}
        -1     &  1     & 0      & \cdots & 0      \\
        0      & -1     & 1      & \ddots & \vdots \\
        \vdots & \ddots & \ddots & \ddots & \vdots \\
        0      & \cdots & 0      & -1     & 1
    \end{bmatrix}

(approximation of the derivative operator) is used as the Tikhonov matrix. The
idea is that the admixture of the derivative norm to the minimization problem
leads to a smoother solution. The regularization parameter :math:`\alpha`
controls how much attention is paid to the derivative: when :math:`\alpha = 0`,
the exact solution to :eq:`forward` is obtained, even if very noisy; when
:math:`\alpha \to \infty`, the solution becomes very smooth, even if
reproducing the data poorly. A reasonably chosen value of :math:`\alpha` can
result in a significant suppression of high-frequency noise without noticeably
affecting the signal.

The minimization problem :eq:`minimization` leads again to a linear matrix
equation, and the regularized inverse transform is obtained by using the
regularized matrix inverse:

.. math::
    \tilde{\mathbf x} = \mathbf A_\text{Tik}^{-1} \mathbf b_\text{Tik},
    \quad
    A_\text{Tik} = (\mathbf A^T \mathbf A +
                    \alpha \mathbf A \mathbf L^T \mathbf L),
    \quad
    \mathbf b_\text{Tik} = A^T \mathbf b.
    :label: tikhonov

(Note: here :math:`\mathbf x` and :math:`\mathbf b` are column vectors, but in
PyAbel they are row vectors corresponding to image rows, so all the equations
are transposed; moreover, instead of processing row vectors separately, they
are transformed as the image matrix at once.)


PyAbel additions
----------------

Basis sets
^^^^^^^^^^

The step-function approximation used in the original method implies a basis set
consisting of rectangular functions

.. math::
    f_i(r) = \begin{cases}
        1, & r \in [i - 1/2, i + 1/2], \\
        0  & \text{otherwise}.
    \end{cases}

This approximation can be considered rather coarse, so in addition to these
zero-order piecewise polynomials we also implement basis sets consisting of
piecewise polynomials up to 3rd order. An example of a test function composed
of broad and narrow Gaussian peaks and its approximations of various orders is
shown below:

.. plot:: transform_methods/daun-basis.py
    :align: center

Here the solid black line is the test function, and the dashed black line is
its approximation of order :math:`n`, equal to the sum of the colored basis
functions.

order = 0:
    Rectangular functions produce a stepwise approximation. This is the only
    approach mentioned in the original article and corresponds to the usual
    “onion peeling” transform.
order = 1:
    `Triangular functions
    <https://en.wikipedia.org/wiki/Triangular_function>`__ produce a continuous
    piecewise linear approximation. Besides being continuous (although not
    smooth), this also corresponds to how numerical data is usually plotted
    (with points connected by straight lines), so such plots would faithfully
    convey the underlying method assumptions.
order = 2:
    Piecewise quadratic functions

    .. math::
        f_i(r) = \begin{cases}
            2[r - (i - 1)]^2, & r \in [i - 1,   i - 1/2], \\
            1 - 2[r - i]^2,   & r \in [i - 1/2, i + 1/2], \\
            2[r - (i + 1)]^2, & r \in [i + 1/2, i + 1], \\
            0                 & \text{otherwise}.
        \end{cases}

    produce a smooth piecewise quadratic approximation. While resembling
    :ref:`BASEX basis functions <BASEXcomp>` in shape, these are localized
    within ±1 pixel, sum to unity (although produce oscillations on slopes),
    and their projections are much faster to compute.
order = 3:
    Combinations of `cubic Hermite basis functions
    <https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation_on_a_single_interval>`__
    produce a cubic-spline approximation (with endpoint derivatives clamped to
    zero for 2D smoothness). Offers the most accurate representation for
    sufficiently smooth distributions, but produces ringing artifacts around
    sharp features, which can result in negative interpolated intensities even
    for non-negative data points.

(The projections of all these basis functions are calculated as described in
:ref:`Polynomials`.)

In practice, however, the choice of the basis set has negligible effect on the
transform results, as can be seen from an example :ref:`below
<example_orders>`.


Regularization methods
^^^^^^^^^^^^^^^^^^^^^^

:math:`L_2` norm
""""""""""""""""

In addition to the original derivative (difference) Tikhonov regularization,
PyAbel also implements the usual :math:`L_2` regularization, as in
:ref:`BASEX`, with the identity matrix :math:`\mathbf I` used instead of
:math:`\mathbf L` in :eq:`tikhonov`. The results are practically identical to
the BASEX method, especially with **order** = 2, except that the basis set is
computed much faster.

Non-negativity
""""""""""""""

A more substantial addition is the implementation of the non-negativity
regularization. Namely, instead of solving the unconstrained quadratic problem
:eq:`minimization`, non-negativity constraints are imposed on the original
problem:

.. math::
    \tilde{\mathbf x} = \operatorname{arg\,min}_{\mathbf x \geqslant 0}
        \|\mathbf A \mathbf x - \mathbf b\|^2.

This `non-negative least-squares
<https://en.wikipedia.org/wiki/Non-negative_least_squares>`__ solution yields
the distribution without negative intensities that reproduces the input data as
good as possible. In situations where the distribution must be non-negative,
this is the best physically meaningful solution.

The noise-filtering properties of this method come from the fact that noise in
the inverse Abel transform is strongly oscillating, so if negative-going spikes
are forbidden in the solution, the positive-going spikes must also be reduced
in order to preserve the overall intensity. Thus the method is most beneficial
for very noisy images, for which linear methods produce a large amount of noise
reaching negative values. For clean images of non-negative distributions, the
constrained solution exactly matches the solution of the original problem
:eq:`forward`. And unlike Tikhonov regularization, it does not blur legitimate
sharp features in any case.

Notice that constrained quadratic minimization remains a *non-linear* problem.
This has two important implications. First, it is much more computationally
challenging, so that transforming a megapixel image takes many seconds instead
of several milliseconds (and depends on the image itself). Second, the average
of transformed images is generally not equal to the transform of the averaged
image. It is thus recommended to perform as much averaging (image
symmetrization and summation of multiple images if applicable) as possible
before applying the transform. In particular, using ``symmetry_axis=(0, 1)`` in
:class:`abel.transform.Transform` would in fact require transforming only one
quadrant, which is 4 times faster that transforming the whole image.


When to use it
--------------

This method with default parameters (0th order, 0 regularization parameter) is
identical to the :doc:`“onion peeling” <onion_peeling>` method, but can also be
used for the forward transform.

The original (derivative/difference) Tikhonov regularization with non-zero
regularization parameter helps to remove high-frequency oscillations from the
transformed image. However, an excessively large regularization parameter can
lead to oversmoothing and broadening of the useful signal and under/overshoots
around sharp features. As recommended by Daun et al., by systematically
adjusting the heuristic regularization parameter, the analyst can find a
solution that represents an acceptable compromise between accuracy and
regularity.

The :math:`L_2` Tikhonov regularization approach is equivalent to that in the
:ref:`BASEX` method and has the same use cases and [dis]advantages.

The non-negativity regularization is recommended for very noisy images and
images with sharp features without a broad background. However, due to its
slowness, it cannot be used for real-time data processing.


How to use it
-------------

The inverse Abel transform of a full image can be done with the
:class:`abel.Transform <abel.transform.Transform>` class::

    abel.Transform(myImage, method='daun').transform

For the forward Abel transform, simply add :attr:`direction='forward'`::

    abel.Transform(myImage, method='daun', direction='forward').transform

Additional parameters can be passed through the :attr:`transform_options`
parameter. For example, to use the original regularization method with the
regularization parameter set to 100::

    abel.Transform(myImage, method='daun',
                   transform_options=dict{reg=100}).transform

The :math:`L_2` regularization can be applied using ::

    abel.Transform(myImage, method='daun',
                   transform_options=dict{reg=('L2', 100)}).transform

And the non-negative solution is obtained by ::

    abel.Transform(myImage, method='daun',
                   transform_options=dict{reg='nonneg'}).transform

In this case, it is recommended to use symmetrization::

    abel.Transform(myImage, method='daun',
                   symmetry_axis=0,  # or symmetry_axis=(0, 1) if applicable
                   transform_options=dict{reg='nonneg'}).transform

unless independent analysis of all image parts is desired.

The algorithm can be also accessed directly (to transform a right-side
half-image or properly oriented quadrants) through the
:func:`abel.daun.daun_transform()` function.

.. note::
    If you use any non-default options (order, regularization), please cite not
    only the article by Daun et al. and the PyAbel article, but also *this
    PyAbel release* |zenodo|, because these capabilities are not present in the
    original work by Daun et al. and were added to PyAbel after the RSI
    publication.

.. |zenodo| image:: https://zenodo.org/badge/30170345.svg
    :target: https://zenodo.org/badge/latestdoi/30170345


Examples
--------

Performance of various regularization methods for the Dribinski sample image
with added Poissonian noise:

.. plot:: ../examples/example_daun_reg.py

:doc:`(source code) </example_daun_reg>`

.. _example_orders:

The order of basis-set polynomials has almost no effect on the results (shown
here for :attr:`reg=0`):

.. plot:: ../examples/example_daun_order.py

:doc:`(source code) </example_daun_order>`


Citation
--------

.. [1] \ K. J. Daun, K. A. Thomson, F. Liu, G. J. Smallwood,
       “Deconvolution of axisymmetric flame properties using Tikhonov
       regularization”,
       `Appl. Opt. 45, 4638–4646 (2006)
       <https://doi.org/10.1364/AO.45.004638>`_.
