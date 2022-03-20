.. _rBasex:

rBasex
======


Introduction
------------

This method resembles the pBasex [1]_ approach of expanding a velocity-map
image over a 2D basis set in polar coordinates, but uses more convenient basis
functions with analytical Abel transforms, developed by M. Ryazanov [2]_.


How it works
------------

In velocity-map imaging (VMI) with cylindrically symmetric photodissociation
(in a broad sense, including photoionization and photodetachment) the 3D
velocity distribution at each speed (3D radius) consists of a finite number of
spherical harmonics :math:`Y_{nm}(\theta, \varphi)` with :math:`m = 0`, which
are also representable as Legendre polynomials :math:`P_n(\cos\theta)`. This
means that an :math:`N \times N` image has only :math:`N_r \times N_a` degrees
of freedom, where :math:`N_r` is the number of radial samples, usually :math:`N
/ 2`, and :math:`N_a` is the number of angular terms, a small number depending
on the studied process. These degrees of freedom correspond to the “radial
distribution” extracted from the transformed image in other, general
Abel-inversion methods.

However, if these radial distributions are considered as a basis, the 3D
distribution can be represented as a linear combination of these basis
functions with some coefficients. And the corresponding image, being the
forward Abel transform of the 3D distribution, will be represented as a linear
combination of basis-function projections, that is, their forward Abel
transforms, with the same coefficients. The reverse is also true: finding the
expansion coefficients of an experimental velocity-map image over the projected
basis directly gives the expansion coefficients of the initial 3D velocity
direction and thus the sought radial distributions.

Finding the expansion coefficients is a simple linear problem, and the forward
Abel transforms of the basis functions can be calculated easily if the basis is
chosen wisely.

See :ref:`rBasexmath` for the complete description.


Differences from pBasex
^^^^^^^^^^^^^^^^^^^^^^^

While rBasex is similar to pBasex in the idea of using VMI-oriented 3D basis
functions, it has several key differences:

1. Triangular radial basis functions are used instead of Gaussians. They are
   more compact/orthogonal (only the adjacent functions overlap) and have
   analytical Abel transforms.

2. Cosine powers are used instead of Legendre polynomials for angular basis
   functions. This makes the projected basis functions also separable into
   radial and angular parts.

3. The basis separability allows decomposition of the problem in two steps:
   first, radial distributions are extracted from the image (without
   intermediate rebinning to polar grid, thus faster and avoiding accumulation
   of resampling errors); second, these radial distributions are expanded over
   radial bases for each angular order. This eliminates the necessity to work
   with large matrices.

4. Custom pixel weighting can be used, for example, to exclude image areas
   “damaged” in some way (obscured by a beam block, contaminated by parasitic
   signals, affected by detector imperfections and so on). Partial images (not
   including the whole angular range) can be reconstructed as well.

5. The forward Abel transform is implemented in addition to the inverse
   transform.

6. Additional (better) regularization methods are implemented.


Differences from the reconstruction method described in [2]_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many ideas used in rBasex, including the analytically transformable basis
functions, are taken from the previous work [2]_, but with some omissions,
additions and modifications.

1. Instead of working with individual pixels and weighting them according to
   Poisson statistics, the binned radial distributions (not weighted by
   default) are transformed. This is less accurate, but much faster, especially
   in Python.

2. Slicing is not implemented.

3. Only the non-negativity constraints are implemented. However, several
   linear regularization options are added.

4. Odd angular orders can be included.


When to use it
--------------

This method makes additional assumptions (beyond cylindrical symmetry) about
the data, so it can be applied only to velocity-map images or in other similar
situations involving a finite number of spherical harmonics. However, in this
special case, it offers several benefits:

1. The reconstructed radial distributions, which are often the primary interest
   in VMI studies, are obtained directly.

2. Limitations on the angular behavior of the distribution also put strong
   constraints on the reconstruction noise, making the reconstructed images
   much cleaner.

3. Several optional :ref:`regularization <rBasexmathregex>` methods help to
   further reduce noise in reconstructed images, especially near the center.
   Regularization strengths can be adjusted to produce a desirable balance
   between noise reduction and blurring of sharp features.

4. Unlike general Abel-transform methods, which have time complexity with
   *cubic* dependence on the image size, this method is only *quadratic*, once
   the transform matrix is computed. Computing the transform matrix is still
   cubic, but after it is done, transforming a series of images is faster,
   especially for large images.

5. The optional non-negativity constraints implemented in this method allow
   obtaining physically meaningful intensity and anisotropy distributions. They
   can also help in denoising experimental images with very low event counts.


How to use it
-------------

The method can be accessed through the universal :class:`abel.Transform
<abel.transform.Transform>` class::

    res = abel.Transform(image, method='rbasex')
    recon = res.transform
    distr = res.distr

optionally using other :class:`Transform <abel.transform.Transform>` arguments
and passing additional rBasex parameters (see
:func:`abel.rbasex.rbasex_transform` documentation for their full description)
through the ``transform_options`` argument. Alternatively, it might be more
convenient to use the method by calling its transform function directly::

    recon, distr = abel.rbasex.rbasex_transform(image)
    r, I, beta = distr.rIbeta()

It returns the transformed image ``recon`` and a :class:`Distributions.Results
<abel.tools.vmi.Distributions.Results>` object ``distr``, from which various
radial distributions can be retrieved, such as the intensity and
anisotropy-parameter distributions in this example.

If only the distributions are needed, but not the transformed image itself, the
calculations can be accelerated by disabling the creation of the output image::

    _, distr = abel.rbasex.rbasex_transform(image, out=None)
    r, I, beta = distr.rIbeta()

Note that rBasex does not require the input image to be centered. Thus instead
of centering it with :func:`~abel.tools.center.center_image` (or using the
``origin`` argument of :class:`Transform <abel.transform.Transform>`), which
will crop some data or fill it with zeros, it is better to pass the image
origin directly to the transform function, determining it automatically, if
needed::

    origin = abel.tools.center.find_origin(image, method='convolution')
    recon, distr = abel.rbasex.rbasex_transform(image, origin=origin)

This also *must* be done if optional pixel weighting is used, since otherwise
the centered image would become inconsistent with the weights array. For
example, when using the :class:`Transform <abel.transform.Transform>` class,
pass the origin as follows::

    res = abel.Transform(image, method='rbasex',
                         transform_options=dict(origin=..., weights=...))

The weights array can also be used as a mask, using zero weights to exclude
unwanted pixels, as demonstrated in :doc:`../example_rbasex_block`. In
practice, instead of defining the mask geometry in the code, it might be more
convenient to save the analyzed data as an image file::

    # save as an RGB image using a chosen colormap
    plt.imsave('imagemask.png', image, cmap='hot')

then open it in any raster graphics editor, paint the areas to be excluded with
some distinct color (for example, blue in case of ``cmap='hot'``) and save it.
This painted image then can be loaded in the program, and the mask is easily
extracted from it::

    # read as an array with R, G, B (or R, G, B, A) components
    mask = plt.imread('imagemask.png')
    # set zero weights for pixels with blue channel (2) > red channel (0)
    # and unit weights for other pixels
    weights = 1.0 - (mask[..., 2] > mask[..., 0])

(for other image colormaps and mask colors, adapt the comparison logic
accordingly). These weights then can be used in the transform of the original
data, as well as any other data having the same mask geometry.


Citation
--------

This method has not yet been published elsewhere, so please cite it as the
“rBasex method from the PyAbel package”, using the current Zenodo DOI (see
:ref:`README <READMEcitation>` for details).

.. |ref1| replace:: \ G. A. Garcia, L. Nahon, I. Powis,
       “Two-dimensional charged particle image inversion using a polar basis
       function expansion”,
       `Rev. Sci. Instrum. 75, 4989–4996 (2004)
       <https://doi.org/10.1063/1.1807578>`__.

.. |ref2| replace:: \ M. Ryazanov,
       “Development and implementation of methods for sliced velocity map
       imaging. Studies of overtone-induced dissociation and isomerization
       dynamics of hydroxymethyl radical (CH\ :sub:`2`\ OH and
       CD\ :sub:`2`\ OH)”,
       Ph.D. dissertation, University of Southern California, 2012.
       (`ProQuest <https://www.proquest.com/docview/1289069738>`__,
       `USC <https://digitallibrary.usc.edu/asset-management/2A3BF169XWB4>`__).

.. [1] |ref1|

.. [2] |ref2|

.. only:: latex

    * |ref1|
    * |ref2|


.. toctree::
    :hidden:

    rbasex-math
