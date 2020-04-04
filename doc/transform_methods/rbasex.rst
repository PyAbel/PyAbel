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

In velocity-map imaging with cylindrically symmetric photodissociation (in a
broad sense, including photoionization and photodetachment) the 3D velocity
distribution at each speed (3D radius) consists of a finite number of spherical
harmonics :math:`Y_{nm}(\theta, \varphi)` with :math:`m = 0`, which are also
representable as Legendre polynomials :math:`P_n(\cos\theta)`. This means that
an :math:`N \times N` image has only :math:`N_r \times N_a` degrees of freedom,
where :math:`N_r` is the number of radial samples, usually :math:`N / 2`, and
:math:`N_a` is the number of angular terms, a small number depending on the
number of photons. These are the “radial distribution” extracted from the
transformed image in other, general Abel-inversion methods.

However, if these radial distributions are considered as a basis, the 3D
distribution can be represented as a linear combination of these basis
functions with some coefficients. And the corresponding image, being the
forward Abel transform of the 3D distribution, will be represented as a linear
combination of basis-function projections, that is, their forward Abel
transforms, with the same coefficients. The reverse is also true: finding the
expansion coefficients of an experimental VM image over the projected basis
directly gives the expansion coefficients of the initial 3D velocity direction
and thus the sought radial distributions.

Finding the expansion coefficients is a simple linear problem, and the forward
Abel transforms of the basis functions can be calculated easily if the basis is
chosen wisely.

See :ref:`rBasexmath` for the complete description.

.. toctree::
    :hidden:

    rbasex-math


Differences from pBasex
^^^^^^^^^^^^^^^^^^^^^^^

1. Triangular radial basis functions are used instead of Gaussians. They are
   more compact/orthogonal (only the adjacent functions overlap) and have
   analytical Abel transforms.

2. Cosine powers are used instead of Legendre polynomials for angular basis
   functions. This makes the projected basis functions also separable into
   radial and angular parts.

3. The basis separability allows decomposition of the problem in two steps:
   first, radial distributions are extracted from the image (without
   intermediate rebinning to polar grid, what is faster and avoids accumulation
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

Although this method can be applied only to velocity-map images, and not just
cylindrically symmetric images, it has several benefits in this special case.

1. The reconstructed radial distributions, which are of the primary interest in
   VMI studies, are obtained directly.

2. Limitations on the angular behavior of the distribution also put strong
   constraints on the reconstruction noise, making the reconstructed images
   much cleaner.

3. Unlike general Abel transform methods, which have time complexity with
   *cubic* dependence on the image size, this method is only *quadratic*, once
   the transform matrix is computed. Computing the transform matrix is still
   cubic, but after it is done, transforming a series of images is faster,
   especially for large images.

4. The optional non-negativity constraints implemented in this method allow
   obtaining physically meaningful intensity and anisotropy distributions. They
   can also help in denoising experimental images with very low event counts.


How to use it
-------------

The method can be used by directly calling its transform function::

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

Note that this method does not require the input image to be centered. Thus
instead of centering it with :func:`~abel.tools.center.center_image`, which
will crop some data or fill it with zeros, it is better to pass the image
origin directly to the transform function, determining it automatically, if
needed::

    origin = abel.tools.center.find_origin(image, method='convolution')
    recon, distr = abel.rbasex.rbasex_transform(image, origin=origin)

See :func:`abel.rbasex.rbasex_transform` documentation for the full description
of all available transform parameters.

Alternatively, the method can be accessed through the universal
:class:`Transform <abel.transform.Transform>` class::

    res = abel.Transform(image, method='rbasex')
    recon = res.transform
    distr = res.distr

passing additional rBasex parameters through the ``transform_options``
argument. However, keep in mind that if you want to use all the data from an
off-center image, do not use the ``origin`` argument of :class:`Transform
<abel.transform.Transform>`, but pass it inside ``transform_options``. This
also *must* be done if optional pixel weighting is used, since otherwise
:class:`Transform <abel.transform.Transform>` will shift the image, but not the
weights array.

The weights array can be used as a mask, using zero weights to exclude unwanted
pixels, as demonstrated in :doc:`../example_rbasex_block`. In practice, instead
of defining the mask geometry in the code, it might be more convenient to save
the analyzed data as an image file::

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

This method has not yet been published elsewhere, so cite it as the “rBasex
method from the PyAbel package”, using the current Zenodo DOI (see :ref:`README
<READMEcitation>` for details).

.. [1] \ G. A. Garcia, L. Nahon, I. Powis,
       “Two-dimensional charged particle image inversion using a polar basis
       function expansion”,
       `Rev. Sci. Instrum. 75, 4989–4996 (2004)
       <https://doi.org/10.1063/1.1807578>`_.

.. [2] \ M. Ryazanov,
       “Development and implementation of methods for sliced velocity map
       imaging. Studies of overtone-induced dissociation and isomerization
       dynamics of hydroxymethyl radical (CH\ :sub:`2`\ OH and
       CD\ :sub:`2`\ OH)”,
       Ph.D. dissertation, University of Southern California, 2012.
       (`ProQuest <https://search.proquest.com/docview/1289069738>`_,
       `USC <https://digitallibrary.usc.edu/cdm/ref/collection/p15799coll3/id/
       112619>`_).
