.. _BASEX:

BASEX
=====


Introduction
------------

The BASEX (“basis set expansion”) Abel-transform method utilizes well-behaved functions (i.e., functions that have a known analytic Abel transform) to transform images.
In the current iteration of PyAbel, these functions (called basis functions) are Gaussian-like functions, following the original description of the method, developed in 2002 at USC and UC Irvine by Dribinski, Ossadtchi, Mandelshtam, and Reisler [Dribinski2002]_.


How it works
------------

This method is based on expressing line-of-sight projection images (``raw_data``) as sums of functions that have known analytic Abel inverses. The provided raw images are expanded in a basis set composed of these basis functions, with the expansion coefficients determined through a least-squares fitting process.
These coefficients are then applied to the (known) analytic inverse of these basis functions, which directly provides the Abel inverse of the raw images. Thus, the transform can be completed using simple linear algebra.

In the current iteration of PyAbel, these basis functions are Gaussian-like (see equations (14) and (15) in [Dribinski2002]_). The process of evaluating these functions is computationally intensive, and the basis-set generation process can take several seconds to minutes for larger images (larger than ~1000×1000 pixels). However, once calculated, these basis sets can be reused, and are therefore stored on disk and loaded quickly for future use.
The transform then proceeds very quickly, since each raw-image Abel inversion is a simple matrix multiplication.


When to use it
--------------

According to Dribinski et al., BASEX has several advantages:

1. For synthetic noise-free projections, BASEX reconstructs an essentially exact and artifact-free image, eschewing the need for interpolation procedures, which may introduce additional errors or assumptions.

2. BASEX is computationally cheap and only requires matrix multiplication, once the basis sets have been generated and saved to disk.

3. The current basis set is composed of the Gaussian-like functions, which are highly localized, uniform in coverage, and sufficiently narrow. This allows resolution of very sharp features in the raw data. Moreover, the reconstruction procedure does not contribute to noise in the reconstructed image; noise appears in the image only when it exists in the projection.

4. Resolution of images reconstructed with BASEX is superior to those obtained with the Fourier–Hankel method, particularly for noisy projections. However, to obtain maximal resolution, it is important to properly center the projections prior to transforming with BASEX.

5. BASEX-reconstructed images have an exact analytical expression, which allows an analytical high-resolution calculation of the speed distribution, without increasing computation time. (This is not yet implemented in PyAbel.)


.. _BASEXhowto:

How to use it
-------------

The recommended way to complete the inverse Abel transform using the BASEX algorithm for a full image is to use the :py:class:`abel.transform.Transform` class::

    abel.transform.Transform(raw_image, method='basex', direction='inverse').transform

The additional BASEX parameters are described in :py:func:`abel.basex.basex_transform` an can be passed to ``Transform()`` using the ``transform_options`` argument.

If you would like to access the BASEX algorithm directly (to transform a right-side half-image), you can use :py:func:`abel.basex.basex_transform`.

The behavior of the original `BASEX.exe` program by Karpichev with top–bottom symmetry and the “narrow” basis set can be reproduced as follows::

    rescale = math.sqrt(math.pi) / 2

    raw_image = <centered raw image>
    reg = <regularization parameter>
    reconst = abel.Transform(raw_image, direction='inverse', symmetry_axis=(0, 1),
                             method='basex', transform_options=dict(
                                 reg=reg*(rescale**2), correction=False
                            )).transform.clip(min=0) * rescale

(The ``rescale`` factor accounts for the wrong factor used in the `BASEX.exe` program for the basis projections, see :ref:`BASEXcomp`.)


PyAbel improvements
-------------------

* As noted above, the BASEX method implementation in PyAbel uses correct expressions for the basis projections, so unlike `BASEX.exe`, it is consistent with the original method description in [Dribinski2002]_ and with other methods implemented in PyAbel.

* Basis sets for any image size are generated automatically.

* Basis functions with any width parameter :math:`\sigma` (specified by the ``sigma`` parameter) can be used. They are :math:`\rho_k(r) \approx \exp[-2(r/\sigma - k)^2]`, so their :math:`1/e^2` width is :math:`2\sigma`, and the full width at half-maximum (FWHM) is :math:`\sqrt{2 \ln 2}\,\sigma \approx 1.18\,\sigma`. The spacing between the maxima of the adjacent basis functions is :math:`\sigma`, which automatically determines the number of basis functions.

* An automatic intensity correction is available (enabled by default) for reducing the artifacts caused by the basis-functions shape and the sampling of their projections, as well as the intensity drop (especially near the axis) introduced by Tikhonov regularization.

* The forward Abel transform is also implemented, using the same method but swapping the basis functions and their projections.


Citation
--------
.. [Dribinski2002] `Dribinski et al, 2002 (Rev. Sci. Instrum. 73, 2634) <http://dx.doi.org/10.1063/1.1482156>`_, (`pdf <http://www-bcf.usc.edu/~reisler/assets/pdf/67.pdf>`_)
