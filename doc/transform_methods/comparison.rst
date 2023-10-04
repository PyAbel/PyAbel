Comparison of Abel Transform Methods
====================================

Abstract
--------

This document provides a comparison of the quality and efficiency of the various Abel transform methods that are implemented in PyAbel. Some of the information presented here is adapted from [hickstein2019]_.

Introduction
------------

The projection of a three-dimensional (3D) object onto a two-dimensional (2D) surface takes place in many measurement processes; a simple example is the recording of an X-ray image of a soup bowl, donut, egg, wineglass, or other cylindrically symmetric object :numref:`fig_overview`, where the axis of cylindrical symmetry is parallel to the plane of the detector. Such a projection is an example of a *forward* Abel transform and occurs in numerous experiments, including photoelectron/photoion spectroscopy ([dribinski2002]_, [bordas1996]_, [chandler1987]_) the studies of plasma plumes ([glasser1978]_), flames ([deiluliis1998]_, [cignoli2001]_, [snelling1999]_, [das2017]_), and solar occulation of planetary atmospheres ([gladstone2016]_, [lumpe2007]_). The analysis of data from these experiments requires the use of the *inverse* Abel transform to recover the 3D object from its 2D projection.

.. _fig_overview:
.. figure:: ../overview.*
   :figclass: align-center

   The forward Abel transform maps a cylindrically symmetric three-dimensional (3D) object to its two-dimensional (2D) projection, a physical process that occurs in many experimental situations. For example, an X-ray image of the object on the left would produce the projection shown on the right. The *inverse* Abel transform takes the 2D projection and mathematically reconstructs the 3D object. As indicated by the Abel transform equations (below), the 3D object is described in terms of :math:`(r, z)` coordinates, while the 2D projection is recorded in :math:`(y, z)` coordinates.


While the forward and inverse Abel transforms may be written as simple, analytical expressions, attempts to naively evaluate them numerically for experimental images does not yield reliable results [whitaker2003]_. Consequently, many numerical methods have been developed to provide approximate solutions to the Abel transform (for example: [dribinski2002]_, [bordas1996]_, [chandler1987]_, [dasch1992]_, [rallis2014]_, [gerber2013]_, [harrison2018]_, [demicheli2017]_, [dick2014]_). Each method was created with specific goals in mind, with some taking advantage of pre-existing knowledge about the shape of the object, some prioritizing robustness to noise, and others offering enhanced computational efficiency. Each algorithm was originally implemented with somewhat different mathematical conventions and with often conflicting requirements for the size and format of the input data. Fortunately, PyAbel provides a consistent interface for the Abel-transform methods via the Python programming language, which allows for a straightforward, quantitative comparison of the output.

The following sections present several comparisons of the quality and speed of the various Abel-transform algorithms presented in PyAbel. In general, all of the methods provide reasonably quality results, with some methods providing options for additional smoothing of the data. However, some methods are orders-of-magnitude more efficient than others.


Math
----

The **forward Abel transform** is given by

.. math:: F(y,z) = 2 \int_y^{\infty} \frac{f(r,z)\,r}{\sqrt{r^2-y^2}}\,dr,


where :math:`y`, :math:`r`, and :math:`z` are the spatial coordinates as shown in :numref:`fig_overview`, :math:`f(r, z)` is the density of the 3D object at :math:`(r, z)`, and :math:`F(y, z)` is the intensity of the projection in the 2D plane.

The **inverse Abel transform** is given by

.. math:: f(r,z) = -\frac{1}{\pi} \int_r^{\infty} \frac{dF(y,z)}{dy}\, \frac{1}{\sqrt{y^2-r^2}}\,dy.

While the transform equations can be evaluated analytically for some mathematical functions, experiments typically generate discrete data (e.g., images collected with a digital camera), which must be evaluated numerically. Several issues arise when attempting to evaluate the Abel transform numerically. First, the simplest computational interpretation of inverse Abel transform equation involves three loops: over :math:`z`, :math:`r`, and :math:`y`, respectively. Such nested loops can be computationally expensive. Additionally, :math:`y = r` presents a singularity where the denominator goes to zero and the integrand goes to infinity. Finally, a simple approach requires a large number of sampling points in order to provide an accurate transform. Indeed, a simple numerical integration of the above equations has been shown to provide unreliable results [whitaker2003]_.

Various algorithms have been developed to address these issues. PyAbel incorporates numerous algorithms for the inverse Abel transform, and some of these algorithms also support the forward Abel transform. The following comparisons focus on the results of the inverse Abel transform, because it is the inverse Abel transform that is used most frequently to interpret experimental data.

Note that the forward and inverse Abel transforms are defined on the whole space, with infinite integration limits, but in reality, experimental data are limited to finite ranges of :math:`r` or :math:`y`. Thus the intensity distributions :math:`f` and :math:`F` must be zero outside these ranges, otherwise the transforms cannot be performed correctly. In other words, only localized objects can be transformed, and the object must be contained entirely within the image frame. If the image has any background, it must be subtracted before applying the transform, such that the image intensity goes to zero near the edge (however, the :doc:`direct` and :doc:`hansenlaw` methods effectively disregard a constant background).


List of Abel-Transform Methods in PyAbel
----------------------------------------

Below is a list that describes the basic approach and characteristics of all the Abel-transform algorithms implemented in PyAbel. The title of each algorithm is the keyword that can be passed to the ``method`` argument in :meth:`abel.transform.Transform`. Algorithms that pre-compute matrices for a specific image size, and (optionally) save them to disk for subsequent reuse, are indicated with the letter S. All methods implement the inverse Abel transform, while methods that also implement a forward transform are indicated with an F.

- ``basex`` (F, S) -- The "BAsis Set EXpansion" (BASEX) method of Dribinski and co-workers [dribinski2002]_ uses a basis set of Gaussian-like functions. This is one of the *de facto* standard methods in photoelectron/photoion spectroscopy [whitaker2003]_ and is highly recommended for general-purpose Abel transforms. The number of basis functions and their width can be varied. However, following the basis set provided with the original BASEX.exe program, by default the ``basex`` algorithm use a basis set where the full width at :math:`1/e^2` of the maximum is equal to 2 pixels and the basis functions are located at each pixel. Thus, the resolution of the image is roughly maintained. The ``basex`` algorithms allows a "Tikhonov regularization" to be applied, which suppresses intensity oscillations, producing a less noisy image. In the experimental comparisons presented below, the Tikhonov regularization factor is set to 200, which provides reasonable suppression of noise while still preserving the fine features in the image. See :doc:`basex` and :meth:`abel.basex.basex_transform`.

- ``onion_peeling`` (S) -- This method, and the following two methods (``three_point``, ``two_point``), are adapted from the 1992 paper by Dasch [dasch1992]_. All of these methods reduce the core Abel transform to a simple matrix-algebra operation, which allows a computationally efficient transform. Dasch emphasizes that these techniques work best in cases where the difference between adjacent points is much greater than the noise in the projections (i.e., where the raw data is not oversampled). This "onion-peeling deconvolution" method is one of the simpler and faster inverse Abel-transform methods. See :doc:`onion_peeling` and :meth:`abel.dasch.onion_peeling_transform`.

- ``three_point`` (S) -- This "three-point" algorithm [dasch1992]_ provides slightly more smoothing than the similar ``two_point`` or ``onion_peeling`` methods. The name refers to the fact that three neighboring pixels are considered, which improves the accuracy of the method for transforming smooth functions, as well as reducing the noise in the transformed image. The trade-off is that the ability of the method to transform very sharp (single-pixel) features is reduced. This is an excellent general-purpose algorithm for the Abel transform. See :doc:`three_point` and :meth:`abel.dasch.three_point_transform`.

- ``two_point`` (S) -- The "two-point method" (also described by Dasch [dasch1992]_) is a simplified version of the ``three_point`` algorithm and provides similar transform speeds. Since it only considers two adjacent points in the function, it allows sharper features to be transformed than the ``three_point`` method, but does not offer as much noise suppression. This method is also appropriate for most Abel transforms.  See :doc:`two_point` and :meth:`abel.dasch.two_point_transform`.

- ``direct`` (F) -- The "direct" algorithms [yurchak2015]_ uses a simple numerical integration, which closely resembles the basic Abel-transform equations (above). If the ``direct`` algorithm is used in its most naive form, the agreement with analytical solutions is poor, due to the singularity in the integral when :math:`r=y`. However, a correction can be applied, where the function is assumed to be piecewise-linear across the pixel where this condition is met. This simple approximation allows a reasonably accurate transform to be completed. Fundamentally, the ``direct`` algorithm requires that the input function be finely sampled to achieve good results. PyAbel incorporates two implementations of the ``direct`` algorithm, which produce identical results, but with different calculation speeds. The ``direct_Python`` implementation is written in pure Python, for easy interpretation and modification. The ``direct_C`` implementation is written in `Cython <https://cython.org/>`_, a Python-like language that is converted to C and compiled, providing higher computational efficiency. This method is included mainly for educational and comparison purposes. In most cases, other methods will provide more reliable results and higher computational efficiency.  See :doc:`direct` and :meth:`abel.direct.direct_transform`.

- ``hansenlaw`` (F) -- The recursive method of Hansen and Law ([hansen1985]_, [hansen1985b]_, [gascooke2000]_) interprets the Abel transform as a linear space-variant state-variable equation, to provide a reliable, computationally efficient transform. The  ``hansenlaw`` method also provides an efficient forward Abel transform. It is recommended for most applications. See :doc:`hansenlaw` and :meth:`abel.hansenlaw.hansenlaw_transform`.

- ``linbasex`` (S) -- The "lin-BASEX" method of Gerber et al. [gerber2013]_ models the 2D projection using spherical functions, which evolve slowly as a function of polar angle. Thus, it can offer a substantial increase in signal-to-noise ratio in many situations, but **it is only appropriate for transforming certain projections that are appropriately described by these basis functions**. This is the case for typical velocity-map-imaging photoelectron/photoion spectroscopy [chandler1987]_ experiments, for which the algorithm was designed. However, for example, it would not be appropriate for transforming the object shown in :numref:`fig_overview`. The algorithm directly produces the coefficients of the involved spherical functions, which allows both the angular and radially integrated distributions to be produced analytically. This ability, combined with the strong noise-suppressing capability of using smooth basis functions, can aid the interpretation of photoelectron/photoion distributions. See :doc:`linbasex` and :meth:`abel.linbasex.linbasex_transform`.

- ``onion_bordas`` -- The onion-peeling method of Bordas et al. [bordas1996]_ is a Python adaptation of the MATLAB implementation of Rallis et al. [rallis2014]_. While it is conceptually similar to the ``onion_peeling`` method, the numerical implementation is significantly different. This method is reasonably slow, and is therefore not recommended for general use. See :doc:`onion_bordas` and :meth:`abel.onion_bordas.onion_bordas_transform`

- ``rbasex`` (F, S) -- The rBasex method is based on the pBasex method of Garcia et al. [garcia2004]_, using basis functions developed by Ryazanov [ryazanov2012]_. This method evaluates radial distributions of velocity-map images and transforms them to radial distributions of the reconstructed 3D distributions. Similar to ``linbasex``, the ``rbasex`` method makes additional assumptions about the symmetry of the data is not applicable to all situations. See :doc:`rbasex` and :meth:`abel.rbasex.rbasex_transform`.

- ``daun`` (F, S) -- The method by Daun et al. [daun2006]_ applies Tikhonov regularization to onion-peeling deconvolution. It is conceptually similar to "BASEX" (``basex``), but instead of :math:`L_2` regularization uses the first-order difference operator (approximating the derivative operator) as the Tikhonov matrix to suppress high-frequency oscillations, making the transform less sensitive to perturbations in the projected data. The PyAbel implementation also includes several extensions to the original method. First, in addition to the rectangular basis functions implied in onion peeling, explicit basis sets of piecewise polynomials up to 3rd degree (cubic splines) can be chosen. Second, the :math:`L_2` regularization (as in BASEX) is implemented for comparison. And most importantly, the non-negative least-squares solution to the deconvolution problem can be obtained, which produces meaningful results in situations where the transformed intensities must not be negative, and at the same time greatly reduces the baseline noise. See :doc:`daun` and :meth:`abel.daun.daun_transform`.


Implementation
--------------

The :meth:`abel.transform.Transform` class provides a uniform interface to all of the transform methods, as well as numerous related functions for centering and symmetrizing the input images. So, this interface can be used to quickly switch between transform methods to determine which method works best for a specific dataset.

Generating a sample image, performing a forward Abel transform, and completing an inverse Abel transform requires just a few lines of Python code:

.. code-block:: python

    import abel
    im0 = abel.tools.analytical.SampleImage().func
    im1 = abel.Transform(im0,
                         direction='forward',
                         method='hansenlaw').transform
    im2 = abel.Transform(im1,
                         direction='inverse',
                         method='three_point').transform


Choosing a different method for the forward or inverse transform requires only that the ``method`` argument be changed. Additional arguments can be passed to the individual transform functions using the ``transform_options`` argument. A basic graphical user interface (GUI) for PyAbel is also available as `example_GUI.py <https://github.com/PyAbel/PyAbel/blob/master/examples/example_GUI.py>`_ in the examples directory.

In addition to the transform methods themselves, PyAbel provides many of the pre-processing methods required to obtain optimal Abel transforms. For example, an accurate Abel transform requires that the center of the image is properly identified. Several approaches allow to perform this identification in PyAbel, including the center-of-mass, convolution, and Gaussian-fitting. Additionally, PyAbel incorporates a "circularization" method [gascooke2017]_, which allows the correction of images that contain features that are expected to be circular (such as photoelectron and photoion momentum distributions). Moreover, the :mod:`abel.tools` module contains a host of *post*-processing algorithms, which provide, for example, efficient projection into polar coordinates and radial or angular integration.


Conventions
-----------

The conventions for PyAbel are listed in the :ref:`Conventions <READMEconventions>` section of the :doc:`../readme_link`.

In order to provide similar results, PyAbel ensures that the numerical conventions are consistent across the various transform methods. For example. when dealing with pixel data, an ambiguity arises: do intensity values of the pixels represent the value of the data at :math:`r=\{0,\,1,\,2,\,...,\,n-1\}`, where :math:`n` is an integer, or do they correspond to :math:`r=\{0.5,\, 1.5,\, 2.5, \,..., \,n-0.5\}`? Either convention is reasonable, but comparing results from methods that adopt differing conventions can lead to small but significant shifts. PyAbel adopts the convention that the pixel values correspond to :math:`r=\{0,\,1,\,2,\,...,\,n-1\}`. One consequence of this is that, when considering an experimental image that contains both the left and right sides of the image, the total image width must be odd, such that :math:`r=\{1-n, \, ..., \, -2, \, -1,\, 0,\,1,\,2,\,...,\,n-1\}`. A potential disadvantage of our "odd image" convention is that 2D detectors typically have a grid of pixels with an *even* width (for example, a 512×512-pixel camera). If the image were perfectly centered on the detector, the odd-image convention would not match the data, and a half-pixel shift would be required. However, in nearly all real-world experiments, the image is not perfectly centered on the detector and a shift of *several* pixels is required, so the additional half-pixel shift is of no significance.

A similar ambiguity exists with regards to the left--right and top--bottom symmetry of the image. In principle, since the Abel transform assumes cylindrical symmetry, left--right symmetry should always exist, and it should only be necessary to record one side of the projection. However, many experiments record both sides of the projection. Additionally, many experiments record object that possess top--bottom symmetry. Thus, in some situations, it is best to average all of the image quadrants into a single quadrant and perform a single Abel transform on this quadrant. On the other hand, the quadrants may not be perfectly symmetric due to imperfections or noise in the experiment, and it may be best to perform the Abel transform on each quadrant separately and select the quadrant that produces the highest-quality data. PyAbel offers full flexibility, providing the ability to selectively enforce top--bottom and left--right symmetry, and to specify which quadrants are averaged. By default, each quadrant is processed separately and recombined into in composite image that does not assume either top--bottom or left--right symmetry. For more details, see :meth:`abel.transform.Transform`.

In the following performance benchmarks, left--right symmetry is assumed, because this is the most common benchmark presented in other studies ([rallis2014]_, [harrison2018]_). However, the image size is listed as the width of a square image. For example, :math:`n=513` corresponds to the time for the transformation of a 513×513-pixel image with the axis of symmetry located in the center. Since the Abel transform makes the assumption of cylindrical symmetry, both sides of the image are identical, and it is sufficient to perform the Abel transform on only one side of the image, or on an average of the two sides. So, to complete an Abel transform of a typical 513×513-pixel image, it is only necessary to perform the Abel transform on a 513×257-pixel array.

Another fundamental question about real-world Abel transforms is whether negative values are allowed in the transform result. In most situations, negative values are not physical, and some implementations set all negative values to zero. In contrast, PyAbel allows negative values, which enables its use in situations where negative values are physically reasonable. Moreover, maintaining negative values keeps the transform methods linear and gives users the option to average, smooth, or fit images either before or after the Abel transform without causing a systematic error in the baseline. Suppression of negative values in a transformed image ``im`` can easily be achieved by executing ``im[im<0] = 0``. On the other hand, the ``daun`` and ``rbasex`` methods offer optional non-linear regularization methods specifically designed to produce non-negative values without systematically shifting the baseline. It is recommended to use them instead of artificially zeroing negative values in situations where negative values are undesirable but the transform speed is not essential.


Comparison of Transform Results
-------------------------------

Since PyAbel incorporates numerous Abel-transform methods into the same interface, it is straightforward to directly compare the results. Consequently, a good approach is to simply try several (or all!) of the transform methods and see which produces the best results or performance for a specific application. Nevertheless, the following provides a brief comparison of the various transform methods in several cases. First, the methods are applied to a simple Gaussian function (for which an analytical Abel transform exists) in order to assess the accuracy of each transform method. Second, each method is applied to a "comb" function constructed of narrow peaks with noise added in order to closely examine the fundamental resolution of each method and how noise accumulates. Third, each method is used to provide the inverse Abel transform a high-resolution photoelectron-spectroscopy image in order to examine the ability of each method to handle real-world data.

The Abel transform of a Gaussian is simply a Gaussian, which allows a comparison of each numerical transform method with the analytical result in the case of a one-dimensional (1D) Gaussian (:numref:`fig_gaussian`). As expected, each transform method exhibits a small discrepancy compared with the analytical result. However, as the number of pixels is increased, the agreement between the transform and the analytical result improves. Even with only 70 points (the case shown in :numref:`fig_gaussian`), all of the method produce reasonable agreement. While all methods show a systematic error as :math:`r` approaches zero, the ``basex``, ``daun`` (especially with 3rd-degree basis functions), ``three_point``, and ``onion_peeling`` methods seem to provide the best agreement with the analytical result. The direct methods show fairly good agreement with the analytical curve, which is a result of the "correction" discussed above. We note that the results from the ``direct_Python`` and the ``direct_C`` methods produce identical results to within a factor of :math:`10^{-9}`.


.. plot:: transform_methods/comparison/fig_gaussian/gaussian.py
    :nofigs:

.. _fig_gaussian:
.. figure:: comparison/fig_gaussian/gaussian.*
    :figclass: align-center

    Comparison of inverse Abel-transform methods for a 1D Gaussian function with 70 points. All of the inverse Abel transform methods show reasonable agreement for the inverse Abel transform of a Gaussian function. The root-mean-square error (RMSE) for each method is listed in the figure legend. In the limit of many pixels, the error trends to zero. However, when a small number of pixels is used, systematic errors are seen, especially near the origin (:math:`r=0`). The error near the origin is more pronounced in some methods than others. The lowest error seen from the ``basex``, ``daun``, ``three_point``, and ``onion_peeling`` methods. The ``daun`` method with degree=0 is identical to ``onion_peeling`` and with degree=2 is slightly better (RMSE=0.05%). The ``linbasex`` and ``rbasex`` methods are not included in this figure because they are not applicable to 1D functions.


Applying the various transform methods to a synthetic "comb" function that consists of triangular peaks with one-pixel halfwidth -- the sharpest features representable on the pixel grid -- allows the fundamental resolution of each method to be visualized (:numref:`fig_comb`). In order to provide an understanding of how each method responds to noise, the function transformed in :numref:`fig_comb` also has uniformly distributed random noise added to each pixel. The figure reveals that some methods (``basex``, ``daun``, ``hansenlaw``, ``onion_peeling``, and ``two_point``) are capable of faithfully reproducing the sharpest features, while other methods (``direct``, ``onion_bordas``, and ``three_point``) provide some degree of smoothing. In general, the methods that provide the highest resolution also produce the highest noise, which is most obvious at low *r* values. The exception is the ``basex`` and ``daun`` methods using a moderate regularization factor (:numref:`fig_comb` b, c), which exhibit low noise near the center, while still displaying good resolution. The ``daun`` method with non-negativity regularization (:numref:`fig_comb` d), besides producing no negative values, significantly suppresses the baseline noise without affecting the sharp features. Thus, it seems that experiments that benefit from an optimal balance of noise suppression and resolution would benefit from inverse Abel-transform methods that incorporate regularization.


.. plot:: transform_methods/comparison/fig_comb/comb.py
    :nofigs:

.. _fig_comb:
.. figure:: comparison/fig_comb/comb.*
    :figclass: align-center

    Inverse Abel-transform methods applied to a synthetic "comb" function of one-pixel-width peaks with noise added. The gray line represents the analytical inverse Abel transform in the absence of noise. Some methods reproduce the height of the peaks, while other methods reduce noise while somewhat smoothing the peaks. The regularization in the ``basex`` and ``daun`` methods provides strong noise suppression near the origin, while maintaining peak height at higher values of :math:`r`. The ``daun`` method without regularization is identical to ``onion_peeling``, and its :math:`L_2` regularization is very similar to ``basex`` regularization.


Applying the various inverse Abel-transform methods to an experimental photoelectron-spectroscopy image (photoelectron spectrum of O\ :sub:`2`\ :sup:`−` photodetachment using a 455 nm laser, as described by Van Duzor et al. [vanduzor2010]_) provides a comparison of how the noise in the reconstructed image depends on the transform method (:numref:`fig_experiment`).


.. plot:: transform_methods/comparison/fig_experiment/experiment.py
    :nofigs:

.. raw:: latex

    % hack to insert negative space before figure (doesn't fit otherwise)
    \let\savecentering\centering
    \def\centering{\vspace*{-1.5em}\savecentering}

.. _fig_experiment:
.. figure:: comparison/fig_experiment/experiment.*
    :figclass: align-center

    Comparison of inverse Abel-transform methods applied to an experimental photoelectron velocity-map image. While all methods provide a faithful reconstruction of the experimental image, some of them cause a greater amplification of the noise present in the original image. The ``linbasex`` and ``rbasex`` methods models the image using a basis set of functions that vary slowly as a function of angle, which strongly reduces the high-frequency noise seen in the other transform methods. Besides the ``basex`` and ``daun`` method with regularization, the ``direct`` and ``three_point`` methods seem particularly suited for providing a low-noise transform. The ``daun`` method without regularization is identical to ``onion_peeling``, and its :math:`L_2` regularization is very similar to ``basex`` regularization.

.. raw:: latex

    % restore hacked definition
    \let\centering\savecentering


To a first approximation, the results of all the transform methods look similar. The ``rbasex`` and ``linbasex`` methods produces the "smoothest" image, which is a result of the fact that it models the projection using functions fitted to the image, that vary only slowly as a function of angle. The ``basex`` and ``daun`` methods incorporate a user-adjustable Tikhonov regularization factor, which tends to suppress noise, especially near the symmetry axis. Here, we set the regularization factor to 200 for ``basex`` and 100 for ``daun``, which provides significant noise suppression without noticeable broadening of the narrow features. When the regularization factor is set to zero, the ``basex`` and ``daun`` methods provide a transform that appears very similar to the ``onion_peeling`` method. For the other transform methods, the ``direct`` and ``three_point`` methods appear to have the strongest noise-filtering properties.


.. _fig_integration:
.. figure:: comparison/fig_experiment/integration.*
    :figclass: align-center

    Comparison of photoelectron spectra obtained by angular integration of the transformed images shown in :numref:`fig_experiment`, corresponding to various inverse Abel-transform methods applied to the same experimental velocity-map image. a) Looking at the entire photoelectron speed distribution, all of the transform methods appear to produce similar results. b) Closely examining two of the peaks shows that all of the methods produce similar results, but that some methods produce broader peaks than others. c) Examining the small peaks in the low-energy region reveals that some methods accumulate somewhat more noise than others. Notice the absence on negative intensities in the ``daun`` method with non-negativity regularization and the corresponding suppression of baseline oscillations.


:numref:`fig_integration` uses the same dataset as :numref:`fig_experiment`, but with an angular integration performed to show the 1D photoelectron spectrum. Good agreement is seen between most of the methods, even on a one-pixel level. Small but noticeable differences can be seen in the broadness of the peaks (:numref:`fig_integration`\ b). The ``hansenlaw``, ``onion_peeling`` and ``two_point`` methods show the sharpest peaks, suggesting that they provide enhanced ability to resolve sharp features. Of course, the differences between the methods are emphasized by the very high resolution of this dataset. In most cases, more pixels per peak yield a much better agreement between the transform methods. Interestingly, the ``linbasex`` method shows more baseline noise than the other methods. :numref:`fig_integration`\ c shows a close examination of the two lowest-energy peaks in the image. The methods that produce that sharpest peaks (``hansenlaw``, ``onion_peeling``, and ``two_point``) also exhibit somewhat more noise than the rest (except ``linbasex``).


Efficiency optimization
-----------------------

High-level efficiency optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For many applications of the inverse Abel transform, the speed at which transform can be completed is important. Even for those who are only aiming to transform a few images, the ability to perform Abel transforms efficiently may enable more effective data analysis. For example, faster Abel-transform method allow many different schemes for noise removal, smoothing, centering, and circularization to be explored more rapidly and effectively.

While PyAbel offers improvements to the raw computational efficiency of each transform method, it also provides improvements to the efficiency of the overall workflow, which are likely to provide a significant improvements for most applications. For example, since PyAbel provides a straightforward interface to switch between different transform methods (using :meth:`abel.transform.Transform`), a comparison of the results from each method can easily be made and the fastest method that produces acceptable results can be selected. Additionally, PyAbel provides fast algorithms for angular and radial integration, which can be the rate-limiting step for some data-processing workflows.

In addition, when the computational efficiency of the various Abel transform methods is evaluated, a distinction must be made between those methods that can pre-compute, save, and re-use information for a specific image size (``basex``, ``daun``, ``linbasex``, ``onion_peeling``, ``rbasex``, ``three_point``, ``two_point``) and those that do not (``direct``, ``hansenlaw``, ``onion_bordas``). Often, the time required for the pre-computation is orders of magnitude longer than the time required to complete the transform. One solution to this problem is to pre-compute information for a specific image size and provide this data as part of the software. Indeed, the popular BASEX application [dribinski2002]_ includes a "basis set" for transforming 1000×1000-pixel images. While this approach relieves the end user of the computational cost of generating basis sets, it often means that the ideal basis set for efficiently transforming an image of a specific size is not available. Thus, "padding" is necessary for smaller images, resulting in increased computational time, while larger higher-resolution images must be downsampled or cropped.

PyAbel provides the ability to pre-compute information for any image size and cache it to disk for future use. Moreover, a cached basis set intended for transforming a larger image can be automatically cropped for use on a smaller image, avoiding unnecessary computations. The ``basex`` algorithm in PyAbel also includes the ability to extend a basis set intended for transforming a smaller image for use on a larger image. This allows the ideal basis set to be efficiently generated for an arbitrary image size.


Low-level computational efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

General Advice
""""""""""""""
Transforming very large images, or a large number of images, requires inverse Abel-transform methods with high computational efficiency. PyAbel is written in Python, a high-level programming language that is easy to read, understand, and modify. A common criticism of high-level interpreted (non-compiled) languages like Python is that they provide significantly lower computational efficiency than low-level compiled languages, such as C or Fortran. However, such slowdowns can be avoided by calling functions from optimized math libraries for the key operations that serve as bottlenecks. For most of the transform methods (and indeed, all of the fastest methods), the operation that bottlenecks the transform process is a matrix-algebra operation, such as matrix multiplication. PyAbel uses matrix-algebra functions provided by the NumPy library, which are, in turn, provided by the Basic Linear Algebra Subprograms (BLAS) library. Thus, the algorithms in PyAbel have comparable performance to optimized C/Fortran.

One subtle consequence of this reliance on the BLAS algorithms is that the performance is dependent on the exact implementation of BLAS that is installed, and users seeking the highest level of performance may wish to experiment with different implementations. Different NumPy/SciPy distributions use different libraries by default, and some also provide a choice between several libraries. If the transform speed is important, it is advisable to run the benchmarks on all available configurations to select the fastest for the specific combination of the transform method, operating system and hardware.

Among the widely available options, the `Intel Math Kernel Library <https://en.wikipedia.org/wiki/Math_Kernel_Library>`_ (MKL) generally provides the best performance for Intel CPUs, although its installed size is rather huge and its performance on AMD CPUs is quite poor. It is used by default in `Anaconda Python <https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)>`_. `OpenBLAS <https://en.wikipedia.org/wiki/OpenBLAS>`_ generally provides the best performance for AMD CPUs and reasonably good performance for Intel CPUs. It is used by default in some distributions. AMD develops numerical libraries optimized for its own CPUs, but they are `not yet <https://github.com/numpy/numpy/issues/7372>`_ officially integrated with NumPy/SciPy.

Another important issue for modern Intel CPUs is that they suffer a performance degradation when `denormal numbers <https://en.wikipedia.org/wiki/Denormal_number>`_ are encountered, which sometimes happens in the intermediate calculations even if the input and output are “normal”. In this case, configuring the CPU to treat denormals as zeros does help. There is no official way to achieve this in NumPy/SciPy, but a third-party module `daz <https://github.com/chainer/daz>`_ can be used for this purpose. At least some modern AMD CPUs are less or not affected by this issue, although it's always better to run the tests to make sure.


Speed benchmarks
""""""""""""""""

The :class:`abel.benchmark.AbelTiming` class provides the ability to benchmark the speeds of the Abel transform algorithms. Here we show these benchmarks completed using a personal computer equipped with a 3.0 GHz Intel i7-9700 processor and 32 GB RAM running GNU/Linux (see also :doc:`comparison/fig_benchmarks/benchmarks` for some other systems).

A comparison of the time required to complete an inverse Abel transform versus the width of a square image is presented in :numref:`fig_transform_time`. All methods are benchmarked using their default parameters, with the following exceptions:

* **basex(var)** and **daun(var)** mean “variable regularization”, that is changing the regularization parameter for each transformed image.
* **daun(nonneg)** shows only the result of transforming the O\ :sub:`2`\ :sup:`−` image (see :numref:`fig_experiment`) with non-negativity regularization. Since the time needed for this non-linear transform strongly depends on the data, it is impossible to provide "universal" benchmarks; however, the general scaling is also expected to be roughly cubic.
* **direct_C** and **direct_Python** correspond to the “direct” method using its C (Cython) and Python backends respectively.
* **linbasex** and **rbasex** show whole-image (*n* × *n*) transforms, while all other methods show half-image (*n* rows, (*n* + 1)/2 columns) transforms.
* **rbasex(None)** means no output-image creation (only the transformed radial distributions).


.. plot:: transform_methods/comparison/fig_benchmarks/transform_time.py
    :nofigs:

.. _fig_transform_time:
.. figure:: comparison/fig_benchmarks/transform_time.*
    :figclass: align-center

    Computational efficiency of inverse Abel-transform methods. The time to complete an inverse Abel transform increases with the size of the image. Most of the methods display a roughly :math:`n^3` scaling (dashed gray line). The ``basex``, ``onion_peeling``, ``three_point``, and ``two_point`` methods all rely on similar matrix-algebra operations as their rate-limiting step, and consequently exhibit identical performance for typical experimental image sizes.


:numref:`fig_transform_time` reveals the computational scaling of each method as the image size is increased. At image sizes below :math:`n=100`, most of the transform methods exhibit a fairly flat relationship between image size and transform time, suggesting that the calculation is limited by the computational overhead. For image sizes of 1000 pixels and above, all the methods show a steep increase in transform time with increasing image size. A direct interpretation of the integral for the inverse Abel transform involves three nested loops, one over :math:`z`, one over :math:`r`, and one over :math:`y`, and we should expect :math:`n^3` scaling. Indeed, the ``direct_C`` and ``direct_Python`` methods scale as nearly :math:`n^3`. Several of the fastest methods (``basex``, ``onion_peeling``, ``three_point``, and ``two_point``) rely on matrix multiplication (or back substitution in case of ``daun``). These methods also scale roughly as :math:`n^3`, which is approximately the expected scaling for matrix-multiplication operations [coppersmith1990]_. For typical image sizes (~500--1000 pixels width), ``basex``, ``daun`` and the methods of Dasch [dasch1992]_ consistently out-perform other methods, often by several orders of magnitude. Interestingly, the ``hansenlaw`` and ``rbasex`` algorithms exhibits a nearly :math:`n^2` scaling and should outperform other algorithms for large image sizes. While the ``linbasex`` method does not provide the fastest transform, we note that it analytically provides the angular-integrated intensity and anisotropy parameters. Thus, if those parameters are desired outcomes -- as they often are during the analysis of photoelectron spectroscopy datasets -- then ``linbasex`` may provide an efficient analysis. The ``rbasex`` method also provides the intensity and anisotropy distributions directly. Moreover, if only these qualities are needed, without the transformed image, the transform can be completed faster and starts to outperform the fastest general-purpose methods for image sizes of ≳1000 pixels (extracting the desired distributions from the results of these methods requires additional time, not included in their plotted transform times).


.. plot:: transform_methods/comparison/fig_benchmarks/throughput.py
    :nofigs:

.. _fig_throughput:
.. figure:: comparison/fig_benchmarks/throughput.*
    :figclass: align-center

    The performance can also be viewed as a pixels-per-second rate. Here, it is clear that some methods provide sufficient throughput to transform images at rates far exceeding high-definition video (1000×1000 pixels at 30 frames per second is :math:`3\times10^7` pixels per second).


.. plot:: transform_methods/comparison/fig_benchmarks/basis_time.py
    :nofigs:

.. _fig_btime:
.. figure:: comparison/fig_benchmarks/basis_time.*
    :figclass: align-center

    Computational efficiency of the basis-set generation calculation.


The ``basex``, ``onion_peeling``, ``three_point``, and ``two_point`` methods run much faster if appropriately sized basis sets have been pre-calculated. For the ``basex`` method, the time for this pre-calculation is orders of magnitude longer than the transform time (:numref:`fig_btime`). For the Dasch methods (``onion_peeling``, ``three_point``, and ``two_point``), the pre-calculation is significantly longer than the transform time for image sizes smaller than 2000 pixels. For larger image sizes, the pre-calculation of the basis sets approaches the same speed as the transform itself. In particular, for the ``two_point`` method, the pre-calculation of the basis sets actually becomes faster than the image transform for *n* ≳ 4000. For the ``daun`` and ``linbasex`` methods, the pre-calculation of the basis sets is consistently faster than the transform itself, suggesting that the pre-calculation of basis sets isn't necessary for these methods.


Conclusion
----------

The various Abel-transform methods in PyAbel provide advantages for different situations. Nevertheless, certain recommendations can be made.

Methods recommended for general-purpose Abel transforms:

* ``basex``
* ``daun``
* ``hansenlaw``
* ``three-point``
* ``two-point``
* ``onion-peeling``
* ``direct``

Methods recommended for photoelectron/photoion datasets, or for images with similar shape:

* ``rbasex``
* ``linbasex``

Methods recommended for educational purposes only (these methods are generally slower and somewhat less accurate than competing transform methods):

* ``onion_bordas``


.. raw:: html

    <hr>

.. only:: html

    .. rubric:: References

.. [bordas1996] \ C. Bordas, F. Paulig, H. Helm, and D. L. Huestis. Photoelectron imaging spectrometry: Principle and inversion method. Rev. Sci. Instrum., **67**, 2257, 1996. DOI: `10.1063/1.1147044 <https://doi.org/10.1063/1.1147044>`_.

.. [chandler1987] David W. Chandler and Paul L. Houston. Two-dimensional imaging of state-selected photodissociation products detected by multiphoton ionization. J. Chem. Phys., **87**, 1445, 1987. DOI: `10.1063/1.453276 <https://doi.org/10.1063/1.453276>`_.

.. [cignoli2001] Francesco Cignoli, Silvana De Iuliis, Vittorio Manta, and Giorgio Zizak. Two-dimensional two-wavelength emission technique for soot diagnostics. Appl. Opt., **40**, 5370, 2001. DOI: `10.1364/AO.40.005370 <https://doi.org/10.1364/AO.40.005370>`_.

.. [coppersmith1990] Don Coppersmith and Shmuel Winograd. Matrix multiplication via arithmetic progressions. J. Symb. Comput., **9**, 251, 1990. DOI: `10.1016/S0747-7171(08)80013-2 <https://doi.org/10.1016/S0747-7171(08)80013-2>`_.

.. [dasch1992] Cameron J. Dasch. One-dimensional tomography: a comparison of Abel, onion-peeling, and filtered backprojection methods. Appl. Opt., **31**, 1146, 1992. DOI: `10.1364/AO.31.001146 <https://doi.org/10.1364/AO.31.001146>`_.

.. [daun2006] Kyle J. Daun, Kevin A. Thomson, Fengshan Liu, Fengshan J. Smallwood, Deconvolution of axisymmetric flame properties using Tikhonov regularization. Appl. Opt., **45**, 4638, 2006. DOI: `10.1364/AO.45.004638 <https://doi.org/10.1364/AO.45.004638>`_.

.. [demicheli2017] Enrico De Micheli. A fast algorithm for the inversion of Abel’s transform. Appl. Math. Comput., **301**, 12, 2017. DOI: `10.1016/j.amc.2016.12.009 <https://doi.org/10.1016/j.amc.2016.12.009>`_.

.. [dick2014] Bernhard Dick. Inverting ion images without Abel inversion: maximum entropy reconstruction of velocity maps. Phys. Chem. Chem. Phys., **16**, 570, 2014. DOI: `10.1039/C3CP53673D <https://doi.org/10.1039/C3CP53673D>`_.

.. [deiluliis1998] \ S. De Iuliis, M. Barbini, S. Benecchi, F. Cignoli, and G. Zizak. Determination of the soot volume fraction in an ethylene diffusion flame by multiwavelength analysis of soot radiation. Combust. Flame, **115**, 253, 1998. DOI: `10.1016/S0010-2180(97)00357-X <https://doi.org/10.1016/S0010-2180(97)00357-X>`_.

.. [dribinski2002] Vladimir Dribinski, Alexei Ossadtchi, Vladimir A. Mandelshtam, and Hanna Reisler. Reconstruction of Abel-transformable images: The Gaussian basis-set expansion Abel transform method. Rev. Sci. Instrum., *73*, 2634, 2002. DOI: `10.1063/1.1482156 <https://doi.org/10.1063/1.1482156>`_.

.. [das2017] Dhrubajyoti D. Das, William J. Cannella, Charles S. McEnally, Charles J. Mueller, and Lisa D. Pfefferle. Two-dimensional soot volume fraction measurements in flames doped with large hydrocarbons. Proc. Combust. Inst., **36**, 871, 2017. DOI: `10.1016/j.proci.2016.06.047 <https://doi.org/10.1016/j.proci.2016.06.047>`_.

.. [garcia2004] Gustavo A. Garcia, Laurent Nahon, and Ivan Powis. Two-dimensional charged particle image inversion using a polar basis function expansion. Rev. Sci. Instrum., **75**, 4989, 2004. DOI: `10.1063/1.1807578 <https://doi.org/10.1063/1.1807578>`_.

.. [gascooke2000] Jason R. Gascooke. Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals Molecule Dissociation. PhD thesis, Flinders University, SA 5001, Australia, 2000. Available at `github.com/PyAbel/abel_info/blob/master/Gascooke_Thesis.pdf <https://github.com/PyAbel/abel_info/blob/master/Gascooke_Thesis.pdf>`_.

.. [gascooke2017] Jason R. Gascooke, Stephen T. Gibson, and Warren D. Lawrance. A “circularisation” method to repair deformations and determine the centre of velocity map images. J. Chem. Phys., **147**, 013924, 2017. DOI: `10.1063/1.4981024 <https://doi.org/10.1063/1.4981024>`_.

.. [gerber2013] Thomas Gerber, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi, Peter Radi, and Yaroslav Sych. Charged particle velocity map image reconstruction with one-dimensional projections of spherical functions. Rev. Sci. Instrum., **84**, 033101, 2013. DOI: `10.1063/1.4793404 <https://doi.org/10.1063/1.4793404>`_.

.. [gladstone2016] Par G. Randall Gladstone, S. Alan Stern, Kimberly Ennico, Catherine B. Olkin, Harold A. Weaver, Leslie A. Young, Michael E. Summers, Darrell F. Strobel, David P. Hinson, Joshua A. Kammer, Alex H. Parker, Andrew J. Steffl, Ivan R. Linscott, Joel Wm. Parker, Andrew F. Cheng, David C. Slater, Maarten H. Versteeg, Thomas K. Greathouse, Kurt D. Retherford, Henry Throop, Nathaniel J. Cunningham, William W. Woods, Kelsi N. Singer, Constantine C. C. Tsang, Eric Schindhelm, Carey M. Lisse, Michael L. Wong, Yuk L. Yung, Xun Zhu, Werner Curdt, Panayotis Lavvas, Eliot F. Young, G. Leonard Tyler, and The New Horizons Science Team. The atmosphere of Pluto as observed by New Horizons. Science, **351**, 6279, 2016. DOI: `10.1126/science.aad8866 <https://doi.org/10.1126/science.aad8866>`_.

.. [glasser1978] \ J. Glasser, J. Chapelle, and J. C. Boettner. Abel inversion applied to plasma spectroscopy: a new interactive method. Appl. Opt., **17**, 3750, 1978. DOI: `10.1364/AO.17.003750 <https://doi.org/10.1364/AO.17.003750>`_.

.. [hansen1985] Eric W. Hansen and Phaih-Lan Law. Recursive methods for computing the abel transform and its inverse. J. Opt. Soc. Am. A, **2**, 510, Apr 1985. DOI: `10.1364/JOSAA.2.000510 <https://doi.org/10.1364/JOSAA.2.000510>`_.

.. [hansen1985b] \ E. Hansen. Fast hankel transform algorithm. IEEE Trans. Acoust., **33**, 666–671, 1985. DOI: `10.1109/tassp.1985.1164579 <https://doi.org/10.1109/tassp.1985.1164579>`_.

.. [harrison2018] \ G. R. Harrison, J. C. Vaughan, B. Hidle, and G. M. Laurent. DAVIS: a direct algorithm for velocity-map imaging system. J of Chem. Phys., **148**, 194101, 2018. DOI: `10.1063/1.5025057 <https://doi.org/10.1063/1.5025057>`_.

.. [hickstein2019] Daniel D. Hickstein, Stephen T. Gibson, Roman Yurchak, Dhrubajyoti D. Das, Mikhail Ryazanov. A direct comparison of high-speed methods for the numerical Abel transform. Rev. Sci. Instrum., **90**, 065115, 2019. DOI: `10.1063/1.5092635 <https://doi.org/10.1063/1.5092635>`_.

.. [lumpe2007] \ J. D. Lumpe, L. E. Floyd, L. C. Herring, S. T. Gibson, and B. R. Lewis. Measurements of thermospheric molecular oxygen from the solar ultraviolet spectral irradiance monitor. J. Geophys. Res. Atmos., **112**, D16308, 2007. DOI: `10.1029/2006JD008076 <https://doi.org/10.1029/2006JD008076>`_.

.. [rallis2014] \ C. E. Rallis, T. G. Burwitz, P. R. Andrews, M. Zohrabi, R. Averin, S. De, B. Bergues, Bethany Jochim,A. V. Voznyuk, Neal Gregerson, B. Gaire, I. Znakovskaya, J. McKenna, K. D. Carnes, M. F. Kling, I. Ben-Itzhak, and E. Wells. Incorporating real time velocity map image reconstruction into closed-loop coherent control. Rev. Sci. Instrum., **85**, 113105, 2014. DOI: `10.1063/1.4899267 <https://doi.org/10.1063/1.4899267>`_.

.. [ryazanov2012] Mikhail Ryazanov. Development and implementation of methods for sliced velocity map imaging. Studies of overtone-induced dissociation and isomerization dynamics of hydroxymethyl radical (CH\ :sub:`2`\ OH and CD\ :sub:`2`\ OH). PhD thesis, University of Southern California, 2012. `www.proquest.com/docview/1289069738 <https://www.proquest.com/docview/1289069738>`_

.. [snelling1999] David R. Snelling, Kevin A. Thomson, Gregory J. Smallwood, and Ömer L. Gülder. Two-dimensional imaging of soot volume fraction in laminar diffusion flames. Appl. Opt., **38**, 2478, 1999. DOI: `10.1364/AO.38.002478 <https://doi.org/10.1364/AO.38.002478>`_.

.. [vanduzor2010] Matthew Van Duzor, Foster Mbaiwa, Jie Wei, Tulsi Singh, Richard Mabbs, Andrei Sanov, Steven J. Cavanagh, Stephen T. Gibson, Brenton R. Lewis, and Jason R. Gascooke. Vibronic coupling in the superoxide anion: The vibrational dependence of the photoelectron angular distribution. J. Chem. Phys., **133**, 174311, 2010. DOI: `10.1063/1.3493349 <https://doi.org/10.1063/1.3493349>`_.

.. [whitaker2003] \ B. J. Whitaker. Imaging in Molecular Dynamics: Technology and Applications. Cambridge University Press, 2003. ISBN 9781139437905. `books.google.com/books?id=m8AYdeM3aRYC <https://books.google.com/books?id=m8AYdeM3aRYC>`_.

.. [yurchak2015] Roman Yurchak. Experimental and numerical study of accretion-ejection mechanisms in laboratory astrophysics. Thesis, Ecole Polytechnique (EDX), 2015. `theses.hal.science/tel-01338614 <https://theses.hal.science/tel-01338614>`_.


.. toctree::
    :hidden:

    comparison/fig_benchmarks/benchmarks
