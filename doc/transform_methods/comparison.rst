Comparison of Abel Transform Methods
====================================

Abstract
--------

This document provides a comparison of the quality and efficiency of the various Abel transform methods that are implemented in PyAbel.

Introduction
------------

The projection of a three-dimensional (3D) object onto a two-dimensional (2D) surface takes place in many measurement processes; a simple example is the recording of an X-ray image of a soup bowl, donut, egg, wineglass, or other cylindrically symmetric object :numref:`fig_overview`, where the axis of cylindrical symmetry is parallel to the plane of the detector. Such a projection is an example of a *forward* Abel transform and occurs in numerous experiments, including photoelectron/photoion spectroscopy ([dribinski2002]_ [bordas1996]_ [chandler1987]_) the studies of plasma plumes [glasser1978]_, flames ([deiluliis1998] [cignoli2001] [snelling1999] [daun2006] [liu2014] [das2017], and solar occulation of planetary atmospheres ([gladstone2016]_ [lumpe2007] [craig1979]). The analysis of data from these experiments requires the use of the *inverse* Abel transform to recover the 3D object from its 2D projection.

.. _fig_overview:
.. figure:: https://user-images.githubusercontent.com/1107796/48970223-1b477b80-efc7-11e8-9feb-c614d6cadab6.png
   :width: 600px
   :alt: PyAbel
   :figclass: align-center
   
   The forward Abel transform maps a cylindrically symmetric three-dimensional (3D) object to its two-dimensional (2D) projection, a physical process that occurs in many experimental situations. For example, an X-ray image of the object on the left would produce the projection shown on the right. The *inverse* Abel transform takes the 2D projection and mathematically reconstructs the 3D object. As indicated by the Abel transform equations (below), the 3D object is described in terms of (*r,z*) coordinates, while the 2D projection is recorded in (*y,z*) coordinates.
  
  
While the forward and inverse Abel transforms may be written as simple, analytical expressions, attempts to naively evaluate them numerically for experimental images does not yield reliable results \cite{whitaker2003}. Consequently, many numerical methods have been developed to provide approximate solutions to the Abel transform ([dribinski2002]_ [bordas1996]_ [chandler1987]_ [dasch1992]_ [rallis2014]_ [gerber2013]_ [harrison2018]_, [demicheli2017]_, [dick2014]_). Each method was created with specific goals in mind, with some taking advantage of pre-existing knowledge about the shape of the object, some prioritizing robustness to noise, and others offering enhanced computational efficiency. Each algorithm was originally implemented with somewhat different mathematical conventions and with often conflicting requirements for the size and format of the input data. Fortunately, PyAbel provides a consistent interface for the Abel-transform methods via the Python programming language, which allows for a straightforward, quantitative comparison of the output.

The following sections present various comparisons of the quality and speed of the various Abel transform algorithms presented in PyAbel. In general, all of the methods provide reasonably quality results, with some methods providing options for additional smoothing of the data. However, some methods are orders-of-magnitude most efficient than others. 


Math
----

The **forward Abel transform** is given by

.. math:: F(y,z) = 2 \int_y^{\infty} \frac{f(r,z)\,r}{\sqrt{r^2-y^2}}\,dr,


where *y*, *r*, and *z* are the spatial coordinates as shown in :numref:`fig_overview`, *f(r,z)* is the density of the 3D object at (*r,z*), and *F(y,z)* is the intensity of the projection in the 2D plane. 

The **inverse Abel transform** is given by

.. math:: f(r,z) = -\frac{1}{\pi} \int_r^{\infty} \frac{dF(y,z)}{dy}\, \frac{1}{\sqrt{y^2-r^2}}\,dy.

While the transform equations can be evaluated analytically for some mathematical functions, experiments typically generate discrete data (e.g., images collected with a digital camera), which must be evaluated numerically. Several issues arise when attempting to evaluate the Abel transform numerically. First, the simplest computational interpretation of inverse Abel transform equation involves three loops: over *z*, *r*, and *y*, respectively. Such nested loops can be computationally expensive. Additionally, *y = r* presents a singularity where the denominator goes to zero and the integrand goes to infinity. Finally, a simple approach requires a large number of sampling points in order to provide an accurate transform. Indeed, a simple numerical integration of the above equations has been shown to provide unreliable results [whitaker2003]_.  

Various algorithms have been developed to address these issues. PyAbel incorporates numerous algorithms for the inverse Abel transform, and some of these algorithms also support the forward Abel transform. The following comparisons focus on the results of the inverse Abel transform, because it is the inverse Abel transform that is used most frequently to interpret experimental data.


List of Abel Transform Methods in PyAbel
----------------------------------------

Below is a list that describes the basic approach and characteristics of all the Abel transform algorithms implemented in PyAbel. The title of each algorithm is the keyword that can be passed to the ``method`` keyword in :meth:`abel.transform.Transform`. Algorithms that pre-compute matrices for a specific image size, and (optionally) save them to disk for subsequent reuse, are indicated with the letter S. All methods implement the inverse Abel transform, while methods that also implement a forward transform are indicated with an F.

- ``basex`` (F,S) -- The "BAsis Set EXpansion" (BASEX) method of Dribinski and co-workers [dribinski2002]_ uses a basis set of Gaussian-like functions. This is one of the *de facto* standard methods in photoelectron/photoion spectroscopy [whitaker2003]_ and is highly recommended for general-purpose Abel transforms. The number of basis functions and their width can be varied. However, following the basis set provided with the original BASEX.exe program, by default the ``basex`` algorithm use a basis set where the full width at $1/e^2$ of the maximum is equal to 2~pixels and the basis functions are located at each pixel. Thus, the resolution of the image is roughly maintained. The ``basex`` algorithms allows a "Tikhonov regularization" to be applied, which suppresses intensity oscillations, producing a less noisy image. In the experimental comparisons presented below, the Tikhonov regularization factor is set to 200, which provides reasonable suppression of noise while still preserving the fine features in the image. See :doc:`basex` and :meth:`abel.basex.basex_transform`.

- ``onion_peeling`` (S) -- This method, and the following two methods (``three_point``, ``two_point``), are adapted from the 1992 paper by [dasch1992]_. All of these methods reduce the core Abel transform to a simple matrix-algebra operation, which allows a computationally efficient transform. Dasch emphasizes that these techniques work best in cases where the difference between adjacent points is much greater than the noise in the projections (i.e., where the raw data is not oversampled). This "onion-peeling deconvolution"" method is one of the simpler and faster inverse Abel-transform methods.See :doc:`onion_peeling` and :meth:`abel.dasch.onion_peeling_transform`.

- ``three_point`` (S) -- This "three point" algorithm [dasch1992]_ provides slightly more smoothing than the similar ``two_point`` or ``onion_peeling`` methods. The name refers to the fact that three neighboring pixels are considered, which improves the accuracy of the method for transforming smooth functions, as well as reducing the noise in the transformed image. The trade-off is that the ability of the method to transform very sharp (single pixel) features is reduced. This is an excellent general-purpose algorithm for the Abel transform. See :doc:`three_point` and :meth:`abel.dasch.three_point_transform`

- ``two_point`` (S) -- The "two-point method" (also described by Dasch [dasch1992]_) is a simplified version of the ``three_point`` algorithm and provides similar transform speeds. Since it only considers two adjacent points in the function, it allows sharper features to be transformed than the ``three_point`` method, but does not offer as much noise suppression. This method is also appropriate for most Abel transforms. 


- ``direct`` (F) -- The "direct" algorithms [yurchak2015]_ use a simple numerical integration, which closely resembles the basic Abel-transform equations (above). If the ``direct`` algorithm is used in its most naive form, the agreement with analytical solutions is poor, due to the singularity in the integral when *r=y*. However, a correction can be applied, where the function is assumed to be piecewise-linear across the pixel where this condition is met. This simple approximation allows a reasonably accurate transform to be completed. Fundamentally, the ``direct`` algorithm requires that the input function be finely sampled to achieve good results. PyAbel incorporates two implementations of the ``direct`` algorithm, which produce identical results, but with different calculation speeds. The ``direct_Python`` implementation is written in pure Python, for easy interpretation and modification. The ``direct_C`` implementation is written in `Cython <https://cython.org/>`_, a Python-like language that is converted to C and compiled, providing higher computational efficiency. This method is included mainly for educational and comparison purposes. In most cases, other methods will provide more reliable results and higher computational efficiency.  See :doc:`direct` and :meth:`abel.direct.direct_transform`.

- ``hansenlaw`` (F) -- The recursive method of Hansen and Law ([hansen1985]_ [hansen1985b]_ [gascooke2000]_) interprets the Abel transform as a linear space-variant state-variable equation, to provide a reliable, computationally efficient transform. The  ``hansenlaw`` method also provides an efficient forward Abel transform. It is recommended for most applications. See :doc:`hansenlaw` and :meth:`abel.hansenlaw.hansenlaw_transform`.

- ``linbasex`` (S) -- The "lin-BASEX" method of Gerber et al. [gerber2013]_ models the 2D projection using spherical functions, which evolve slowly as a function of polar angle. Thus, it can offer a substantial increase in signal-to-noise ratio in many situations, but **it is only appropriate for transforming certain projections that are appropriately described by these basis functions**. This is the case for typical velocity-map-imaging photoelectron/photoion spectroscopy [chandler1987]_ experiments, for which the algorithm was designed. However, for example, it would not be appropriate for transforming the object shown in :numref:`fig_overview`. The algorithm directly produces the coefficients of the involved spherical functions, which allows both the angular and radially integrated distributions to be produced analytically. This ability, combined with the strong noise-suppressing capability of using smooth basis functions, aids the interpretation of photoelectron/photoion distributions. See :doc:`linbasex` and :meth:`abel.linbasex.linbasex_transform`.

- ``onion_bordas`` -- The onion-peeling method of Bordas et al. [bordas1996]_ is a Python adaptation of the MatLab implementation of Rallis et al. [rallis2014]. While it is conceptually similar to the ``onion_peeling`` method, the numerical implementation is significantly different. This method is reasonably slow, and is therefore not recommended for general use. See :doc:`onion_bordas` and :meth:`abel.onion_bordas.onion_bordas_transform`

- ``rbasex`` (S) --  rBasex method is based on the pBasex method of Garcia et al. [garcia2004]_ and basis functions developed by Ryazanov [ryazanov2012]_. Evaluates radial distributions of velocity-map images and transforms them to radial distributions of the reconstructed 3D distributions. Similar to ``linbasex``, the ``rbasex`` method makes additional assumptions about the symmetry of the data is not applicable to all situations. See :doc:`rbasex` and :meth:`abel.rbasrx.rbasex_transform`.


Implementation
--------------

The :meth:`abel.transform.Transform` class provides a uniform interface to all of the transform methods, as well as numerous related functions for centering and symmetrizing the input images. So, this interface can be used to quickly switch between transform methods to see which works best for a specific dataset.

Generating a sample image, performing a forward Abel transform, and completing an inverse Abel transform requires just a few lines of Python code:

.. code-block:: python

    import abel
    im0 = abel.tools.analytical.SampleImage().image
    im1 = abel.Transform(im0, 
               direction = 'forward', 
               method = 'hansenlaw').transform
    im2 = abel.Transform(im1,
               direction = 'inverse',
               method = 'three_point').transform


Choosing a different method for the forward or inverse transform requires only that the ``method`` argument be changed. Additional arguments can be passed to the individual transform functions using the ``transform_options`` keyword. A basic graphical user interface (GUI) for PyAbel is also available: `github.com/PyAbel/PyAbel/blob/master/examples/example_GUI <https://github.com/PyAbel/PyAbel/blob/master/examples/example_GUI.py>`_

In addition to the transform methods themselves, PyAbel provides many of the pre-processing methods required to obtain optimal Abel transforms. For example, an accurate Abel transform requires that the center of the image is properly identified. Several approaches allow to perform this identification in PyAbel, including the center-of-mass, convolution, and Gaussian-fitting. Additionally, PyAbel incorporates a "circularization" method [gascooke2017]_, which allows the correction of images that contain features that are expected to be circular (such as photoelectron and photoion momentum distributions). Moreover, the :mod:`abel.tools` module contains a host of *post*-processing algorithms, which provide, for example, efficient projection into polar coordinates and radial or angular integration.


Conventions
-----------

The conventions for PyAbel are listed in the Conventions section of the :doc:`../readme_link`. 

In order to provide similar results, PyAbel ensures that the numerical conventions are consistent across the various transform methods. For example. when dealing with pixel data, an ambiguity arises: do intensity values of the pixels represent the value of the data at *r={0, 1, 2, ..., n-1}*, where *n* is an integer, or do they correspond to *r={0.5, 1.5, 2.5, ..., n-0.5}*? Either convention is reasonable, but comparing results from methods that adopt differing conventions can lead to small but significant shifts. PyAbel adopts the convention that the pixel values correspond to *r={0, 1, 2, ..., n-1}*. One consequence of this is that, when considering an experimental image that contains both the left and right sides of the image, the total image width must be odd, such that *r={1-n, ..., -2, -1, 0, 1, 2, ..., n-1}*. A potential disadvantage of our "odd image" convention is that 2D detectors typically have a grid of pixels with an *even* width (for example, a 512x512-pixel camera). If the image were perfectly centered on the detector, the odd-image convention would not match the data, and a half-pixel shift would be required. However, in nearly all real-world experiments, the image is not perfectly centered on the detector and a shift of *several* pixels is required, so the additional half-pixel shift is of no significance.

A similar ambiguity exists with regards to the left--right and top--bottom symmetry of the image. In principle, since the Abel transform assumes cylindrical symmetry, left--right symmetry should always exist, and it should only be necessary to record one side of the projection. However, many experiments record both sides of the projection. Additionally, many experiments record object that possess top--bottom symmetry. Thus, in some situations, it is best to average all of the image quadrants into a single quadrant and perform a single Abel transform on this quadrant. On the other hand, the quadrants may not be perfectly symmetric due to imperfections or noise in the experiment, and it may be best to perform the Abel transform on each quadrant separately and select the quadrant that produces the highest quality data. PyAbel offers full flexibility, providing the ability to selectively enforce top--bottom and left--right symmetry, and to specify which quadrants are averaged. By default, each quadrant is processed separately and recombined into in composite image that does not assume either top--bottom or left--right symmetry. For more details, see :meth:`abel.transform.Transform`.

In these performance benchmarks, left--right symmetry is assumed, because this is the most common benchmark presented in other studies ([rallis2014]_, [harrison2018]_). However, the image size is listed as the width of a square image. For example, *n=513* corresponds to the time for the transformation of a *513x513*-pixel image with the axis of symmetry located in the center. Since the Abel transform makes the assumption of cylindrical symmetry, both sides of the image are identical, and it is sufficient to perform the Abel transform on only one side of the image, or on an average of the two sides. So, to complete an Abel transform of a typical *513x513*-pixel image, it is only necessary to perform the Abel transform on a *513x257*-pixel array.

Another fundamental question about real-world Abel transforms is whether negative values are allowed in the transform result. In most situations, negative values are not physical, and some implementations set all negative values to zero. In contrast, PyAbel allows negative values, which enables its use in situations where negative values are physically reasonable. Moreover, maintaining negative values keeps the transform methods linear and gives users the option to average, smooth, or fit images either before or after the Abel transform without causing a systematic error in the baseline. Suppression of negative values can easily be achieved by including ``A[A<0] = 0``. 


Comparison of Transform Results
-------------------------------

Since PyAbel incorporates numerous Abel-transform methods into the same interface, it is straightforward to directly compare the results. Consequently, a good approach is to simply try several (or all!) of the transform methods and see which produces the best results or performance for a specific application. Nevertheless, the following provides a brief comparison of the various transform methods in several cases. First, the methods are applied to a simple Gaussian function (for which an analytical Abel transform exists) in order to assess the accuracy of each transform method. Second, each method is applied to a synthetic function constructed of narrow peaks with noise added in order to closely examine the fundamental resolution of each method and how noise accumulates. Third, each method is used to provide the inverse Abel transform a high-resolution photoelectron-spectroscopy image in order to examine the ability of each method to handle real-world data. 




The Abel transform of a Gaussian is simply a Gaussian, which allows a comparison of each numerical transform method with the analytical result in the case of a one-dimensional (1D) Gaussian (:numref:`fig_gaussian`). As expected, each transform method exhibits a small discrepancy compared with the analytical result. However, as the number of pixels is increased, the agreement between the transform and the analytical result improves. Even with only 70 points (the case shown in :numref:`fig_gaussian`), all of the method produce reasonable agreement. While all methods show a systematic error as *r* approaches zero, the ``basex``, ``three_point``, and ``onion_peeling`` methods seem to provide the best agreement with the analytical result. The direct methods show fairly good agreement with the analytical curve, which is a result of the "correction" discussed above. We note that the results from the ``direct_Python`` and the ``direct_C`` methods produce identical results to within a factor of 1e-9.


.. plot:: transform_methods/comparison/fig_gaussian/gaussian.py
    :nofigs:

.. _fig_gaussian:
.. figure:: comparison/fig_gaussian/gaussian.svg
    :width: 300px
    :alt: gaussian
    :figclass: align-center

    Comparison of inverse Abel-transform methods for a 1D Gaussian function with 70 points. All of the inverse Abel transform methods show reasonable agreement for the inverse Abel transform of a Gaussian function. The root-mean-square error (RMSE) for each method is listed in the figure legend. In the limit of many pixels, the error trends to zero. However, when a small number of pixels is used, systematic errors are seen near the origin. This effect is more pronounced in some methods than others. The lowest error seen from the basex, three_point, and onion_peeling methods. The linbasex and rBasex methods are not included in this figure because they are not applicable to 1D functions.
    
    
Applying the various transform methods to a synthetic function that consists of triangular peaks with one-pixel halfwidth -- the sharpest features representable on the pixel grid -- allows the fundamental resolution of each method to be visualized (:numref:`fig_comb`). In order to provide an understanding of how each method responds to noise, the function transformed in :numref:`fig_comb` also has uniformly distributed random noise added to each pixel. The figure reveals that some methods (``basex``, ``hansenlaw``), ``onion_peeling``, and ``two_point``) are capable of faithfully reproducing the sharpest features, while other methods (``direct``, ``onion_bordas``, and ``three_point``) provide some degree of smoothing. In general, the methods that provide the highest resolution also produce the highest noise, which is most obvious at low *r* values. The exception is the ``basex`` method using a moderate regularization factor (:numref:`fig_comb` b), which exhibits low noise near the center, while still displaying good resolution. Thus, it seems that experiments that benefit from an optimal balance of noise suppression and resolution would benefit from inverse Abel-transform methods that incorporate regularization.


.. plot:: transform_methods/comparison/fig_comb/comb.py
    :nofigs:

.. _fig_comb:
.. figure:: comparison/fig_comb/comb.svg
    :width: 300px
    :figclass: align-center

    Inverse Abel-transform methods applied to a synthetic image of one-pixel peaks with noise added.} a-h) The gray line represents the analytical inverse Abel transform in the absence of noise. Some methods reproduce the height of the peaks, while other methods reduce noise while somewhat smoothing the peaks. The regularization in the \texttt{basex} method provides strong noise suppression near the origin, while maintaining peak height at higher values of *r*.



.. plot:: transform_methods/comparison/fig_experiment/experiment.py
    :nofigs:

.. _fig_experiment:
.. figure:: comparison/fig_experiment/experiment.svg
    :width: 300px
    :figclass: align-center

    Comparison of inverse Abel-transform methods for an experimental photoelectron spectrum. While all methods provide a faithful reconstruction of the experimental image, some of them cause a greater amplification of the noise present in the original image. The ``linbasex`` method models the image using a basis set of functions that vary slowly as a function of angle, which strongly reduces the high-frequency noise seen in the other transform methods. Besides the ``basex`` method with adjustable regularization, the ``direct`` and ``three_point`` methods seem particularly suited for providing a low-noise transform. This dataset is the photoelectron spectrum of O2 photodetachment using a 455 nm laser, as described in [vanduzor2010]_.


Applying the various inverse Abel-transform methods to an experimental photoelectron-spectroscopy image (:numref:`fig_integration`) provides a comparison of how the noise in the reconstructed image depends on the transform method. To a first approximation, the results of all the transform methods look similar. The ``linbasex`` method produces the "smoothest" image, which is a result of the fact that it models the projection using functions fitted to the image, that vary only slowly as a function of angle. The ``basex`` method incorporates a user-adjustable Tikhonov regularization factor, which tends to suppress noise, especially near the symmetry axis. Here, we set the regularization factor to 200, which provides significant noise suppression while providing no noticeable broadening of the narrow features. When the regularization factor is set to zero, the ``basex`` method provides a transform that appears very similar to the ``onion_peeling`` method. For the other transform methods, the ``direct`` and ``three_point`` methods appear to have the strongest noise-filtering properties. 


.. _fig_integration:
.. figure:: comparison/fig_experiment/integration.svg
    :width: 300px
    :figclass: align-center
    
    Comparison of inverse Abel-transform methods applied to an experimental photoelectron spectrum and angularly integrated. The results shown in this figure are simply the angularly integrated 2D spectra shown in :numref:`fig_experiment`. a) Looking at the entire photoelectron speed distribution, all of the transform methods appear to produce similar results. b) Closely examining two of the peaks shows that all of the methods produce similar results, but that some methods produce broader peaks than others. c) Examining the small peaks in the low-energy region reveals that some methods accumulate somewhat more noise than others.


:numref:`fig_integration` uses the same dataset as :numref:`fig_experiment`, but an angular integration performed to show the 1D photoelectron spectrum. Good agreement is seen between most of the methods, even on a one-pixel level. Small but noticeable differences can be seen in the broadness of the peaks (:numref:`fig_integration` b). The ``hansenlaw``, ``onion_peeling`` and ``two_point`` methods show the sharpest peaks, suggesting that they provide enhanced ability to resolve sharp features. Of course, the differences between the methods are emphasized by the very high resolution of this dataset. In most cases, more pixels per peak yield a much better agreement between the transform methods. Interestingly, the ``linbasex`` method shows more baseline noise than the other methods. :numref:`fig_integration` c shows a close examination of the two lowest-energy peaks in the image. The methods that produce that sharpest peaks (``hansenlaw``, ``onion_peeling``, and ``two_point``) also exhibit somewhat more noise than the rest (except ``linbasex``).


Efficiency optimization
-----------------------

High-level efficiency optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For many applications of the inverse Abel transform, the speed at which transform can be completed is important. Even for those who are only aiming to transform a few images, the ability to perform Abel transforms efficiently may enable more effective data analysis. For example, users may want to explore many different schemes for noise removal, smoothing, centering, and circularization, and faster Abel-transform algorithms allow this parameter space to be explored more rapidly and effectively.

While PyAbel offers improvements to the raw computational efficiency of each transform method, it also provides improvements to the efficiency of the overall workflow, which are likely to provide a significant improvements for most applications. For example, since PyAbel provides a straightforward interface to switch between different transform methods, a comparison of the results from each method can easily be made and the fastest method that produces acceptable results can be selected. Additionally, PyAbel provides fast algorithms for angular and radial integration, which can be the rate-limiting step for some data-processing workflows.

In addition, when the computational efficiency of the various Abel transform methods is evaluated, a distinction must be made between those methods that can pre-compute, save, and re-use information for a specific image size (``basex``, ``three_point``, ``two_point``, ``onion_peeling``, ``linbasex``) and those that do not (``hansenlaw``, ``direct``, ``onion_bordas``). Often, the time required for the pre-computation is orders of magnitude longer than the time required to complete the transform. One solution to this problem is to pre-compute information for a specific image size and provide this data as part of the software. Indeed, the popular BASEX application includes a "basis set" for transforming 1000x1000-pixel images. While this approach relieves the end user of the computational cost of generating basis sets, it often means that the ideal basis set for efficiently transforming an image of a specific size is not available. Thus, padding is necessary for smaller images, resulting in increased computational time, while higher-resolution images must be downsampled or cropped. PyAbel provides the ability to pre-compute information for any image size and cache it to disk for future use. Moreover, a cached basis set intended for transforming a larger image can be automatically cropped for use on a smaller image, avoiding unnecessary computations. The ``basex`` algorithm in PyAbel also includes the ability to extend a basis set intended for transforming a smaller image for use on a larger image. This allows the ideal basis set to be efficiently generated for an arbitrary image size.



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

The :class:`abel.benchmark.AbelTiming` class provides the ability to benchmark the speeds of the Abel transform algorithms. A comparison of the time required to complete an inverse Abel transform versus the width of a square image is presented in :numref:`fig_transform_time`. All method are benchmarked using their default parameters, with the following exceptions:

* **basex(var)** means “variable regularization”, that is changing the regularization parameter for each transformed image.
* **direct_C** and **direct_Python** correspond to the “direct” method using its C (Cython) and Python backends respectively.
* **linbasex** and **rbasex** show whole-image (*n* × *n*) transforms, while all other methods show half-image (*n* rows, (*n* + 1)/2 columns) transforms.
* **rbasex(None)** means no output-image creation (only the transformed radial distributions).


.. plot:: transform_methods/comparison/fig_benchmarks/transform_time.py
    :nofigs:

.. _fig_transform_time:
.. figure:: comparison/fig_benchmarks/transform_time.svg
    :width: 500px
    :figclass: align-center
    
    Computational efficiency of inverse Abel-transform methods. The time to complete an inverse Abel transform increases with the size of the image. Most of the methods display a roughly *n^3* scaling (dashed gray line). The ``basex``, ``two_point``, ``three_point``, and ``onion_peeling`` methods all rely on similar matrix-algebra operations as their rate-limiting step, and consequently exhibit identical performance for typical experimental image sizes. These benchmarks were completed using a personal computer equipped with a 3.0 GHz Intel i7-9700 processor and 32 GB RAM running GNU/Linux.
    
   
:numref:`fig_transform_time` reveals the computational scaling of each method as the image size is increased. At image sizes below *n=100*, most of the transform methods exhibit a fairly flat relationship between image size and transform time, suggesting that the calculation is limited by the computational overhead. For image sizes of 1000 pixels and above, the all methods show a steep increase in transform time with increasing image size. A direct interpretation of the integral for the inverse Abel transform involves three nested loops, one over *z*, one over *r*, and one over *y*, and we should expect *n^3* scaling. Indeed, the ``direct_C`` and ``direct_Python`` methods scale as nearly *n^3*. Several of the fastest methods (``basex``, ``onion_peeling``, two_point``, and ``three_point``) rely on matrix multiplication. These methods scale roughly as *n^{3}*, which is approximately the expected scaling for matrix-multiplication operations [coppersmith1990]_. For typical image sizes (~500--1000 pixels width), ``basex`` and the methods of Dasch [dasch1992]_ consistently out-perform other methods, often by several orders of magnitude. Interestingly, the ``hansenlaw`` algorithm exhibits a nearly *n^2* scaling and should outperform other algorithms for large image sizes. While the ``linbasex`` method does not provide the fastest transform, we note that it analytically provides the angular-integrated intensity and anisotropy parameters. Thus, if those parameters are desired outcomes -- as they often are during the analysis of photoelectron spectroscopy datasets -- then ``linbasex`` may provide an efficient analysis.


.. plot:: transform_methods/comparison/fig_benchmarks/throughput.py
    :nofigs:

.. _fig_throughput:
.. figure:: comparison/fig_benchmarks/throughput.svg
    :width: 500px
    :figclass: align-center

    The performance can also be viewed in terms of pixels-per-second rate. Here, it is clear that some methods provide sufficient throughput to transform images at rates far exceeding high-definition video.


.. plot:: transform_methods/comparison/fig_benchmarks/btime.py
    :nofigs:


.. _fig_btime:
.. figure:: comparison/fig_benchmarks/btime.svg
    :width: 500px
    :figclass: align-center

    Computational efficiency of the basis set generation calculation.



The ``basex``, ``two_point``, ``three_point``, and ``onion_peeling`` methods run much faster if appropriately sized basis sets have been pre-calculated. For the ``basex`` method, the time for this pre-calculation is orders of magnitude longer than the transform time (:numref:`fig_btime`). For the Dasch methods (``three_point``, ``onion_peeling``, and ``two_point``), the pre-calculation is significantly longer than the transform time for image sizes smaller than 2000 pixels. For larger image sizes, the pre-calculation of the basis sets approaches the same speed as the transform itself. In particular, for the ``two_point`` method, the pre-calculation of the basis sets actually becomes faster than the image transform for *n* greater than about 4000. For the ``linbasex`` method, the pre-calculation of the basis sets is consistently faster than the transform itself, suggesting that the pre-calculation of basis sets isn't necessary for this method.


Conclusion
----------

...conclusion goes here...



References
----------

.. [bordas1996] C. Bordas, F. Paulig, H. Helm, and D. L. Huestis. Photoelectron imaging spectrometry: Principle and inversion method. Rev. Sci. Instrum., **67**, 2257, 1996. DOI:`10.1063/1.1147044 <https://doi.org/10.1063/1.1147044>`_

.. [chandler1987] David W. Chandler and Paul L. Houston. Two-dimensional imaging of state-selected photodissociation products detected by multiphoton ionization. J. Chem. Phys., **87**, 1445, 1987. DOI: `10.1063/1.453276 <https://doi.org/10.1063/1.453276>`_.

.. [coppersmith1990] Don Coppersmith and Shmuel Winograd. Matrix multiplication via arithmetic progressions. J. Symb. Comput., **9**,251, 1990. DOI: `10.1016/S0747-7171(08)80013-2 <https://doi.org/10.1016/S0747-7171(08)80013-2>`_.

.. [dasch1992] Cameron J. Dasch. One-dimensional tomography: a comparison of abel, onion-peeling, and filtered backprojection methods. Appl. Opt., **31**:1146, 1992. DOI:`10.1364/AO.31.001146 <https://doi.org/10.1364/AO.31.001146>`_.

.. [demicheli2017] Enrico De Micheli. A fast algorithm for the inversion of abel’s transform. Appl. Math. Comput., **301**, 12, 2017. DOI: `10.1016/j.amc.2016.12.009 <https://doi.org/10.1016/j.amc.2016.12.009>`_.

.. [dick2014] Bernhard Dick. Inverting ion images without abel inversion: maximum entropy reconstruction of velocity maps. Phys. Chem. Chem. Phys., **16**, 570, 2014. DOI:`10.1039/C3CP53673D <http://doi.org/10.1039/C3CP53673D>`_.

.. [dribinski2002] Vladimir Dribinski, Alexei Os- sadtchi, Vladimir A. Mandelshtam, and Hanna Reisler. Reconstruction of abel-transformable images: The gaussian basis-set expansion abel transform method. Rev. Sci. Instrum., *73*, 2634, 2002. DOI:`10.1063/1.1482156 <https://doi.org/10.1063/1.1482156>`_.

.. [garcia2004] Gustavo A. Garcia, Laurent Nahon, and Ivan Powis. Two- dimensional charged particle image inversion using a polar basis function expansion. Rev. Sci. Instrum., **75**, 4989, 2004. DOI:`10.1063/1.1807578 <https://doi.org/10.1063/1.1807578>`_.

.. [gascooke2000] Jason R. Gascooke. Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals Molecule Dissociation. PhD thesis, Flinders University, SA 5001, Australia, 2000. Available at `github.com/PyAbel/abel_info/blob/master/Gascooke_Thesis.pdf <https://github.com/PyAbel/abel_info/blob/master/Gascooke_Thesis.pdf>`_.

.. [gascooke2017] Jason R. Gascooke, Stephen T. Gibson, and Warren D. Lawrance. A “circularisation” method to repair deformations and determine the centre of velocity map images. J. Chem. Phys., **147**, 013924, 2017. DOI: `10.1063/1.4981024 <http://doi.org10.1063/1.4981024>`_.

.. [gerber2013] Thomas Gerber, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi, Peter Radi, and Yaroslav Sych. Charged particle velocity map image reconstruction with one-dimensional projections of spherical functions. Rev. Sci. Instrum., **84**, 033101, 2013. DOI:`10.1063/1.4793404 <https://doi.org/10.1063/1.4793404>`_.

.. [gladstone2016] Par G. Randall Gladstone, S. Alan Stern, Kimberly Ennico, Catherine B. Olkin, Harold A. Weaver, Leslie A. Young, Michael E. Summers, Darrell F. Strobel, David P. Hinson, Joshua A. Kammer, Alex H. Parker, Andrew J. Steffl, Ivan R. Linscott, Joel Wm. Parker, Andrew F. Cheng, David C. Slater, Maarten H. Versteeg, Thomas K. Greathouse, Kurt D. Retherford, Henry Throop, Nathaniel J. Cunningham, William W. Woods, Kelsi N. Singer, Constantine C. C. Tsang, Eric Schindhelm, Carey M. Lisse, Michael L. Wong, Yuk L. Yung, Xun Zhu, Werner Curdt, Panayotis Lavvas, Eliot F. Young, G. Leonard Tyler, and The New Horizons Science Team. The atmosphere of pluto as observed by new horizons. Science, **351**, 6279, 2016. DOI: `10.1126/science.aad8866 <https://doi.org/10.1126/science.aad8866>`_.

.. [glasser1978] J. Glasser, J. Chapelle, and J. C. Boettner. Abel inversion applied to plasma spectroscopy: a new interactive method. Appl. Opt., **17**, 3750, 1978. DOI: `10.1364/AO.17.003750 <https://doi.org/10.1364/AO.17.003750>`_.

.. [hansen1985] Eric W. Hansen and Phaih-Lan Law. Recursive methods for computing the abel transform and its inverse. J. Opt. Soc. Am. A, **2**, 510, Apr 1985. DOI:`10.1364/JOSAA.2.000510 <https://doi.org/10.1364/JOSAA.2.000510>`_.

.. [hansen1985b] E. Hansen. Fast hankel transform algorithm. IEEE Trans. Acoust., **33**, 666–671, 1985. DOI:`10.1109/tassp.1985.1164579 <https://doi.org/10.1109/tassp.1985.1164579>`_.

.. [harrison2018] G. R. Harrison, J. C. Vaughan, B. Hidle, and G. M. Laurent. DAVIS: a direct algorithm for velocity-map imaging system. J of Chem. Phys., **148**, 194101, 2018. DOI:`10.1063/1.5025057 <https://doi.org/10.1063/1.5025057>`_.

.. [rallis2014] C. E. Rallis, T. G. Burwitz, P. R. Andrews, M. Zohrabi, R. Averin, S. De, B. Bergues, Bethany Jochim,A. V. Voznyuk, Neal Gregerson, B. Gaire, I. Znakovskaya, J. McKenna, K. D. Carnes, M. F. Kling,I. Ben-Itzhak, and E. Wells. Incorporating real time velocity map image reconstruction into closed-loop coherent control. Rev. Sci. Instrum., **85**, 113105, 2014. DOI: `10.1063/1.4899267 <https://doi.org/10.1063/1.4899267>`_.

.. [ryazanov2012] Mikhail Ryazanov. Development and implementation of methods for sliced velocity map imaging. Studies of overtone-induced dissociation and isomerization dynamics of hydroxymethyl radical (CH2OH and CD2OH). PhD thesis, University of Southern California, 2012. `search.proquest.com/docview/1289069738 <https://search.proquest.com/docview/1289069738>`_

.. [vanduzor2010] Matthew Van Duzor, Foster Mbaiwa, Jie Wei, Tulsi Singh, Richard Mabbs, Andrei Sanov, Steven J. Cavanagh, Stephen T. Gibson, Brenton R. Lewis, and Jason R. Gascooke. Vibronic coupling in the superoxide anion: The vibrational dependence of the photoelectron angular distribution. J. Chem. Phys., **133**, 174311, 2010. DOI: `10.1063/1.3493349 <https://doi.org/10.1063/1.3493349>`_.

.. [whitaker2003] B.J. Whitaker. Imaging in Molecular Dynamics: Technology and Ap- plications. Cambridge University Press, 2003. ISBN 9781139437905. `books.google.com/books?id=m8AYdeM3aRYC <https://books.google.com/books?id=m8AYdeM3aRYC>`_.

.. [yurchak2015] Roman Yurchak. Experimental and numerical study of accretion-ejection mecha- nisms in laboratory astrophysics. Thesis, Ecole Polytechnique (EDX), 2015. `tel.archives-ouvertes.fr/tel-01338614 <https://tel.archives-ouvertes.fr/tel-01338614>`_.


