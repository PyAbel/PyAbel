Comparison of Abel Transform Methods
====================================

Abstract
--------

This document provides a comparison of the quality and efficiency of the various Abel transform methods that are implemented in PyAbel.

Introduction
------------

The projection of a three-dimensional (3D) object onto a two-dimensional (2D) surface takes place in many measurement processes; a simple example is the recording of an X-ray image of a soup bowl, donut, egg, wineglass, or other cylindrically symmetric object :numref:`fig_overview`, where the axis of cylindrical symmetry is parallel to the plane of the detector. Such a projection is an example of a *forward* Abel transform and occurs in numerous experiments, including photoelectron/photoion spectroscopy [dribinski2002]_ [bordas1996]_ (rallis2014, chandler1987, ryazanov2012, renth2006, garcia2004) the studies of plasma plumes (glasser1978), flames \cite{deiluliis1998, cignoli2001, snelling1999, daun2006, liu2014, das2017}, and solar occulation of planetary atmospheres~\cite{gladstone2016, lumpe2007, craig1979}. The analysis of data from these experiments requires the use of the *inverse* Abel transform to recover the 3D object from its 2D projection.

.. _fig_overview:
.. figure:: https://user-images.githubusercontent.com/1107796/48970223-1b477b80-efc7-11e8-9feb-c614d6cadab6.png
   :width: 600px
   :alt: PyAbel
   :figclass: align-center
   
   The forward Abel transform maps a cylindrically symmetric three-dimensional (3D) object to its two-dimensional (2D) projection, a physical process that occurs in many experimental situations. For example, an X-ray image of the object on the left would produce the projection shown on the right. The *inverse* Abel transform takes the 2D projection and mathematically reconstructs the 3D object. As indicated by the Abel transform equations (below), the 3D object is described in terms of (*r,z*) coordinates, while the 2D projection is recorded in (*y,z*) coordinates.
  
  
While the forward and inverse Abel transforms may be written as simple, analytical expressions, attempts to naively evaluate them numerically for experimental images does not yield reliable results \cite{whitaker2003}. Consequently, many numerical methods have been developed to provide approximate solutions to the Abel transform [dribinski2002]_ bordas1996, chandler1987, dasch1992, rallis2014, [gerber2013]_ harrison2018, demicheli2017, dick2014}. Each method was created with specific goals in mind, with some taking advantage of pre-existing knowledge about the shape of the object, some prioritizing robustness to noise, and others offering enhanced computational efficiency. Each algorithm was originally implemented with somewhat different mathematical conventions and with often conflicting requirements for the size and format of the input data. Fortunately, PyAbel provides a consistent interface for the Abel-transform methods via the Python programming language, which allows for a straightforward, quantitative comparison of the output.

The following sections present various comparisons of the quality and speed of the various Abel transform algorithms presented in PyAbel. In general, all of the methods provide reasonably quality results, with some methods providing options for additional smoothing of the data. However, some methods are orders-of-magnitude most efficient than others. 


Transform algorithms
--------------------

The **forward Abel transform** is given by

.. math:: F(y,z) = 2 \int_y^{\infty} \frac{f(r,z)\,r}{\sqrt{r^2-y^2}}\,dr,


where *y*, *r*, and *z* are the spatial coordinates as shown in :numref:`fig_overview`, *f(r,z)* is the density of the 3D object at (*r,z*), and *F(y,z)* is the intensity of the projection in the 2D plane. 

The **inverse Abel transform** is given by

.. math:: f(r,z) = -\frac{1}{\pi} \int_r^{\infty} \frac{dF(y,z)}{dy}\, \frac{1}{\sqrt{y^2-r^2}}\,dy.

While the transform equations can be evaluated analytically for some mathematical functions, experiments typically generate discrete data (e.g., images collected with a digital camera), which must be evaluated numerically. Several issues arise when attempting to evaluate the Abel transform numerically. First, the simplest computational interpretation of inverse Abel transform equation involves three loops: over *z*, *r*, and *y*, respectively. Such nested loops can be computationally expensive. Additionally, *y = r* presents a singularity where the denominator goes to zero and the integrand goes to infinity. Finally, a simple approach requires a large number of sampling points in order to provide an accurate transform. Indeed, a simple numerical integration of the above equations has been shown to provide unreliable results [whitaker2003]_.  

Various algorithms have been developed to address these issues. PyAbel incorporates numerous algorithms for the inverse Abel transform, and some of these algorithms also support the forward Abel transform. The following comparisons focus on the results of the inverse Abel transform, because it is the inverse Abel transform that is used most frequently to interpret experimental data.

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


Speed benchmarks
----------------

The :class:`abel.benchmark.AbelTiming` class provides the ability to benchmark the speeds of the Abel transform algorithms.


Examples
^^^^^^^^

To give some sense of the relative and absolute speeds of each method, here we provide the results obtained on a system with a 3.0 GHz Intel i7-9700 processor and 32 GB RAM running GNU/Linux (see also our :ref:`publications <READMEcitation>` for the older 3.4 GHz Intel i7-6700 results).


Sustained transform speed
"""""""""""""""""""""""""

.. plot:: benchmarks/transform_time.py

.. plot:: benchmarks/throughput.py

* All method are benchmarked using their default parameters (exceptions are noted below).
* **basex(var)** means “variable regularization”, that is changing the regularization parameter for each transformed image.
* **direct_C** and **direct_Python** correspond to the “direct” method using its C (Cython) and Python backends respectively.
* **linbasex** and **rbasex** show whole-image (*n* × *n*) transforms, while all other methods show half-image (*n* rows, (*n* + 1)/2 columns) transforms.
* **rbasex(None)** means no output-image creation (only the transformed radial distributions).

Basis-set generation
""""""""""""""""""""

.. plot:: benchmarks/basis_time.py


General advice
^^^^^^^^^^^^^^

Most of the methods rely on matrix operations, and therefore their speed depends significantly on the performance of the underlying linear-algebra libraries. Different NumPy/SciPy distributions use different libraries by default, and some also provide a choice between several libraries. If the transform speed is important, it is advisable to run the benchmarks on all available configurations to select the fastest for the specific combination of the transform method, operating system and hardware.

Among the widely available options, the `Intel Math Kernel Library <https://en.wikipedia.org/wiki/Math_Kernel_Library>`_ (MKL) generally provides the best performance for Intel CPUs, although its installed size is rather huge and its performance on AMD CPUs is quite poor. It is used by default in `Anaconda Python <https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)>`_. `OpenBLAS <https://en.wikipedia.org/wiki/OpenBLAS>`_ generally provides the best performance for AMD CPUs and reasonably good performance for Intel CPUs. It is used by default in some distributions. AMD develops numerical libraries optimized for its own CPUs, but they are `not yet <https://github.com/numpy/numpy/issues/7372>`_ officially integrated with NumPy/SciPy.

Another important issue for modern Intel CPUs is that they suffer a severe performance degradation when `denormal numbers <https://en.wikipedia.org/wiki/Denormal_number>`_ are encountered, which sometimes happens in the intermediate calculations even if the input and output are “normal”. In this case, configuring the CPU to treat denormals as zeros does help. There is no official way to achieve this in NumPy/SciPy, but a third-party module `daz <https://github.com/chainer/daz>`_ can be used for this purpose. At least some modern AMD CPUs are less or not affected by this issue, although it's always better to run the tests to make sure.


Transform quality
-----------------

...coming soon! ...


References
----------

.. [bordas1996] C. Bordas, F. Paulig, H. Helm, and D. L. Huestis. Photoelectron imaging spectrometry: Principle and inversion method. Rev. Sci. Instrum., **67**, 2257, 1996. DOI:`10.1063/1.1147044 <https://doi.org/10.1063/1.1147044>`_

.. [chandler1987] David W. Chandler and Paul L. Houston. Two-dimensional imaging of state-selected photodissociation products detected by multiphoton ionization. J. Chem. Phys., **87**, 1445, 1987. DOI: `10.1063/1.453276 <https://doi.org/10.1063/1.453276>`_.

.. [dasch1992] Cameron J. Dasch. One-dimensional tomography: a comparison of abel, onion-peeling, and filtered backprojection methods. Appl. Opt., **31**:1146, 1992. DOI:`10.1364/AO.31.001146 <https://doi.org/10.1364/AO.31.001146>`_.

.. [dribinski2002] Vladimir Dribinski, Alexei Os- sadtchi, Vladimir A. Mandelshtam, and Hanna Reisler. Reconstruction of abel-transformable images: The gaussian basis-set expansion abel transform method. Rev. Sci. Instrum., *73*, 2634, 2002. DOI:`10.1063/1.1482156 <https://doi.org/10.1063/1.1482156>`_.

.. [garcia2004] Gustavo A. Garcia, Laurent Nahon, and Ivan Powis. Two- dimensional charged particle image inversion using a polar basis function expansion. Rev. Sci. Instrum., **75**, 4989, 2004. DOI:`10.1063/1.1807578 <https://doi.org/10.1063/1.1807578>`_.

.. [gascooke2000] Jason R. Gascooke. Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals Molecule Dissociation. PhD thesis, Flinders University, SA 5001, Australia, 2000. Available at `github.com/PyAbel/abel_info/blob/master/Gascooke_Thesis.pdf <https://github.com/PyAbel/abel_info/blob/master/Gascooke_Thesis.pdf>`_.

.. [gascooke2017] Jason R. Gascooke, Stephen T. Gibson, and Warren D. Lawrance. A “circularisation” method to repair deformations and determine the centre of velocity map images. J. Chem. Phys., **147**, 013924, 2017. DOI: `10.1063/1.4981024 <http://doi.org10.1063/1.4981024>`_.

.. [gerber2013] Thomas Gerber, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi, Peter Radi, and Yaroslav Sych. Charged particle velocity map image reconstruction with one-dimensional projections of spherical functions. Rev. Sci. Instrum., **84**, 033101, 2013. DOI:`10.1063/1.4793404 <https://doi.org/10.1063/1.4793404>`_.

.. [hansen1985] Eric W. Hansen and Phaih-Lan Law. Recursive methods for computing the abel transform and its inverse. J. Opt. Soc. Am. A, **2**, 510, Apr 1985. DOI:`10.1364/JOSAA.2.000510 <https://doi.org/10.1364/JOSAA.2.000510>`_.

.. [hansen1985b] E. Hansen. Fast hankel transform algorithm. IEEE Trans. Acoust., **33**, 666–671, 1985. DOI:`10.1109/tassp.1985.1164579 <https://doi.org/10.1109/tassp.1985.1164579>`_.

.. [ryazanov2012] Mikhail Ryazanov. Development and implementation of methods for sliced velocity map imaging. Studies of overtone-induced dissociation and isomerization dynamics of hydroxymethyl radical (CH2OH and CD2OH). PhD thesis, University of Southern California, 2012. `search.proquest.com/docview/1289069738 <https://search.proquest.com/docview/1289069738>`_

.. [whitaker2003] B.J. Whitaker. Imaging in Molecular Dynamics: Technology and Ap- plications. Cambridge University Press, 2003. ISBN 9781139437905. `books.google.com/books?id=m8AYdeM3aRYC <https://books.google.com/books?id=m8AYdeM3aRYC>`_.

.. [yurchak2015] Roman Yurchak. Experimental and numerical study of accretion-ejection mecha- nisms in laboratory astrophysics. Thesis, Ecole Polytechnique (EDX), 2015. `tel.archives-ouvertes.fr/tel-01338614 <https://tel.archives-ouvertes.fr/tel-01338614>`_.


