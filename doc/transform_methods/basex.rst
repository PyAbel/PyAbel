BASEX
=====


Introduction
------------

BASEX (Basis set expansion method) transform utilizes well-behaved functions (i.e., functions which have a known analytic Abel inverse) to transform images. 
In the current iteration of PyAbel, these functions (called basis functions) are modified Gaussian-type functions.
This technique was developed by Dribinski et al [1].

How it works
------------

This technique is based on expressing line-of-sight projection images (``raw_data``) as sums of functions which have known analytic Abel inverses. The provided raw images are expanded in a basis set composed of these basis functions with some weighting coefficients determined through a least-squares fitting process. 
These weighting coefficients are then applied to the (known) analytic inverse of these basis functions, which directly provides the Abel inverse of the raw images. Thus, the transform can be completed using simple linear algebra. 

In the current iteration of PyAbel, these basis functions are modified gaussians (see Eqs 14 and 15 in Dribinski et al. 2002). The process of evaluating these functions is computationally intensive, and the basis set generation process can take several seconds to minutes for larger images (larger than ~1000x1000 pixels). However, once calculated, these basis sets can be reused, and are therefore stored on disk and loaded quickly for future use. 
The transform then proceeds very quickly, since each raw image inversion is a simple matrix multiplication step.


When to use it
--------------

According to Dribinski et al. BASEX has several advantages:

1. For synthesized noise-free projections, BASEX reconstructs an essentially exact and artifact-free image, eschewing the need for interpolation procedures, which may introduce additional errors or assumptions.

2. BASEX is computationally cheap and only requires matrix multiplication once the basis sets have been generated and saved to disk.

3. The current basis set is composed of the modified Gaussian functions which are highly localized, uniform in coverage, and sufficiently narrow. This allows for resolution of very sharp features in the raw data with the basis functions. Moreover, the reconstruction procedure does not contribute to noise in the reconstructed image; noise appears in the image only when it exists in the projection.

4. Resolution of images reconstructed with BASEX are superior to those obtained with the Fourier-Hankel method, particularly for noisy projections. However, to obtain maximum resolution, it is important to properly center the projections prior to transforming with BASEX.

5. BASEX reconstructed images have an exact analytical expression, which allows for an analytical, higher resolution, calculation of the speed distribution, without increasing computation time.


How to use it
-------------

The recommended way to complete the inverse Abel transform using the BASEX algorithm for a full image is to use the :func:`abel.transform` function: ::

	abel.transform(myImage, method='basex', direction='inverse')

Note that the forward BASEX transform is not yet implemented in PyAbel. 

If you would like to access the BASEX algorithm directly (to transform a right-side half-image), you can use :func:`abel.basex.basex_transform`.


Notes
-----
More information about interpreting the equations in the paper and implementing the up/down asymmetric transform is discussed in `PyAbel Issue #54 <https://github.com/PyAbel/PyAbel/pull/54#issuecomment-164898116>`_


Citation
--------
[1] `Dribinski et al, 2002 (Rev. Sci. Instrum. 73, 2634) <http://dx.doi.org/10.1063/1.1482156>`_, (`pdf <http://www-bcf.usc.edu/~reisler/assets/pdf/67.pdf>`_)