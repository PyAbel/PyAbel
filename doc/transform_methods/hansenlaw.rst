.. |nbsp| unicode:: 0xA0 
   :trim:

Hansen-Law
==========


Introduction
------------

The Hansen and Law transform is the work of E. W. Hansen with P.-L. Law,
zero-order hold approximation, [1] and the more accurate 
first-order hold approximation. [2]

From the abstract [1]:

*... new family of algorithms, principally for Abel inversion, that are 
recursive and hence computationally efficient. The methods are based on a 
linear, space-variant, state-variable model of the Abel transform. The model 
is the basis for deterministic algorithms, applicable when data are noise free, 
and least-squares estimation (Kalman filter) algorithms, which accommodate 
the noisy data case.*

The key advantage of the algorithm that is is fast, providing both the **forward** Abel and **inverse** Abel transform.


How it works
------------

.. figure:: https://cloud.githubusercontent.com/assets/10932229/13543157/c83d3796-e2bc-11e5-9210-12be6d24b8fc.png
   :width: 200px
   :alt: projection diag
   :align: right
   :figclass: align-center

   Projection geometry (Fig. 1 [1])

image function |nbsp|  :math:`f(r)`, |nbsp| projected function |nbsp|  :math:`g(R)`

forward Abel transform 

.. math:: g(R) = 2 \int_R^\infty \frac{f(r) r}{\sqrt{r^2 - R^2}} dr 

inverse Abel transform 

.. math:: f(r) = -\frac{1}{\pi}  \int_r^\infty \frac{g^\prime(R)}{\sqrt{R^2 - r^2}} dR



The Hansen and Law method makes use of a coordinate transformation, which is 
used to model the Abel transform, and derive *reconstruction* filters. The Abel
transform is treated as a system modeled by a set of linear differential 
equations. In this framework the forward Abel transform :math:`g(R)` is 
the solution of a differential equation with :math:`f(r)` as its driving 
function. Similarly, the Abel inversion :math:`f(r)` is a solution of a 
differential equation with :math:`g^\prime(R)` as its driving function. 

.. figure:: https://cloud.githubusercontent.com/assets/10932229/13544803/13bf0d0e-e2cf-11e5-97d5-bece1e61d904.png 
   :width: 350px
   :alt: recursion
   :align: right
   :figclass: align-center

   Recursion: pixel value from adjacent outer-pixel


forward transform

.. math:: 

  x_{n+1} &= \Phi_n x_n + \Gamma_n f_n 

  g_n &= \tilde{C} x_n

inverse transform

.. math:: 

  x_{n+1} &= \Phi_n x_n + \Gamma_n g^\prime_n 

  f_n &= \tilde{C} x_n


Note the only difference between the *forward* and *inverse* algorithms is 
the exchange of :math:`f_n` with :math:`g^\prime_n` (or :math:`g_n`).

|
|

:math:`\Phi_n` and :math:`\Gamma_n` are functions with predetermined 
parameter constants, all listed in [1].


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
``hansenlaw method``, simply use the :class:`abel.Transform`: class ::

    abel.Transform(myImage, method='hansenlaw', direction='forward').transform
    abel.Transform(myImage, method='hansenlaw', direction='inverse').transform


If you would like to access the Hansen-Law algorithm directly (to transform a 
right-side half-image), you can use :func:`abel.hansenlaw.hansenlaw_transform`.



Tips
----

`hansenlaw` tends to perform better with images of large size :math:`n \gt 1001` pixel width. For smaller images the angular_integration (speed) profile may look better if sub-pixel sampling is used via: ::

    angular_integration_options=dict(dr=0.5)


Example
-------

.. plot:: ../examples/example_O2_PES_PAD.py


Historical Note
---------------

The Hansen and Law algorithm was almost lost to the scientific community. It was 
rediscovered by Jason Gascooke (Flinders University, South Australia) for use in 
his velocity-map image analysis, and written up in his PhD thesis [3]. 



Citation
--------
[1] `E. W. Hansen and P.-L. Law, "Recursive methods for computing the Abel transform and its inverse", J. Opt. Soc. A2, 510-520 (1985) <http://dx.doi.org/10.1364/JOSAA.2.000510>`_
[2] `E. W. Hansen "Fast Hankel Transform" IEEE Trans. Acoust. Speech Signal
    Proc. 33, 666 (1985) <https://dx.doi.org/10.1109/TASSP.1985.1164579>`_
[3] J. R. Gascooke, PhD Thesis: *"Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals Molecule Dissociation"*, Flinders University (2000).
Available in `PDF format <https://github.com/PyAbel/abel_info/blob/master/Gascooke_Thesis.pdf>`
