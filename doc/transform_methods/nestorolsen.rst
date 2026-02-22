Nestor Olsen
==================


Introduction
------------

The "nestorolsen" deconvolution algorithm is described in the Nestor, Olsen paper [1]_.

How it works
------------

The projection data (raw data :math:`\mathbf{P}`) is expanded as a quadratic function of :math:`r - r_{j*}` in the neighborhood of each data point in :math:`\mathbf{P}`. 
In other words, :math:`\mathbf{P}'(r) = dP/dr` is estimated using a 3-point approximation (to the derivative), similar to central differencing.
In contrast to "Three point" method the coefficient of the linear term is assumed to be zero.
Doing so enables an analytical integration of the inverse Abel integral around each point :math:`r_j`. 
The result of this integration is expressed as a linear operator :math:`\mathbf{D}`, operating on the projection data :math:`\mathbf{P}` to give the underlying radial distribution :math:`\mathbf{F}`.

When to use it
--------------

This method is simple and computationally very efficient. The method
incorporates no smoothing. It can be thought of as a something between "Three point" and "Two point" methods.


How to use it
-------------

To complete the inverse transform of a full image with the ``nestorolsen method``, simply use the :class:`abel.Transform <abel.transform.Transform>` class::

    abel.Transform(myImage, method='nestorolsen').transform

If you would like to access the ``nestorolsen`` algorithm directly (to transform a right-side half-image), you can use :func:`abel.nestorolsen.nestorolsen_transform`.


Citation
--------

.. |ref1| replace:: \ O. H. Nestor and H. N. Olsen, "Numerical Methods for Reducing Line and Surface Probe Data", `SIAM Review, vol. 2, no. 3, 1960, pp. 200–07 <https://doi.org/10.1137/1002042>`__.

.. [1] |ref1|

.. only:: latex

    * |ref1|
