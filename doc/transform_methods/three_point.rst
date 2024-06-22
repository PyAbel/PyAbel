Three Point
===========


Introduction
------------

The "Three Point" Abel transform method exploits the observation that the value of the Abel inverted data at any radial position *r* is primarily determined from changes in the projection data in the neighborhood of *r*. This technique was developed by Dasch [1]_.

How it works
------------

The projection data (raw data :math:`\mathbf{P}`) is expanded as a quadratic function of :math:`r - r_{j*}` in the neighborhood of each data point in :math:`\mathbf{P}`. 
In other words, :math:`\mathbf{P}'(r) = dP/dr` is estimated using a 3-point approximation (to the derivative), similar to central differencing.
Doing so enables an analytical integration of the inverse Abel integral around each point :math:`r_j`. 
The result of this integration is expressed as a linear operator :math:`\mathbf{D}`, operating on the projection data :math:`\mathbf{P}` to give the underlying radial distribution :math:`\mathbf{F}`.



When to use it
--------------

Dasch recommends this method based on its speed of implementation, robustness in the presence of sharp edges, and low noise.
He also notes that this technique works best for cases where the real difference between adjacent projections is much greater than the noise in the projections. This is important, because if the projections are oversampled (raw data :math:`\mathbf{P}` taken with data points very close to each other), the spacing between adjacent projections is decreased, and the real difference between them becomes comparable with the noise in the projections. In such situations, the deconvolution is highly inaccurate, and the projection data :math:`\mathbf{P}` must be smoothed before this technique is used. (Consider smoothing with `scipy.ndimage.gaussian_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html>`_.)


How to use it
-------------

To complete the inverse transform of a full image with the ``three_point method``, simply use the :class:`abel.Transform <abel.transform.Transform>` class::

    abel.Transform(myImage, method='three_point', direction='inverse').transform

Note that the forward Three point transform is not yet implemented in PyAbel.

If you would like to access the Three Point algorithm directly (to transform a right-side half-image), you can use :func:`abel.dasch.three_point_transform`.


Example
-------

.. plot:: ../examples/example_dasch_methods.py
    :include-source:


Notes
-----

The algorithm contained two typos in Eq. (7) in the original citation [1]_. A corrected form of these equations is presented in Karl Martin's 2002 PhD thesis [2]_. PyAbel uses the corrected version of the algorithm.

For more information on the PyAbel implementation of the three-point algorithm, please see `issue #61 <https://github.com/PyAbel/PyAbel/issues/61>`_ and `Pull Request #64 <https://github.com/PyAbel/PyAbel/pull/64>`_.


Citation
--------

.. |ref1| replace:: \ C. J. Dasch, "One-dimensional tomography: a comparison of Abel, onion-peeling, and filtered backprojection methods", `Appl. Opt. 31, 1146–1152 (1992) <https://doi.org/10.1364/AO.31.001146>`__.

.. |ref2| replace:: \ K. Martin, PhD Thesis: "Acoustic Modification of Sooting Combustion", University of Texas at Austin (2002) (`record <https://repositories.lib.utexas.edu/items/53b5dc6d-df47-41a0-a5b7-0552a3f0bf8b>`__, `PDF <https://repositories.lib.utexas.edu/server/api/core/bitstreams/f5a54b91-cb02-47f3-9cf7-c1189184b2ff/content>`__).

.. [1] |ref1|

.. [2] |ref2|

.. only:: latex

    * |ref1|
    * |ref2|
