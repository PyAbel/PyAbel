Three Point
===========


Introduction
------------

The "Three Point" Abel transform method exploits the observation that the value of the Abel inverted data at any radial position ``r`` is primarily determined from changes in the projection data in the neighborhood of ``r``. This technique was developed by Dasch [1].

How it works
------------

The projection data (raw data :math:`\mathbf{P}`) is expanded as a quadratic function of :math:`r - r_{j*}` in the neighborhood of each data point in :math:`\mathbf{P}`. 
In other words, :math:`\mathbf{P}'(r) = dP/dr` is estimated using a 3-point approximation (to the derivative), similar to central differencing.
Doing so enables an analytical integration of the inverse Abel integral around each point :math:`r_j`. 
The result of this integration is expressed as a linear operator :math:`\mathbf{D}`, operating on the projection data :math:`\mathbf{P}` to give the underlying radial distribution :math:`\mathbf{F}`.



When to use it
--------------

Dasch recommends this method based on its speed of implementation, robustness in the presence of sharp edges, and low noise.
He also notes that this technique works best for cases where the real difference between adjacent projections is much greater than the noise in the projections. This is important, because if the projections are oversampled (raw data :math:`\mathbf{P}` taken with data points very close to each other), the spacing between adjacent projections is decreased, and the real difference between them becomes comparable with the noise in the projections. In such situations, the deconvolution is highly inaccurate, and the projection data :math:`\mathbf{P}` must be smoothed before this technique is used. (Consider smoothing with `scipy.ndimage.filters.gaussian_filter <http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html>`_.)


How to use it
-------------

To complete the inverse transform of a full image with the ``hansenlaw method``, simply use the :func:`abel.transform` function: ::

	abel.transform(myImage, method='three_point', direction='inverse')
	
Note that the forward Three point transform is not yet implemented in PyAbel.

If you would like to access the Three Point algorithm directly (to transform a right-side half-image), you can use :func:`abel.three_point.three_point_transform`.


Example
-------

.. plot:: ../examples/example_three_point.py
	:include-source:


Notes
-----

The algorithm contained two typos in Eq (7) in the original citation [1]. A corrected form of these equations is presented in Karl Martin's 2002 PhD thesis [2]. PyAbel uses the corrected version of the algorithm.

For more information on the PyAbel implementation of the three-point algorithm, please see `Issue #61 <https://github.com/PyAbel/PyAbel/issues/61>`_ and `Pull Request #64 <https://github.com/PyAbel/PyAbel/pull/64>`_.



Citation
--------
[1] `Dasch, Applied Optics, Vol 31, No 8, March 1992, Pg 1146-1152 <(http://dx.doi.org/10.1364/AO.31.001146>`_.

[2] Martin, Karl. PhD Thesis, University of Texas at Austin. Acoustic Modification of Sooting Combustion. 2002: https://www.lib.utexas.edu/etd/d/2002/martinkm07836/martinkm07836.pdf

