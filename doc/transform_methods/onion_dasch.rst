Onion Peeling (Dasch) 
=====================


Introduction
------------

The onion-peeling deconvolution method, as described by Dasch [1], is 
one of the simpler algorithms, which is computionally very efficient. Its
only draw-back is less accuracy/smoothness (yet to be confirmed) than 
the other algorithms.

See the discussion here: https://github.com/PyAbel/PyAbel/issues/56

How it works
------------

The Dasch algorithm effectively treats an image as a one-dimensional rows of pixels. A square operator matrix `D` accounts for all interactions between the
`i`-th pixel and non-`i`-th pixel in each row. The row-independence of the Abel transform is made explicit here.


When to use it
--------------

This is a very computationally efficient algorithm, but is stated to be
less smooth than more sophisticated methods, from the same Dasch stable, 
such as `three-point`.


How to use it
-------------

To complete the inverse transform of a full image with the
``onion_dasch`` method, simply use the :class:`abel.Transform`: class ::

    abel.Transform(myImage, method='onion_dasch').transform

If you would like to access the ``onion_dasch`` algorithm directly i
(to transform a right-side half-image), you can 
use :func:`abel.onion_dasch.onion_dasch_transform`.


Example
-------

.. plot:: ../examples/example_onion_dasch.py


Citation
--------
[1] https://www.osapublishing.org/ao/abstract.cfm?uri=ao-31-8-1146

