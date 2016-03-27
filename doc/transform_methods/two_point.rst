Two Point  (Dasch)
==================


Introduction
------------

The "Dasch two-point" deconvolution algorithm is one of several
described in the Dasch [1] paper. See also the ``three_point`` and 
``onion_dasch`` descriptions.

How it works
------------

The Abel integral is broken into intervals between the :math:`r_j` 
points, and :math:`P^\prime(r)` is assumed constant between :math:`r_j` and
:math:`r_{j+1}`. 

When to use it
--------------

This method is simple and computationally very efficient. The method
incorporates no smoothing.


How to use it
-------------

To complete the inverse transform of a full image with the ``two_point method``, simply use the :class:`abel.Transform` class: ::

    abel.Transform(myImage, method='two_point').transform

If you would like to access the ``two_point`` algorithm directly (to transform a right-side half-image), you can use :func:`abel.two_point.two_point_transform`.


Example
-------

.. plot:: ../examples/example_two_point.py
    :include-source:


or more information on the PyAbel implementation of the ``two_point`` algorithm, please see `Pull Request #155 <https://github.com/PyAbel/PyAbel/pull/155#issuecomment-200630188>`_.



Citation
--------
[1] `Dasch, Applied Optics, Vol 31, No 8, March 1992, Pg 1146-1152 <(http://dx.doi.org/10.1364/AO.31.001146>`_.

