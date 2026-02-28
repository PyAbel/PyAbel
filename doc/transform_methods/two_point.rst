Two Point  (Dasch)
==================


Introduction
------------

The "Dasch two-point" deconvolution algorithm is one of several described in
the Dasch paper [1]_. See also the :doc:`“three-point” <three_point>` and
:doc:`“onion peeling” <onion_peeling>` descriptions.

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

To complete the inverse transform of a full image with the ``two_point method``, simply use the :class:`pyabel.Transform <pyabel.transform.Transform>` class::

    pyabel.Transform(myImage, method='two_point').transform

If you would like to access the ``two_point`` algorithm directly (to transform a right-side half-image), you can use :func:`pyabel.dasch.two_point_transform`.


Example
-------

.. plot:: ../examples/example_dasch_methods.py
    :include-source:


For more information on the PyAbel implementation of the ``two_point`` algorithm, please see `PR #155 <https://github.com/PyAbel/PyAbel/pull/155#issuecomment-200630188>`_.


Citation
--------

.. |ref1| replace:: \ C. J. Dasch, "One-dimensional tomography: a comparison of Abel, onion-peeling, and filtered backprojection methods", `Appl. Opt. 31, 1146–1152 (1992) <https://doi.org/10.1364/AO.31.001146>`__.

.. [1] |ref1|

.. only:: latex

    * |ref1|
