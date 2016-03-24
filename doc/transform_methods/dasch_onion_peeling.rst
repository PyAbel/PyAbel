Onion Peeling (Dasch)
=====================


Introduction
------------

The "Dasch onion peeling" deconvolution algorithm is one of several
described in the Dasch [1] paper. See also the ``three_point``
description.

How it works
------------

In the onion-peeling method the projection is approximated by rings
of constant property between 
:math:`r_j - \Delta r/2` and :math:`r_j + \Delta r/2` for each data 
point :math:`r_j`.

The projection data is given by :math:`P(r_i) = \Delta r \sum_{j=i}^\infty W_{ij} F(r_j)`

where 

.. math::

    W_{ij} = \left{ \begin{array}{c}
                0   (j < i) \\
                \sqrt{(2j+1)^2 - 4i^2} (j=i)\\
                \sqrt{(2j+1)^2 - 4i^2} - \sqrt{(2j-1)^2 - 4i^2} (j > i)
                \end{array}\right.

The onion-peeling deconvolution function is: :math:`D_{ij} = (W^{-1})_{ij}`.


When to use it
--------------

This method has less smoothing that other methods (discussed in Dasch),
however it is computationally simple.


How to use it
-------------

To complete the inverse transform of a full image with the ``dasch_onion_peeling method``, simply use the :class:`abel.Transform` class: ::

    abel.Transform(myImage, method='dasch_onion_peeling', direction='inverse').transform

Note that the forward Three point transform is not yet implemented in PyAbel.

If you would like to access the Three Point algorithm directly (to transform a right-side half-image), you can use :func:`abel.three_point.three_point_transform`.


Example
-------

.. plot:: ../examples/example_dasch_onion_peeling.py
    :include-source:


or more information on the PyAbel implementation of the ``dasch_onion_peeling`` algorithm, please see `Pull Request #155 <https://github.com/PyAbel/PyAbel/pull/155>`_.



Citation
--------
[1] `Dasch, Applied Optics, Vol 31, No 8, March 1992, Pg 1146-1152 <(http://dx.doi.org/10.1364/AO.31.001146>`_.
