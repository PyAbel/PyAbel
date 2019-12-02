Onion Peeling (Dasch)
=====================


Introduction
------------

The "Dasch onion peeling" deconvolution algorithm is one of several
described in the Dasch paper [1]_. See also the ``two_point`` and
``three_point`` descriptions.

How it works
------------

In the onion-peeling method the projection is approximated by rings
of constant property between
:math:`r_j - \Delta r/2` and :math:`r_j + \Delta r/2` for each data
point :math:`r_j`.

The projection data is given by :math:`P(r_i) = \Delta r \sum_{j=i}^\infty W_{ij} F(r_j)`

where

.. math:: W_{ij} = 0 \, \, (j < i)

       \sqrt{(2j+1)^2 - 4i^2} \, \, (j=i)

       \sqrt{(2j+1)^2 - 4i^2} - \sqrt{(2j-1)^2 - 4i^2} \, \, (j > i)


The onion-peeling deconvolution function is: :math:`D_{ij} = (W^{-1})_{ij}`.


When to use it
--------------

This method is simple and computationally very efficient. The article
states that it has less smoothing that other methods (discussed in Dasch).


How to use it
-------------

To complete the inverse transform of a full image with the ``onion_dasch`` method, simply use the :class:`abel.Transform` class: ::

    abel.Transform(myImage, method='onion_peeling').transform

If you would like to access the ``onion_peeling`` algorithm directly (to transform a right-side half-image), you can use :func:`abel.dasch.onion_peeling_transform`.


Example
-------

.. plot:: ../examples/example_dasch_methods.py
    :include-source:


or more information on the PyAbel implementation of the ``onion_peeling`` algorithm, please see `PR #155 <https://github.com/PyAbel/PyAbel/pull/155>`_.



Citation
--------

.. [1] \ C. J. Dasch, "One-dimensional tomography: a comparison of Abel, onion-peeling, and filtered backprojection methods", `Appl. Opt. 31, 1146–1152 (1992) <https://doi.org/10.1364/AO.31.001146>`_.
