Nestor–Olsen
==================


Introduction
------------

The algorithm for integration of the inverted Abel integral equations described by Nestor and Olsen [1]_.


How it works
------------

The inverse Abel transform integral

.. math::
    f(r) = -\frac{1}{\pi} \int_r^R \frac{Q'(x)\,dx}{(x^2 - r^2)^{1/2}},

where :math:`Q(x)` is the measured "line probe" (projection) data, and :math:`f(r)` is the point function (original distribution), is transformed into

.. math::
    f(r[v]) = -\frac{1}{\pi} \int_v^{R^2} \frac{Q'(u)\,du}{(u - v)^{1/2}}

by substituting the variables :math:`v = r^2`, :math:`u = x^2`. Then, assuming that :math:`Q(u)` is linear in :math:`u` (that is, in :math:`x^2`) within each sampling interval, the corresponding samples of :math:`f(r)` can be expressed analytically as

.. math::
    f_k \equiv f(r = k\Delta r) = -\frac{2}{\pi\Delta r} \sum_{n=k}^{N-1} A_{k,n}[Q_{n+1} - Q_n],

where :math:`Q_n \equiv Q(x = n \Delta r)` are the measured data samples, and

.. math::
    A_{k,n} = \frac{[(n + 1)^2 - k^2]^{1/2} - [n^2 - k^2]^{1/2}}{2n + 1}.

Instead of taking the finite difference of the experimental data, the method actually precomputes the difference coefficients

.. math::
    B_{k,n} = \begin{cases}
        -A_{k,k} & \text{for}\ n = k, \\
        A_{k,n-1} - A_{k,n} & \text{for}\ n \geqslant k + 1
    \end{cases}

and uses them to perform the inverse Abel transform as

.. math::
    f_k = -\frac{2}{\pi\Delta r} \sum_{n=k}^N B_{k,n} Q_n.

PyAbel also allows performing the forward Abel transform by solving the above matrix equation for :math:`Q_n`, given :math:`f_k` as the input data.


When to use it
--------------

This method is simple and computationally efficient; it can be thought of as the Dasch "two-point" method in different coordinates. The method incorporates no smoothing.


How to use it
-------------

To complete the inverse transform of a full image with the ``nestorolsen`` method, simply use the :class:`abel.Transform <abel.transform.Transform>` class::

    abel.Transform(myImage, method='nestorolsen').transform

If you would like to access the ``nestorolsen`` algorithm directly (to transform a right-side half-image), you can use :func:`abel.nestorolsen.nestorolsen_transform`.


Citation
--------

.. |ref1| replace:: \ O. H. Nestor, H. N. Olsen, "Numerical methods for reducing line and surface probe data", `SIAM Rev. 2(3), 200–207 (1960) <https://doi.org/10.1137/1002042>`__.

.. [1] |ref1|

.. only:: latex

    * |ref1|
