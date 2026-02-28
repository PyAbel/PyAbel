Direct
======


Introduction
------------

This method attempts a direct integration of the Abel transform integral. It makes no assumptions about the data (apart from cylindrical symmetry), but it typically requires fine sampling to converge. Unlike other methods implemented in PyAbel, it can work with the data sampled on a non-uniform grid. Such methods are typically inefficient, but thanks to this Cython implementation (by Roman Yurchak), this "direct" method is competitive with the other methods.


How it works
------------

The 1D forward and inverse Abel transforms

.. math::
    F(y) = 2 \int_y^\infty \frac{f(r)\,r\,dr}{\sqrt{r^2 - y^2}}, \quad
    f(r) = -\frac1\pi \int_r^\infty \frac{dF(y)}{dy} \frac{dy}{\sqrt{y^2 - r^2}}

can be expressed in the unified form

.. math::
    G(x) = \int_x^\infty \frac{g(r)\,dr}{\sqrt{r^2 - x^2}}, \quad
    g(r) = \begin{cases}
        2rf(r) & \text{forward transform}, \\[0.5ex]
        -\dfrac1\pi \dfrac{F(r)}{dr} & \text{inverse transform}.
    \end{cases}

The "direct" method prepares the corresponding :math:`g(r)` from the input data. In case of the inverse transform, the derivative is taken numerically using :func:`numpy.gradient` or any user-defined numerical differentiation function.

The integration is then performed numerically using the `trapezoidal rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_ (by default) or any user-defined numerical integration function. The numerical integration had to exclude the segment at the lower limit, where :math:`r = x`, and thus the integrand becomes infinite. This causes a systematic underestimation of the integral. To correct for this underestimation, the truncated numerical integral is by default supplemented with an analytical integral over the excluded segment, assuming that :math:`g(r)` is locally linear:

.. math::
    g(r_i + dr) = g(r_i) + g'(r_i)\,dr, \quad
    g'(r_i) = \frac{g(r_{i+1}) - g(r_i)}{r_{i+1} - r_i},

so that

.. math::
    \begin{aligned}
        \int_{x_i}^{x_{i+1}} \frac{g(r)\,dr}{\sqrt{r^2 - x_i^2}} &=
        \operatorname{arccosh}\frac{x_{i+1}}{x_i} \cdot g(x_i) +
            \left(y - x_i \operatorname{arccosh}\frac{x_{i+1}}{x_i}\right) g'(x_i) = \\
        &= \frac{[g(x_i) x_{i+1} + g(x_{i+1}) x_i] \ln\frac{x_{i+1} + y}{x_i} +
                 [g(x_{i+1}) - g(x_i)] y}
                {x_{i+1} - x_i},
    \end{aligned}

where :math:`y = \sqrt{r^2 - x_i^2}` with :math:`r = x_{i+1}`. The axial pixel has :math:`x_0 = 0`, so the above expressions also have singularities, but their analytical limit is simply

.. math::
    \int_0^{x_1} \frac{g(r)\,dr}{\sqrt{r^2 - 0^2}} = g'(0) x_1 = g(x_1),

necessarily assuming that :math:`g(0) = 0`, which is true for well-behaved functions :math:`f` and :math:`F` in the forward and inverse Abel transforms.


When to use it
--------------

When a robust forward transform is required, this method works quite well. It is not typically recommended for the inverse transform, but it can work well for smooth functions that are finely sampled. The sampling does not need to be uniform, so more points can be allocated to more important areas. Note, however, that :class:`pyabel.Transform <pyabel.transform.Transform>` and image-processing tools in PyAbel work only with uniform sampling, thus if you need to use them, it is recommended to resample the original data on a sufficiently fine uniform grid and use one of the more efficient transform methods.


How to use it
-------------

To complete the forward or inverse transform of a full image with the direct method, simply use the :class:`pyabel.Transform <pyabel.transform.Transform>` class::

    pyabel.Transform(myImage, method='direct', direction='forward').transform
    pyabel.Transform(myImage, method='direct', direction='inverse').transform


If you would like to access the Direct algorithm directly (to transform a right-side half-image, possibly with non-uniform sampling), you can use :func:`pyabel.direct.direct_transform`.


Examples
--------


Incomplete data
^^^^^^^^^^^^^^^

The transform integral for any :math:`x` depends only on the function at :math:`r \geqslant x`, thus it is possible to transform incomplete data, with the signal near the symmetry axis unavailable or contaminated. (See also :doc:`../example_rbasex_block`.)

.. plot:: ../examples/example_direct_rmin.py

.. admonition:: Source code ▾
    :collapsible: closed

    .. literalinclude:: /../examples/example_direct_rmin.py


Non-uniform sampling
^^^^^^^^^^^^^^^^^^^^

Forward and inverse Abel transforms of a Gaussian function, using denser sampling in the regions where the function curvature is larger. Only 30 samples are used here to make individual points discernible, but using more points will make the transform more accurate.

.. plot:: ../examples/example_direct_nonuniform.py

.. admonition:: Source code ▾
    :collapsible: closed

    .. literalinclude:: /../examples/example_direct_nonuniform.py


Custom differentiation and integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned above, the default numerical differentiation and integration methods can be replaced by user-defined Python functions. This example estimates the derivative from a smoothing spline fit to the noisy data and uses the composite `Simpson's rule <https://en.wikipedia.org/wiki/Simpson's_rule>`_ for numerical integration.

.. plot:: ../examples/example_direct_custom.py

.. admonition:: Source code ▾
    :collapsible: closed

    .. literalinclude:: /../examples/example_direct_custom.py

.. tip::
    The ``derivative`` function is called only once for the input image, but the ``integral`` function is called for each column of the transformed image and thus should be implemented efficiently, as otherwise the whole transform might become very slow.
