Anisotropy Parameter
====================

For linearly polarized light the angular distribution of photodetached electrons from negative ions is given by

.. math::

  I(\epsilon, \theta) = \frac{\sigma_\text{total}(\epsilon)}{4\pi} [ 1 + \beta(\epsilon) P_2(\cos\theta)],

where :math:`\beta(\epsilon)` is the electron kinetic energy (:math:`\epsilon`) dependent anisotropy parameter, which varies between −1 and +2, and :math:`P_2(\cos\theta)` is the 2nd-order Legendre polynomial in :math:`\cos\theta`. :math:`\sigma_\text{total}` is the total photodetachment cross section. The anisotropy parameter provides phase information about the dynamics of the photon process [1].


Methods
-------

``PyAbel`` provides several methods to determine the anisotropy parameter :math:`\beta`:

   Method 1: :doc:`linbasex <transform_methods/linbasex>` evaluates :math:`\beta` directly, available as the class attribute `Beta[1]`.

       This method fits spherical harmonic functions to the velocity-map image to directly determine the anisotropy parameter as a function of the radial coordinate. This parameter has greater uncertainty in radial regions of low intensity, and so it is commonly plotted as the product :math:`I \times \beta`.  See ``examples/example_linbasex.py``.

   .. image:: https://cloud.githubusercontent.com/assets/10932229/17164544/94adacdc-540c-11e6-955a-c5c9092943cc.png
      :width: 600px
      :alt: example_linbasex output image

   |

   Method 2: using :func:`abel.tools.vmi.radial_integration`.

       This method determines the anisotropy parameter from the inverse Abel-transformed image, by extracting intensity vs angle for each specified radial range and then fitting the intensity formula given above. This method is best applied to the radial ranges corresponding to strong spectral intensity in the image. It has the advantage of providing the least-squares fit error estimate for the parameter(s).

   Method 3: using :class:`abel.tools.vmi.Distributions`.

       This method, like the previous one, works on the inverse Abel-transformed image, but fits the angular intensity dependence at each radius, providing radially dependent anisotropy parameters, like in the first method. If the anisotropy parameters are known to be smooth radial functions, a moving-window averaging can be employed for noise reduction.


Example
-------

See :doc:`example_anisotropy_parameter`. In this case the anisotropy parameter is determined using each method. Note:
 
   * In method 1, the filter parameter ``threshold=0.2`` is set to a larger value so as to exclude evaluation in regions of weak intensity.

   * Method 2 evaluates the anisotropy parameter for particular radial regions of strong intensity.

   * In method 3, the anisotropy parameter is calculated with 9-pixel radial averaging and plotted only in the regions with > 1 % of the maximal intensity.

.. plot:: ../examples/example_anisotropy_parameter.py 


Reference
---------

[1] `J. Cooper and R. N. Zare, "Angular distribution of photoelectrons", J. Chem. Phys. 48, 942 (1968) <http://scitation.aip.org/content/aip/journal/jcp/48/2/10.1063/1.1668742>`_
