Anisotropy Parameter
====================

For linearly polarized light the angular distribution of photodetached electrons from negative-ions is given by:

.. math::

  I(\epsilon, \theta) = \frac{\sigma_{\rm total}(\epsilon)}{4\pi} [ 1 + \beta(\epsilon) P_2(\cos\theta)]


where :math:`\beta(\epsilon)` is the electron kinetic energy anisotropy parameter, that varies between -1 and +2, and :math:`P_2(\cos\theta)` is the 2nd order Legendre polynomial in :math:`\cos\theta`. :math:`\sigma_{\rm total}` is the total photodetachment cross section.


``PyAbel`` provides two methods to determine the anisotropy parameter :math:`\beta`:

   1. :doc:`linbasex <transform_methods/linbasex>` directly evaluates :math:`\beta`, available as the class attribute `Beta[1]`.

       This method fits spherical harmonic functions to the velocity-image to directly determine the anisotropy parameter as a function of the radial coordinate.


   2. `abel.tools.vmi.radial_integration()` 

       This method determines the anisotropy parameter from the inverse Abel transformed image, by extracting intensity vs angle for each specified radial range (tuples) and then fitting the intensity formula given above. This method is best applied to the radial ranges the correspond to strong spectral (intensity) in the image. It has the advantage of providing the least-squares fit error estimate for the parameter(s).



Example
-------

.. plot:: ../examples/example_PAD.py 
