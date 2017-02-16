Circularize Image
=================

Background
----------

For velocity-map imaging experiments the charged particles emanate as Newton spheres that have spherical symmetry, and thus the 2D projection should be perfectly circular. Experimental velocity-map images may have distortions in the circular charged particle energy structure, due to, for example, stray magnetic fields, or optical distortion of the camera lens that images the particle detector. The effect of distortion is to degrade the radial (or velocity or kinetic energy) resolution, since a particular energy peak will "walk" in radial position, depending on the particular angular position on the detector. Imposing a physical circular distribution of particles, may substantially improve the kinetic energy resolution, at the expense of uncertainly in the absolution kinetic-energy position of the transition.

Approach
--------

Compare the radial positions of strong features in angular slice intensity profiles. i.e. follow the radial position of a peak as a function of angle. A linear correction is applied to the radial grid to align the peak at each angle.
::
     before     after
       ^         ^    slice0
         ^       ^    slice1
       ^         ^    slice2
      ^          ^    slice3
       :         :    
        ^        ^    slice#
    radial peak position

Peak alignment is achieved through a radial scaling factor :math:`R_i(actual) = R_i \times scalefactor_i`. The scalefactor is determined by a choice or methods, ``argmax``, where :math:`scalefactor_i = R_0/R_i`, with :math:`R_0` a reference peak. Or ``lsq``, which directly determines the radial scaling factor that best aligns adjacent slice intensity profiles.

Implementation
--------------

Cartesian :math:`(y, x)` image is converted to a polar coordinate image :math:`(r, \theta)` for easy slicing into angular blocks. Each radial intensity profile is compared with its adjacent slice, providing a radial scaling factor that best aligns the two intensity profiles. 

The set of radial scaling factors, for each angular slice, is then spline 
interpolated to correct the :math:`(y, x)` grid, and the image remapped to an
unperturbed grid.

How to use it
-------------
The ``circularize_image()`` function is called directly ::

 IMcirc, angle, radial_correction, radial_correction_function =\
     abel.tools.circularize.circularize_image(IM, method='lsq',\
     center='slice', dr=0.5, dt=0.1, return_correction=True)

The main input parameters are the image `IM`, and the number of angular slices, to use, which is set by :math:`2\pi/dt`. The default `dt = 0.1` uses ~63 slices.
This parameter determines the angular resolution of the distortion correction
function, but is limited by the signal to noise loss with smaller `dt`.
Other parameters may help better define the radial correction function.

Warning
-------
Ensure the returned radial_correction vs angle data is a well behaved function. 
See the example, below, bottom left figure. If necessary limit the ``radial_range=(Rmin, Rmax)``, or change the value of the spline smoothing parameter.

Example
-------

.. plot:: ../examples/example_circularize_image.py
    :include-source:
