Circularize Image
=================

Background
----------
For velocity-map imaging experiments the charged particles emanate as Newton spheres that have spherical symmetry, and thus the 2D projection should be perfectly circular. Experimental velocity-map images may have distortion in the circular charged particle energy structure, due to, for example, stray magnetic fields, or optical distortion of the camera lens that images the particle detector. The effect of distortion is to reduce the radial (or velocity or kinetic energy) resolution, since a particular energy peak will "walk" in radial position, depending on the particular angular position on the detector. Imposing a physical circular distribution of particles, improves the kinetic energy resolution, at the expense of uncertainly in the absolution kinetic-energy position of the transition.

Approach
--------
Compare the radial positions of features in angular slice profiles. i.e. follow the radial position of a peak as a function of angle. A linear correction is applied to the radial grid to align the peak at each angle.

Implementation
--------------
Cartesian :math:`(y, x)` image is converted to a polar coordinate image :math:`(r, theta)` for easy slicing into angular blocks. Each radial intensity profile is compared with its adjacent slice, providing a radial scaling factor that best aligns the two intensity profiles. Two methods are available, `method='argmax'` determines the radial position of the intensity maxium. `method='lsq'` least-squares determines a radial scaling factor that best aligns the intensity profiles.

::
 before          after
   ^               ^         slice 0
      ^            ^         slice 1
     ^             ^         slice 2
      ^            ^         slice 3
 radial position

The radial scaling factor of each slice is spline interpolated to provide the real cartesian grid, which is then used to interpolate the original image.

How to use it
-------------
::
 IMcirc, angle, radial_correction, splinefunction = abel.tools.circularize.circularize_image(IM, method='lsq', center='slice', nslices=32, zoom=2i, return_correction=True)

Warning
-------
Ensure the returned radial_correction vs angle data is a well behaved function. 
See the example (below). If necessary limit the `radial_range=(Rmin, Rmax)`, or change the value of the spline smoothin parameter.

Example
-------

.. plot:: ../examples/example_circularize_image.py
    :include-source:
