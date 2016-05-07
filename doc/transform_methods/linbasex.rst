.. |nbsp| unicode:: 0xA0 
   :trim:

Lin-Basex
=========


Introduction
------------

Inversion procedure based on 1-dimensional projections of VM-images as 
described in Gerber et al. [1]. 

[ *from the abstract* ]

*VM-images are composed of projected Newton spheres with a common centre. 
The 2D images are usually evaluated by a decomposition into base vectors each
representing the 2D projec- tion of a set of particles starting from a centre 
with a specific velocity distribution. We propose to evaluate 1D projections of
VM-images in terms of 1D projections of spherical functions, instead. 
The proposed evaluation algorithm shows that all distribution information can 
be retrieved from an adequately chosen set of 1D projections, alleviating the 
numerical effort for the interpretation of VM-images considerably. The obtained
results produce directly the coefficients of the involved spherical functions, 
making the reconstruction of sliced Newton spheres obsolete.*

How it works
------------

.. figure:: https://cloud.githubusercontent.com/assets/10932229/14975430/ea9c25de-1144-11e6-8824-531c81976160.png
   :width: 350px
   :alt: projection
   :align: right
   :figclass: align-center

   projections (Fig. 2 of [1])

A projection of 3D Newton spheres along the :math:`x` axis yields a compact 1D function:

.. math::

 L(z, u) = \sum_k \sum_\ell P_\ell(u)P_\ell\left(\frac{z}{r_k}\right) \frac{\prod_{r_k}(z)}{2r_k} p_{\ell k}

with :math:`u = \cos(\theta)`. This function constitutes a system of equations
expressing :math:`L(z, u)` as a linear combination of :math:`P_\ell(z/r_k)`. There
exists for a given base a unique set of coefficients :math:`p_{\ell k}` 
producing a least-squares fit to the function :math:`L(z, u)`.

|
|



[ *extract of a comment made by Thomas Gerber (method author)* ]

*Imaging an PES experiment which produces electrons that are distributed on the 
surface of a sphere. This sphere can be described by spherical functions. If 
all electrons have the same energy we expect them on a (Newton) sphere with 
radius* :math:`i`. *This radius is projected to the CCD. The distribution on 
the CCD has (if optics are approriate) the same radius* :math:`i`. 
*Now let us assume that the distribution on the Newton sphere has some 
anisotropy. We can describe the 
distribution on this sphere by spherical functions* :math:`Y_{nm}`. 
*Let's say* :math:`xY_{00} + yY_{20}`. 
*The 1D projection of those spheres produces just* :math:`xP_{i0}(k) +yP_{i2}(k)`
*where* :math:`P_{i}` *denotes Legendre Polynomials scaled to the interval* 
:math:`i` *and* :math:`k` *is the argument (pixel).*

*For one projection Lin-Basex now solves for the parameters* :math:`x` *and* 
:math:`y`. *If we look at another projection turned by an angle, the Basis* 
:math:`P_{i0}` *and* :math:`P_{i2}` 
*has to be modified because the projection of e.g.,* :math:`Y_{20}` *turned 
by an angle yields another function. It was shown that this function for e.g.,* 
:math:`P_{2}` *is just* 
:math:`P_{2}(a)P_{i2}(k)` *where* :math:`a` *is the turning angle. Solving 
the equations for the 1D projection at angle* (:math:`a`) *with this modified 
basis yields the same* :math:`x` and :math:`y` *parameters as before.*

*Lin-Basex aims at the determination of contributions in terms of spherical 
functions calculating the weight of each* :math:`Y_{l0}`. *If we reconstruct 
the 3D object by adding all the* :math:`Y_{l0}` *contributions we get the 
inverse Laplace transform of the image on the CCD from which we can derive 
"Slices".*


When to use it
--------------
[ *another extract from comments by the method author Thomas Gerber* ]

*The advantage of* ``linbasex`` *is, that not so many projections are needed 
(typically* :func:`len(an) ~ len(pol)`). *So,* ``linbasex`` *evaluation using a 
mathematically 
appropriate and correct basis set should eventually be much faster 
than* ``basex``. 

*If our 3D object is "sparse" (i.e., contains a sparse set of Newton spheres) a 
sparse basis may be used. In this case one must have primary information about 
what "sparsity" is appropriate.*

*That means that an Abel transform may be simplified if primary information 
about the object is available. That is not the case with the other methods.*

*Absolute noise increases in each sphere with sqrt(counts) but relative noise 
decreases with* :math:`1/\sqrt{\text{counts}}`. 


How to use it
-------------

To complete the inverse Abel transform of a full image with the 
``linbasex method``, simply use the :class:`abel.Transform`: class ::

    abel.Transform(myImage, method='linbasex').transform


Note, the parameter :attr:`transform_options=dict(return_Beta=True)`, 
provides additional attributes, direct from the transform procedure:
 - ``.linbasex_angular_integration`` - the speed distribution
 - ``.linbasex_radial_integration`` - the anisotropy parameter vs radius
 - ``.linbasex_radial`` - the radial array
A more complete global call, that centers the image, ensures that the size is odd,
and returns the attributes above, would be e.g. ::

    abel.Transform(myImage, method='linbasex', center='convolution',
                   transform_options=dict(return_Beta=True)) 

Alternatively, the linbasex algorithm :func:`abel.linbasex.linbasex_transform_full()` directly 
transforms the full image, with the attributes returned as a tuple in this case.

Tips
----

Including more projection (angles) may improve the transform: ::
   
  an = [0, 45, 90, 135]

or ::

  an = arange(0, 180, 10)

Example
-------

.. plot:: ../examples/example_linbasex.py

Historical
----------

PyAbel python code was extracted from this `jupyter notebook <https://www.psi.ch/sls/vuv/Station1_IntroEN/Lin_Basex0.7.zip>`_ supplied by Thomas Gerber.


Citation
--------
[1] `Gerber, Thomas, Yuzhu Liu, Gregor Knopp, Patrick Hemberger, Andras Bodi, Peter Radi, and Yaroslav Sych, "Charged Particle Velocity Map Image Reconstruction with One-Dimensional Projections of Spherical Functions.” Rev. Sci. Instrum. 84, no. 3, 033101 (2013) <http://dx.doi.org/10.1063/1.4793404>`_

