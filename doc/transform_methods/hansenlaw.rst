Hansen-Law
==========
.. _hansenlaw:


Introduction
------------

The Hansen and Law transform is the work of E. W. Hansen and P.-L. Law [1].

From the abstract:

... new family of algorithms, principally for Abel inversion, that are recursive and hence computationally efficient. The methods are based on a linear, space-variant, state-variable model of the Abel transform. The model is the basis for deterministic algorithms, applicable when data are noise free, and least-squares estimation (Kalman filter) algorithms, which accommodate the noisy data case.

The key advantage of the algorithm is its computational simplicity that amounts to only a few lines of code. 



How it works
------------

.. image:: https://cloud.githubusercontent.com/assets/10932229/13250293/40a333c2-da7d-11e5-9647-d8404a12626a.png
   :width: 650px
   :alt: Path of integration

The Hansen and Law method makes use of a coordinate transformation, which is used to model the Abel transform, and derive *reconstruction* filters. The Abel transform is treated as a system modeled by a set of linear differential equations. 

.. image:: https://cloud.githubusercontent.com/assets/10932229/13251144/88f7a1d0-da82-11e5-8c09-7bf2dc4be830.png
   :width: 550px
   :alt: hansenlaw-iteration

The algorithm iterates along each row of the image, starting at the out edge, and ending at the center.

to be continued...


When to use it
--------------

The Hansen-Law algorithm offers one of the fastest, most robust methods for both the forward and inverse transforms. It requires reasonably fine sampling of the data to provide exact agreement with the analytical result, but otherwise this method is a hidden gem of the field.


How to use it
-------------

To complete the forward or inverse transform of a full image with the ``hansenlaw method``, simply use the :func:`abel.transform` function: ::

	abel.transform(myImage, method='hansenlaw', direction='forward')
	abel.transform(myImage, method='hansenlaw', direction='inverse')
	

If you would like to access the Hansen-Law algorithm directly (to transform a right-side half-image), you can use :func:`abel.hansenlaw.hansenlaw_transform`.


Historical Note
---------------

The Hansen and Law algorithm was almost lost to the scientific community. It was rediscovered by Jason Gascooke (Flinders University, South Australia) for use in his velocity-map image analysis and written up in his PhD thesis: 

J. R. Gascooke, PhD Thesis: *"Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals Molecule Dissociation"*, Flinders University (2000).
Unfortunately, not available in electronic format.



Citation
--------
[1] `E. W. Hansen and P.-L. Law, "Recursive methods for computing the Abel transform and its inverse", J. Opt. Soc. A2, 510-520 (1985) <http://dx.doi.org/10.1364/JOSAA.2.000510>`_
