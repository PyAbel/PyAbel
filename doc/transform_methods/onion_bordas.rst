Onion Peeling (Bordas)
======================


Introduction
------------

The onion peeling method, also known as "back projection" has been 
ported to Python from the original Matlab implementation, created by 
Chris Rallis and Eric Wells of Augustana University, and described in 
[1]_. The algorithm actually originates from Bordas ~et al. [2]_.

See the discussion in `issue #56 <https://github.com/PyAbel/PyAbel/issues/56>`_.


How it works
------------

This algorithm calculates the contributions of particles, at a given 
kinetic energy, to the signal in a given pixel (in a row). This signal is 
then subtracted from the projected (experimental) pixel and also added 
to the back-projected image pixel. The procedure is repeated until the 
center of the image is reached. The whole procedure is done for each pixel 
row of the image.


When to use it
--------------

This is a historical implementation of the onion-peeling method. 


How to use it
-------------

To complete the inverse transform of a full image with the
``onion_bordas`` method, simply use the :class:`abel.Transform`: class ::

    abel.Transform(myImage, method='onion_bordas').transform

If you would like to access the onion-peeling algorithm directly 
(to transform a right-side half-image), you can 
use :func:`abel.onion_bordas.onion_bordas_transform`.


Example
-------

.. plot:: ../examples/example_onion_bordas.py

:doc:`Source code </example_onion_bordas>`


Citation
--------

.. |ref1| replace:: \ C. E. Rallis, T. G. Burwitz, P. R. Andrews, M. Zohrabi, R. Averin, S. De, B. Bergues, B. Jochim, A. V. Voznyuk, N. Gregerson, B. Gaire, I. Znakovskaya, J. McKenna, K. D. Carnes, M. F. Kling, I. Ben-Itzhak, E. Wells, "Incorporating real time velocity map image reconstruction into closed-loop coherent control", `Rev. Sci. Instrum. 85, 113105 (2014) <https://doi.org/10.1063/1.4899267>`__.

.. |ref2| replace:: \ C. Bordas, F. Paulig, "Photoelectron imaging spectrometry: Principle and inversion method", `Rev. Sci. Instrum. 67, 2257–2268 (1996) <https://doi.org/10.1063/1.1147044>`__.

.. [1] |ref1|

.. [2] |ref2|

.. only:: latex

    * |ref1|
    * |ref2|
