Onion Peeling 
=============


Introduction
------------

The onion peeling method, also known as "back projection" has been 
ported to Python from the original Matlab implementation, created by 
Chris Rallis and Eric Wells of Augustana University, and described in 
this paper [1]. The algorithm actually originates from this 1996 RSI paper 
by Bordas ~et al.[2]

See the discussion here: https://github.com/PyAbel/PyAbel/issues/56

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



How to use it
-------------

To complete the inverse transform of a full image with the
``onion peeling method``, simply use the :class:`abel.Transform`: class ::

    abel.Transform(myImage, method='onion_peeling').transform

If you would like to access the onion-peeling algorithm directly (to transform a
right-side half-image), you can use :func:`abel.onion_peeling.onion_peeling_transform`.


Example
-------

.. plot:: ../examples/example_onion.py


Citation
--------
[1] http://scitation.aip.org/content/aip/journal/rsi/85/11/10.1063/1.4899267

[2] http://scitation.aip.org/content/aip/journal/rsi/67/6/10.1063/1.1147044
