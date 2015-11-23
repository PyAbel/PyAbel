# PyAbel

[![Build Status](https://travis-ci.org/PyAbel/PyAbel.svg?branch=master)](https://travis-ci.org/PyAbel/PyAbel)

PyAbel is a Python package for performing Abel and (primarily) inverse Abel transforms. The Abel transform takes a 3D object and finds the 2D projection of that object. The more difficult problem -- the inverse Abel transform -- takes the 2D projection and finds the central slice of the 3D object by assuming cylindrical symmetry in the vertical direction.

Currently, this package only uses the BASEX algorithm creared by Dribinski, Ossadtchi, Mandelshtam, and Reisler [[Rev. Sci. Instrum. 73 2634, (2002)](http://dx.doi.org/10.1063/1.1482156)], but we hope to include more algorithms for the inverse Abel transform in the future! 

The BASEX implementation uses Gaussian basis functions to find the transform instead of analytically solving the inverse Abel transform or applying the Fourier-Hankel method, as both the analytical solution and the Fourier-Hankel methods provide lower quality transforms when applied to real-world datasets (see the RSI paper). The BASEX implementation is quick, robust, and is probably the most common method used to transform velocity-map-imaging (VMI) datasets.

In this code, the axis of cylindrical symmetry is in assumed to be in the vertical direction. If this is not the case for your data, the `numpy.rot90` function may be useful.

### Installation notes

This module requires Python 2.7 or 3.3-3.5. It can be installed with

    python setup.py install --user
	

### Example of use

See an  example in `examples/example_main.py`.

Have fun!
