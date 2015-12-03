# PyAbel

[![Build Status](https://travis-ci.org/PyAbel/PyAbel.svg?branch=master)](https://travis-ci.org/PyAbel/PyAbel)

PyAbel is a Python package for performing Abel and (primarily) inverse Abel transforms. The Abel transform takes a cylindrically symmetric 3D object and finds the 2D projection of that object. The more difficult problem -- the inverse Abel transform -- takes the 2D projection and finds the central slice of the 3D object by assuming cylindrical symmetry in the vertical direction.

The PyAbel package offers several options for completing the inverse Abel transform:

1) The BASEX algorithm creared by Dribinski, Ossadtchi, Mandelshtam, and Reisler [[Rev. Sci. Instrum. 73 2634, (2002)](http://dx.doi.org/10.1063/1.1482156)]. The BASEX implementation uses Gaussian basis functions to find the transform instead of analytically solving the inverse Abel transform.

2) The "Hansen and Law" recusrive method described in [[J. Opt. Soc. Am A 2 (4) 510 (1985)](dx.doi.org/10.1364/JOSAA.2.000510)]

3) In the future, we hope to have more options for the forward and inverse abel transforms.

### Symmetry

In this code, the axis of cylindrical symmetry is in assumed to be in the vertical direction. If this is not the case for your data, the `numpy.rot90` function can be used to rotate your dataset.

### Installation notes

This module requires Python 2.7 or 3.3-3.5. It can be installed with

    python setup.py install --user

Or, if you wish to edit the PyAbel code without re-installing each time (advanced users):

    python setup.py develop

### Example of use

See several example in `examples` folder.

### Contributing

We welcome new implementations of the inverse Abel transform or other code improvements. Please feel free to submit an issue or make a pull request.

Have fun!
