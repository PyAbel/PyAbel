# PyAbel

[![Build Status](https://travis-ci.org/PyAbel/PyAbel.svg?branch=master)](https://travis-ci.org/PyAbel/PyAbel)
[![Appveyor Status](https://ci.appveyor.com/api/projects/status/github/PyAbel/PyAbel?branch=master&svg=true)](https://ci.appveyor.com/project/PyAbel/PyAbel)

<img align="right" src="https://cloud.githubusercontent.com/assets/1107796/13302896/7c7e74e2-db09-11e5-9683-a8f2c523af94.png" width="450">

PyAbel is a Python package that provides functions for the forward and inverse [Abel transforms](https://en.wikipedia.org/wiki/Abel_transform). The forward Abel transform takes a slice of a cylindrically symmetric 3D object and provides the 2D projection of that object. The inverse abel transform takes a 2D projection and reconstructs a slice of the cylindrically symmetric 3D distribution.

Inverse Abel transforms play an important role in analyzing the projections of [angle-resolved photoelectron/photoion spectra](https://en.wikipedia.org/wiki/Photofragment-ion_imaging), plasma plumes, flames, and solar occultation.

The numerical Abel transform is computationally intensive, and a basic numerical integration of the analytical equations does not reliably converge. Consequently, numerous algorithms have been developed in order to approximate the Abel transform in a reliable and efficient manner. So far, PyAbel includes the following transform methods:

1. `*` The [``BASEX``](https://github.com/PyAbel/PyAbel/wiki/BASEX-Transform) method of Dribinski and co-workers, which uses a Gaussian basis set to provide a quick, robust transform. This is one of the de facto standard methods in photoelectron/photoion spectroscopy.

2. The [``hansenlaw``](https://github.com/PyAbel/PyAbel/wiki/Hansen%E2%80%93Law-transform) recursive method of Hansen and Law, which provides an extremely fast transform with low centerline noise.

3. The [``direct``](https://github.com/PyAbel/PyAbel/wiki/Direct-transform) numerical integration of the analytical Abel transform equations, which is implemented in Cython for efficiency. In general, while the forward Abel transform is useful, the inverse Abel transform requires very fine sampling of features (lots of pixels in the image) for good convergence to the analytical result, and is included mainly for completeness and for comparison purposes. For the inverse Abel transform, other methods are generally more reliable. 

4. `*` The [``three_point``](https://github.com/PyAbel/PyAbel/wiki/Three-point-transform) method of Dasch and co-workers, which provides a fast and robust transform by exploiting the observation that underlying radial distribution is primarily determined from changes in the line-of-sight projection data in the neighborhood of each radial data point. This technique works very well in cases where the real difference between adjacent projections is much greater than the noise in the projections (i.e. where the raw data is not oversampled).

5. (Planned implementation) The ``fourierhankel`` method, which is computationally efficient, but contains significant centerline noise and is known to introduce artifacts.

6. (Planned implementation) The [``onionpeeling``](https://github.com/PyAbel/PyAbel/wiki/Onion-peeling) method.

7. (Planned implementation) The [``POP``](https://github.com/PyAbel/PyAbel/wiki/Polar-onion-peeling) (polar onion peeling) method. POP projects the image onto a basis set of Legendre polynomial-based functions, which can greatly reduce the noise in the reconstruction. However, this method only applies to images that contain features at constant radii. I.e., it works for the spherical shells seen in photoelectron/ion spectra, but not for flames.

`*` Methods marked with an asterisk require the generation of basis sets. The first time each method is run for a specific image size, a basis set must be generated, which can take several seconds or minutes. However, this basis set is saved to disk (generally to the current directory) and can be reused, making subsequent transforms very efficient. Users who are transforming numerous images using these methods will want to keep this in mind and specify the directory containing the basis sets.

## Installation

#### With pip

PyAbel requires Python 2.7 or 3.3-3.5. The latest release can be installed from PyPi with

    pip install PyAbel

#### With setuptools

If you prefer the development version from GitHub, download it here, `cd` to the PyAbel directory, and use

    python setup.py install

Or, if you wish to edit the PyAbel code without re-installing each time (advanced users):

    python setup.py develop

## Example of use

Numerous examples are located in the [`examples`](https://github.com/PyAbel/PyAbel/tree/master/examples) folder, as well as at [https://pyabel.readthedocs.org](https://pyabel.readthedocs.org).

Using PyAbel is simple:

	import abel
	original     = abel.tools.analytical.sample_image()
	forward_abel = abel.transform(original,     direction='forward', method='hansenlaw'  )['transform']
	inverse_abel = abel.transform(forward_abel, direction='inverse', method='three_point')['transform']


	# plot the original and transform
	import matplotlib.pyplot as plt
	import numpy as np
	fig, axs = plt.subplots(1,2,figsize=(7,5))
	axs[0].imshow(forward_abel,clim=(0,np.max(forward_abel)*0.3))
	axs[1].imshow(inverse_abel,clim=(0,np.max(inverse_abel)*0.3))

	axs[0].set_title('Forward Abel Transform')
	axs[1].set_title('Inverse Abel Transform')

	plt.show()

In the above, note that the `abel.transform()` function returns a Python `dict` object, where the 2D Abel transform is accessed through the `'transform'` key.

## Documentation
General information about the various Abel transforms available in PyAbel is available in the [PyAbel wiki](https://github.com/PyAbel/PyAbel/wiki). The complete documentation for all of the methods in PyAbel is hosted at [pyabel.readthedocs.org](https://pyabel.readthedocs.org/en/latest/).

## Support
If you have a question or suggestion about PyAbel, the best way to contact the PyAbel Developers Team is to open a new issue here: [https://github.com/PyAbel/PyAbel/issues](https://github.com/PyAbel/PyAbel/issues).

## Contributing

We welcome suggestions for improvement! Either [open a new Issue](https://github.com/PyAbel/PyAbel/issues) or make a [Pull Request](https://github.com/PyAbel/PyAbel/pulls). 

[CONTRIBUTING.md](https://github.com/PyAbel/PyAbel/blob/master/CONTRIBUTING.md) has more information on how to contribute, such as how to run the unit tests and how to build the documentation.


Have fun!