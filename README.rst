PyAbel
======

.. image:: https://travis-ci.org/PyAbel/PyAbel.svg?branch=master
    :target: https://travis-ci.org/PyAbel/PyAbel
.. image:: https://ci.appveyor.com/api/projects/status/g1rj5f0g7nohcuuo
    :target: https://ci.appveyor.com/project/PyAbel/PyAbel
	
``PyAbel`` is a Python package that provides functions for the forward and inverse `Abel transforms <https://en.wikipedia.org/wiki/Abel_transform>`_. The forward Abel transform takes a slice of a cylindrically symmetric 3D object and provides the 2D projection of that object. The inverse abel transform takes a 2D projection and reconstructs a slice of the cylindrically symmetric 3D distribution.

.. image:: https://cloud.githubusercontent.com/assets/1107796/13302896/7c7e74e2-db09-11e5-9683-a8f2c523af94.png
   :width: 430px
   :alt: PyAbel
   :align: right

Inverse Abel transforms play an important role in analyzing the projections of angle-resolved photoelectron/photoion spectra, plasma plumes, flames, and solar occultation.

The numerical Abel transform is computationally intensive, and a basic numerical integration of the analytical equations does not reliably converge. Consequently, numerous algorithms have been developed in order to approximate the Abel transform in a reliable and efficient manner. So far, PyAbel includes the following transform methods:

1. ```*``` The ``BASEX`` method of Dribinski and co-workers, which uses a Gaussian basis set to provide a quick, robust transform. This is one of the de facto standard methods in photoelectron/photoion spectroscopy. https://github.com/PyAbel/PyAbel/wiki/BASEX-Transform

2. The ``hansenlaw`` recursive method of Hansen and Law, which provides an extremely fast transform with low centerline noise. https://github.com/PyAbel/PyAbel/wiki/Hansen%E2%80%93Law-transform

3. The ``direct`` numerical integration of the analytical Abel transform equations, which is implemented in Cython for efficiency. In general, while the forward Abel transform is useful, the inverse Abel transform requires very fine sampling of features (lots of pixels in the image) for good convergence to the analytical result, and is included mainly for completeness and for comparison purposes. For the inverse Abel transform, other methods are generally more reliable. https://github.com/PyAbel/PyAbel/wiki/Direct-transform

4. ``*`` The ``three_point`` method of Dasch and co-workers, which provides a fast and robust transform by exploiting the observation that underlying radial distribution is primarily determined from changes in the line-of-sight projection data in the neighborhood of each radial data point. This technique works very well in cases where the real difference between adjacent projections is much greater than the noise in the projections (i.e. where the raw data is not oversampled). https://github.com/PyAbel/PyAbel/wiki/Three-point-transform

5. (Planned implementation) The ``fourierhankel`` method, which is computationally efficient, but contains significant centerline noise and is known to introduce artifacts. https://github.com/PyAbel/PyAbel/wiki/Fourier%E2%80%93Hankel

6. (Planned implementation) The ```onionpeeling``` method. https://github.com/PyAbel/PyAbel/wiki/Onion-peeling

7. (Planned implementation) The ``POP`` (polar onion peeling) method. POP projects the image onto a basis set of Legendre polynomial-based functions, which can greatly reduce the noise in the reconstruction. However, this method only applies to images that contain features at constant radii. I.e., it works for the spherical shells seen in photoelectron/ion spectra, but not for flames. https://github.com/PyAbel/PyAbel/wiki/Polar-onion-peeling

``*`` Methods marked with an asterisk require the generation of basis sets. The first time each method is run for a specific image size, a basis set must be generated, which can take several seconds or minutes. However, this basis set is saved to disk (generally to the current directory) and can be reused, making subsequent transforms very efficient. Users who are transforming numerous images using these methods will want to keep this in mind and specify the directory containing the basis sets.


Installation
------------

**With pip:**

PyAbel requires Python 2.7 or 3.3-3.5. The latest release can be installed from PyPi with ::

    pip install PyAbel

**With setuptools:**

If you prefer the development version from GitHub, download it here, `cd` to the PyAbel directory, and use ::

    python setup.py install

Or, if you wish to edit the PyAbel code without re-installing each time (advanced users) ::

    python setup.py develop


Example of use
--------------

Numerous examples are located in the examples directory: https://github.com/PyAbel/PyAbel/tree/master/examples, as well as at https://pyabel.readthedocs.org.

.. highlight:: python
   :linenothreshold: 5

Using PyAbel can be simple. The following Python code imports the PyAbel package, generates a sample image, performs a forward transform using the Hansen–Law method, and then a reverse transform using the Three Point method:

.. code-block:: python

	import abel
	original     = abel.tools.analytical.sample_image()
	forward_abel = abel.transform(original,     direction='forward', method='hansenlaw'  )['transform']
	inverse_abel = abel.transform(forward_abel, direction='inverse', method='three_point')['transform']


The results can then be plotted using Matplotlib:

.. code-block:: python

	# plot the original and transform:
	import matplotlib.pyplot as plt
	import numpy as np
	fig, axs = plt.subplots(1,2,figsize=(7,5))
	axs[0].imshow(forward_abel,clim=(0,np.max(forward_abel)*0.3))
	axs[1].imshow(inverse_abel,clim=(0,np.max(inverse_abel)*0.3))

	axs[0].set_title('Forward Abel Transform')
	axs[1].set_title('Inverse Abel Transform')

	plt.show()

In the above, note that the ``abel.transform()`` function returns a Python ``dict`` object, where the 2D Abel transform is accessed through the ``'transform'`` key.


Documentation
-------------
General information about the various Abel transforms available in PyAbel is available at the PyAbel Wiki: https://github.com/PyAbel/PyAbel/wiki. The complete documentation for all of the methods in PyAbel is hosted at https://pyabel.readthedocs.org.

Support
-------
If you have a question or suggestion about PyAbel, the best way to contact the PyAbel Developers Team is to open a new issue here: https://github.com/PyAbel/PyAbel/issues.

Contributing
------------

We welcome suggestions for improvement! Either open a new Issue or make a Pull Request:
https://github.com/PyAbel/PyAbel/issues
https://github.com/PyAbel/PyAbel/pulls 

https://github.com/PyAbel/PyAbel/blob/master/CONTRIBUTING.md has more information on how to contribute, such as how to run the unit tests and how to build the documentation.

License
-------
PyAble is licensed under the oh-so-liberating MIT license, so it can be used for pretty much whatever you want! However, it is provided "as is" with absolutely no warrenty.

Citation
--------
If you find PyAbel useful in you work, it would bring us great joy if you would cite the project. [DOI coming soon!]


Have fun!