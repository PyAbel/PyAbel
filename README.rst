PyAbel README
=============

.. image:: https://travis-ci.org/PyAbel/PyAbel.svg?branch=master
    :target: https://travis-ci.org/PyAbel/PyAbel
.. image:: https://ci.appveyor.com/api/projects/status/g1rj5f0g7nohcuuo
    :target: https://ci.appveyor.com/project/PyAbel/PyAbel
	
Introduction
------------

``PyAbel`` is a Python package that provides functions for the forward and inverse `Abel transforms <https://en.wikipedia.org/wiki/Abel_transform>`_. The forward Abel transform takes a slice of a cylindrically symmetric 3D object and provides the 2D projection of that object. The inverse abel transform takes a 2D projection and reconstructs a slice of the cylindrically symmetric 3D distribution.

Inverse Abel transforms play an important role in analyzing the projections of angle-resolved photoelectron/photoion spectra, plasma plumes, flames, and solar occultation.

PyAbel provides efficient implementations of several Abel transform algorithms, as well as related tools for centering images, symmetrizing images, and calculating properties such as the radial intensity distribution and the anisotropy parameters.

.. image:: https://cloud.githubusercontent.com/assets/1107796/13302896/7c7e74e2-db09-11e5-9683-a8f2c523af94.png
   :width: 430px
   :alt: PyAbel
   :align: right


Transform methods
-----------------

The numerical Abel transform is computationally intensive, and a basic numerical integration of the analytical equations does not reliably converge. Consequently, numerous algorithms have been developed in order to approximate the Abel transform in a reliable and efficient manner. So far, PyAbel includes the following transform methods:


	1. ``*`` The :doc:`BASEX <transform_methods/basex>` method of Dribinski and co-workers, which uses a Gaussian basis set to provide a quick, robust transform. This is one of the de facto standard methods in photoelectron/photoion spectroscopy. 

	2. The :doc:`Hansen–Law <transform_methods/hansenlaw>` recursive method of Hansen and Law, which provides an extremely fast transform with low centerline noise. 

	3. The :doc:`Direct <transform_methods/direct>` numerical integration of the analytical Abel transform equations, which is implemented in Cython for efficiency. In general, while the forward Abel transform is useful, the inverse Abel transform requires very fine sampling of features (lots of pixels in the image) for good convergence to the analytical result, and is included mainly for completeness and for comparison purposes. For the inverse Abel transform, other methods are generally more reliable. 

	4. ``*`` The :doc:`Three Point <transform_methods/three_point>` method of Dasch and co-workers, which provides a fast and robust transform by exploiting the observation that underlying radial distribution is primarily determined from changes in the line-of-sight projection data in the neighborhood of each radial data point. This technique works very well in cases where the real difference between adjacent projections is much greater than the noise in the projections (i.e. where the raw data is not oversampled). 

	5. (Planned implementation) The :doc:`Fourier–Hankel <transform_methods/fh>` method, which is computationally efficient, but contains significant centerline noise and is known to introduce artifacts. 

	6. (Planned implementation) The :doc:`Onion Peeling <transform_methods/onion_peeling>` method. 

	7. (Planned implementation) The :doc:`POP <transform_methods/pop>` (polar onion peeling) method. POP projects the image onto a basis set of Legendre polynomial-based functions, which can greatly reduce the noise in the reconstruction. However, this method only applies to images that contain features at constant radii. I.e., it works for the spherical shells seen in photoelectron/ion spectra, but not for flames.

	``*`` Methods marked with an asterisk require the generation of basis sets. The first time each method is run for a specific image size, a basis set must be generated, which can take several seconds or minutes. However, this basis set is saved to disk (generally to the current directory) and can be reused, making subsequent transforms very efficient. Users who are transforming numerous images using these methods will want to keep this in mind and specify the directory containing the basis sets.


Installation
------------

PyAbel requires Python 2.7 or 3.3-3.5. Numpy and Scipy are also required, and Matplotlib is required to run the examples. If you don't already have Python, we recommend an "all in one" Python package such as the `Anaconda Python Distribution <https://www.continuum.io/downloads>`_, which is available for free.

With pip
~~~~~~~~

The latest release can be installed from PyPi with ::

    pip install PyAbel

With setuptools
~~~~~~~~~~~~~~~

If you prefer the development version from GitHub, download it here, `cd` to the PyAbel directory, and use ::

    python setup.py install

Or, if you wish to edit the PyAbel source code without re-installing each time ::

    python setup.py develop


Example of use
--------------

Numerous examples are located in the `examples directory <https://github.com/PyAbel/PyAbel/tree/master/examples>`_, as well as at https://pyabel.readthedocs.org.

Using PyAbel can be simple. The following Python code imports the PyAbel package, generates a sample image, performs a forward transform using the Hansen–Law method, and then a reverse transform using the Three Point method:

.. code-block:: python

	import abel
	original     = abel.tools.analytical.sample_image()
	forward_abel = abel.transform(original,     direction='forward', method='hansenlaw'  )['transform']
	inverse_abel = abel.transform(forward_abel, direction='inverse', method='three_point')['transform']

Note: the ``abel.transform()`` function returns a Python ``dict`` object, where the 2D Abel transform is accessed through the ``'transform'`` key.

The results can then be plotted using Matplotlib:

.. code-block:: python

	import matplotlib.pyplot as plt
	import numpy as np
	
	fig, axs = plt.subplots(1, 2, figsize=(6, 4))
	
	axs[0].imshow(forward_abel, clim=(0, np.max(forward_abel)*0.6), origin='lower', extent=(-1,1,-1,1))
	axs[1].imshow(inverse_abel, clim=(0, np.max(inverse_abel)*0.4), origin='lower', extent=(-1,1,-1,1))

	axs[0].set_title('Forward Abel Transform')
	axs[1].set_title('Inverse Abel Transform')

	plt.tight_layout()
	plt.show()

Output: 

.. image:: https://cloud.githubusercontent.com/assets/1107796/13401302/d89aed7e-dec8-11e5-944f-fcafa1b75328.png
   :width: 400px
   :alt: example abel transform
   


Documentation
-------------
General information about the various Abel transforms available in PyAbel is available at the links above. The complete documentation for all of the methods in PyAbel is hosted at https://pyabel.readthedocs.org.


Support
-------
If you have a question or suggestion about PyAbel, the best way to contact the PyAbel Developers Team is to `open a new issue <https://github.com/PyAbel/PyAbel/issues>`_.


Contributing
------------

We welcome suggestions for improvement! Either open a new `Issue <https://github.com/PyAbel/PyAbel/issues>`_ or make a `Pull Request <https://github.com/PyAbel/PyAbel/pulls>`_.

`Contributing.md <https://github.com/PyAbel/PyAbel/blob/master/CONTRIBUTING.md>`_ has more information on how to contribute, such as how to run the unit tests and how to build the documentation.


License
-------
PyAble is licensed under the `MIT license <https://github.com/PyAbel/PyAbel/blob/master/LICENSE>`_, so it can be used for pretty much whatever you want! Of course, it is provided "as is" with absolutely no warrenty.


Citation
--------
First and foremost, please cite the paper(s) corresponding to the implementation of the Abel Transform that you use in your work. The references can be found at the links above.

If you find PyAbel useful in you work, it would bring us great joy if you would cite the project. [DOI coming soon!]


**Have fun!**