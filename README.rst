PyAbel README
=============

.. image:: https://travis-ci.org/PyAbel/PyAbel.svg?branch=master
    :target: https://travis-ci.org/PyAbel/PyAbel
.. image:: https://ci.appveyor.com/api/projects/status/g1rj5f0g7nohcuuo
    :target: https://ci.appveyor.com/project/PyAbel/PyAbel

**Note:** This readme is best viewed as part of the `PyAbel Documentation <http://pyabel.readthedocs.io/en/latest/readme_link.html>`_.

Introduction
------------

``PyAbel`` is a Python package that provides functions for the forward and inverse `Abel transforms <https://en.wikipedia.org/wiki/Abel_transform>`_. The forward Abel transform takes a slice of a cylindrically symmetric 3D object and provides the 2D projection of that object. The inverse abel transform takes a 2D projection and reconstructs a slice of the cylindrically symmetric 3D distribution.

Inverse Abel transforms play an important role in analyzing the projections of angle-resolved photoelectron/photoion spectra, plasma plumes, flames, and solar occultation.

PyAbel provides efficient implementations of several Abel transform algorithms, as well as related tools for centering images, symmetrizing images, and calculating properties such as the radial intensity distribution and the anisotropy parameters.

.. image:: https://cloud.githubusercontent.com/assets/1107796/13302896/7c7e74e2-db09-11e5-9683-a8f2c523af94.png
   :width: 430px
   :alt: PyAbel
   :align: right


Transform Methods
-----------------

The outcome of the numerical Abel Transform depends on the exact method used. So far, PyAbel includes the following `transform methods <http://pyabel.readthedocs.io/en/latest/transform_methods.html>`_:

    1. ``basex`` - Gaussian basis set expansion of Dribinski and co-workers.

    2. ``hansenlaw`` - recursive method of Hansen and Law.

    3. ``direct`` - numerical integration of the analytical Abel transform equations.

    4. ``two_point`` - the "two point" method of Dasch and co-workers.

    5. ``three_point`` - the "three point" method of Dasch and co-workers.

    6. ``onion_peeling`` - the "onion peeling" deconvolution method of Dasch and co-workers.

    7. ``onion_bordas`` - "onion peeling" or "back projection" method of Bordas *et al.* based on the MatLab code by Rallis and Wells *et al.*

    8. ``linbasex`` - the 1D-spherical basis set expansion of Gerber *et al.*

    9. ``fh`` - Fourier–Hankel method (not yet implemented).

    10. ``pop`` - polar onion peeling method (not yet implemented).


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

Using PyAbel can be simple. The following Python code imports the PyAbel package, generates a sample image, performs a forward transform using the Hansen–Law method, and then a reverse transform using the Three Point method:

.. code-block:: python

    import abel
    original     = abel.tools.analytical.sample_image()
    forward_abel = abel.Transform(original, direction='forward', method='hansenlaw').transform
    inverse_abel = abel.Transform(forward_abel, direction='inverse', method='three_point').transform

Note: the ``abel.Transform()`` class returns a Python ``class`` object, where the 2D Abel transform is accessed through the ``.transform`` attribute.

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

.. note:: Additional examples can be viewed on the `PyAbel examples <http://pyabel.readthedocs.io/en/latest/examples.html>`_ page and even more are found in the `PyAbel/examples <https://github.com/PyAbel/PyAbel/tree/master/examples>`_ directory.


Documentation
-------------
General information about the various Abel transforms available in PyAbel is available at the links above. The complete documentation for all of the methods in PyAbel is hosted at https://pyabel.readthedocs.io.


Support
-------
If you have a question or suggestion about PyAbel, the best way to contact the PyAbel Developers Team is to `open a new issue <https://github.com/PyAbel/PyAbel/issues>`_.


Contributing
------------

We welcome suggestions for improvement! Either open a new `Issue <https://github.com/PyAbel/PyAbel/issues>`_ or make a `Pull Request <https://github.com/PyAbel/PyAbel/pulls>`_.

`CONTRIBUTING.rst <https://github.com/PyAbel/PyAbel/blob/master/CONTRIBUTING.rst>`_ has more information on how to contribute, such as how to run the unit tests and how to build the documentation.


License
-------
PyAble is licensed under the `MIT license <https://github.com/PyAbel/PyAbel/blob/master/LICENSE>`_, so it can be used for pretty much whatever you want! Of course, it is provided "as is" with absolutely no warrenty.


Citation
--------
First and foremost, please cite the paper(s) corresponding to the implementation of the Abel Transform that you use in your work. The references can be found at the links above.

If you find PyAbel useful in you work, it would bring us great joy if you would cite the project.

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.47423.svg
   :target: http://dx.doi.org/10.5281/zenodo.47423


**Have fun!**
