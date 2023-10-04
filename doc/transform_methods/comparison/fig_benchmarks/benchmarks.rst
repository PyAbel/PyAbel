Additional speed benchmarks
===========================

.. contents::
    :local:
    :depth: 1


All methods are benchmarked using their default parameters, with the following exceptions:

* **basex(var)** and **daun(var)** mean “variable regularization”, that is changing the regularization parameter for each transformed image.
* **direct_C** and **direct_Python** correspond to the “direct” method using its C (Cython) and Python backends respectively.
* **linbasex** and **rbasex** show whole-image (*n* × *n*) transforms, while all other methods show half-image (*n* rows, (*n* + 1)/2 columns) transforms.
* **rbasex(None)** means no output-image creation (only the transformed radial distributions).


Intel i7-9700 (Linux)
---------------------

:CPU:
    `Intel Core i7-9700 <https://ark.intel.com/content/www/us/en/ark/products/191792/intel-core-i79700-processor-12m-cache-up-to-4-70-ghz.html>`_ (8 cores, 8 threads; 3.0 GHz base, 4.7 GHz max)

:RAM:
    32 GB DDR4-2666

:OS:
    `Ubuntu 20.04 LTS <https://releases.ubuntu.com/20.04/>`_

:Libraries:
    * NumPy 1.18.1
    * SciPy 1.4.1
    * MKL 2020
    * `daz <https://github.com/chainer/daz>`_


Results
^^^^^^^

.. plot::
    :align: center

    from transform_time import plot
    plot('i7-9700_Linux', xlim=(5, 1e5), ylim=(7e-7, 1e4), linex=1e2)


.. plot::
    :align: center

    from throughput import plot
    plot('i7-9700_Linux', xlim=(5, 1e5), ylim=(4e3, 2e9), va='top')


.. plot::
    :align: center

    from basis_time import plot
    plot('i7-9700_Linux', xlim=(5, 1e5), ylim=(3e-5, 2e4), linex=1e2)


Intel i7-6700 (Linux)
---------------------

:CPU:
    `Intel Core i7-6700 <https://ark.intel.com/content/www/us/en/ark/products/88196/intel-core-i76700-processor-8m-cache-up-to-4-00-ghz.html>`_ (4 cores, 8 threads; 3.4 GHz base, 4.0 GHz max)

:RAM:
    32 GB DDR4-2133

:OS:
    `Ubuntu 19.10 <https://old-releases.ubuntu.com/releases/19.10/>`_

:Libraries:
    * NumPy 1.18.1
    * SciPy 1.4.1
    * MKL 2019 Update 5
    * `daz <https://github.com/chainer/daz>`_


Results
^^^^^^^

.. plot::
    :align: center

    from transform_time import plot
    plot('i7-6700_Linux', xlim=(5, 1e5), ylim=(7e-7, 2e4), linex=1e2)


.. plot::
    :align: center

    from throughput import plot
    plot('i7-6700_Linux', xlim=(5, 1e5), ylim=(2e3, 2e9), va='bottom')


.. plot::
    :align: center

    from basis_time import plot
    plot('i7-6700_Linux', xlim=(5, 1e5), ylim=(4e-5, 1e4), linex=1e2)


AMD Ryzen 5 5600G (Linux)
-------------------------

:CPU:
    `AMD Ryzen 5 5600G <https://www.amd.com/en/products/apu/amd-ryzen-5-5600g>`_ (6 cores, 12 threads; 3.9 GHz base, 4.4 GHz max)

:RAM:
    32 GB DDR4-3200

:OS:
    `Debian GNU/Linux 12 <https://www.debian.org/releases/bookworm/>`_

:Libraries:
    * NumPy 1.24.2
    * SciPy 1.10.1
    * OpenBLAS 0.3.21


Results
^^^^^^^

.. plot::
    :align: center

    from transform_time import plot
    plot('Ryzen5-5600G_Linux', xlim=(5, 1e5), ylim=(4e-7, 3e3), linex=9e1)


.. plot::
    :align: center

    from throughput import plot
    plot('Ryzen5-5600G_Linux', xlim=(5, 1e5), ylim=(4e3, 1e9), va='top')


.. plot::
    :align: center

    from basis_time import plot
    plot('Ryzen5-5600G_Linux', xlim=(5, 1e5), ylim=(2e-5, 2e3), linex=1e2)


AMD Ryzen 5 5600G (Windows)
---------------------------

:CPU:
    `AMD Ryzen 5 5600G <https://www.amd.com/en/products/apu/amd-ryzen-5-5600g>`_ (6 cores, 12 threads; 3.9 GHz base, 4.4 GHz max)

:RAM:
    32 GB DDR4-3200

:OS:
    `Microsoft Windows 11 <https://www.microsoft.com/en-us/windows/windows-11-specifications>`_

:Libraries:
    * NumPy 1.26.0
    * SciPy 1.11.2
    * OpenBLAS 0.3.23


Results
^^^^^^^

.. plot::
    :align: center

    from transform_time import plot
    plot('Ryzen5-5600G_Windows', xlim=(5, 1e5), ylim=(5e-7, 6e3), linex=8e1)


.. plot::
    :align: center

    from throughput import plot
    plot('Ryzen5-5600G_Windows', xlim=(5, 1e5), ylim=(8e2, 1e9), va='top')


.. plot::
    :align: center

    from basis_time import plot
    plot('Ryzen5-5600G_Windows', xlim=(5, 1e5), ylim=(2e-5, 3e3), linex=8e1)


Raspberry Pi 4B (Linux)
-----------------------

:CPU:
    `Broadcom BCM2711 <https://www.raspberrypi.com/documentation/computers/processors.html#bcm2711>`_ (4 cores; 1.5 GHz)

:RAM:
    4 GB LPDDR4-3200

:OS:
    `Raspbian GNU/Linux 10 <https://web.archive.org/web/20200518230945/https://www.raspberrypi.org/downloads/raspbian//>`_

:Libraries:
    * NumPy 1.16.2
    * SciPy 1.1.0
    * Reference BLAS 3.8.0


Results
^^^^^^^

.. plot::
    :align: center

    from transform_time import plot
    plot('RPi4B_Linux', xlim=(5, 1e4), ylim=(4e-6, 1e3), linex=3e1)


.. plot::
    :align: center

    from throughput import plot
    plot('RPi4B_Linux', xlim=(5, 1e4), ylim=(1e3, 5e7), va='bottom')


.. plot::
    :align: center

    from basis_time import plot
    plot('RPi4B_Linux', xlim=(5, 1e4), ylim=(2e-4, 4e2), linex=7e1)
