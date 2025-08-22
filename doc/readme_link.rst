..
    (Excluding/changing GitHub/PyPI-oriented parts for RTD documentation)

.. include:: ../README.rst
    :end-before: begin-github-only1

.. include:: ../README.rst
    :start-after: end-github-only1
    :end-before: begin-github-only2

.. image:: overview.*
    :align: center

.. include:: ../README.rst
    :start-after: end-github-only2
    :end-before: begin-github-only3

.. |methods| replace:: :doc:`transform methods <transform_methods>`

.. include:: ../README.rst
    :start-after: end-github-only3
    :end-before: begin-github-only4

.. plot::

    import abel
    original = abel.tools.analytical.SampleImage(name='Gerber').func
    forward_abel = abel.Transform(original, direction='forward',
                                  method='hansenlaw').transform
    inverse_abel = abel.Transform(forward_abel, direction='inverse',
                                  method='three_point').transform

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].imshow(forward_abel, clim=(0, None), cmap='ocean_r')
    axs[1].imshow(inverse_abel, clim=(0, None), cmap='ocean_r')

    axs[0].set_title('Forward Abel transform')
    axs[1].set_title('Inverse Abel transform')

    plt.tight_layout()

.. |examples| replace:: in :doc:`PyAbel examples <examples>`

.. include:: ../README.rst
    :start-after: end-github-only4
    :end-before: begin-github-only5

.. |CONTRIBUTING| replace:: :doc:`Contributing to PyAbel <contributing_link>`

.. include:: ../README.rst
    :start-after: end-github-only5
    :end-before: begin-github-only6

.. include:: ../README.rst
    :start-after: end-github-only6
