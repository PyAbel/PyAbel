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

.. plot::

    import abel
    original     = abel.tools.analytical.SampleImage().func
    forward_abel = abel.Transform(original, direction='forward',
                                  method='hansenlaw').transform
    inverse_abel = abel.Transform(forward_abel, direction='inverse',
                                  method='three_point').transform

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].imshow(forward_abel, cmap='ocean_r')
    axs[1].imshow(inverse_abel, cmap='ocean_r')

    axs[0].set_title('Forward Abel transform')
    axs[1].set_title('Inverse Abel transform')

    plt.tight_layout()

.. note:: Additional examples can be viewed in :doc:`PyAbel examples <examples>` and even more are found in the `PyAbel/examples <https://github.com/PyAbel/PyAbel/tree/master/examples>`__ directory.

.. include:: ../README.rst
    :start-after: end-github-only3
    :end-before: begin-github-only4

.. include:: ../README.rst
    :start-after: end-github-only4
