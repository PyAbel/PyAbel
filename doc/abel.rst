abel package
============

    
abel.transform module
---------------------

.. automodule:: abel.transform
    :members:
    :special-members:
    :show-inheritance:

abel.basex module
-----------------

.. automodule:: abel.basex
    :members:
    :undoc-members:
    :show-inheritance:

abel.linbasex module
--------------------

.. automodule:: abel.linbasex
    :members:
    :undoc-members:
    :show-inheritance:

abel.hansenlaw module
---------------------

.. automodule:: abel.hansenlaw
    :members:
    :undoc-members:
    :show-inheritance:

abel.dasch module
-----------------

.. automodule:: abel.dasch
    :members:
    :undoc-members:
    :show-inheritance:

abel.onion_bordas module
------------------------

.. automodule:: abel.onion_bordas
    :members:
    :undoc-members:
    :show-inheritance:

abel.direct module
------------------

.. automodule:: abel.direct
    :members:
    :undoc-members:
    :show-inheritance:


Image processing tools
======================


abel.tools.analytical module
----------------------------

.. automodule:: abel.tools.analytical
    :members:
    :undoc-members:
    :show-inheritance:

abel.tools.center module
------------------------

.. automodule:: abel.tools.center
    :members:
    :undoc-members:
    :show-inheritance:

abel.tools.circularize module
------------------------------

.. automodule:: abel.tools.circularize
    :members:
    :undoc-members:
    :show-inheritance:

abel.tools.math module
----------------------

.. automodule:: abel.tools.math
    :members:
    :undoc-members:
    :show-inheritance:

abel.tools.polar module
-----------------------

.. automodule:: abel.tools.polar
    :members:
    :undoc-members:
    :show-inheritance:

abel.tools.polynomial module
-----------------------

.. automodule:: abel.tools.polynomial
    :members:
    :undoc-members:
    :show-inheritance:

abel.tools.transform_pairs module
----------------------------------

Analytical function Abel transform pairs
****************************************

    profiles 1-7, table 1 of:
    `G. C.-Y Chan and G. M. Hieftje Spectrochimica Acta B 61, 31-41 (2006)
    <http://doi:10.1016/j.sab.2005.11.009>`_

    Note: profile4 does not produce a correct Abel transform pair due
    to typographical errors in the publications

    profile 8, curve B in table 2 of:
    `Hansen and Law J. Opt. Soc. Am. A 2 510-520 (1985)
    <http://doi:10.1364/JOSAA.2.000510>`_

    Note: the transform pair functions are more conveniently accessed via
    the class::

         func = abel.tools.analytical.TransformPair(n, profile=nprofile)

      which sets the radial range r and provides attributes:
      ``.func`` (source), ``.abel`` (projection), ``.r`` (radial range),
      ``.dr`` (step), ``.label`` (the profile name)


    Parameters
    ----------
    r : floats or numpy 1D array of floats
       value or grid to evaluate the function pair: ``0 < r < 1``

    Returns
    -------
    source, projection : tuple of 1D numpy arrays of shape `r`
        source function profile (inverse Abel transform of projection),
        projection functon profile (forward Abel transform of source)

.. automodule:: abel.tools.transform_pairs
    :members:
    :undoc-members:
    :show-inheritance:

abel.tools.symmetry module
--------------------------

.. automodule:: abel.tools.symmetry
    :members:
    :undoc-members:
    :show-inheritance:

abel.tools.vmi module
---------------------

.. automodule:: abel.tools.vmi
    :members:
    :undoc-members:
    :show-inheritance:

abel.benchmark module
---------------------

.. automodule:: abel.benchmark
    :members:
    :undoc-members:
    :show-inheritance:

abel.tests module
-----------------

.. automodule:: abel.tests.run
    :members:
    :undoc-members:
    :show-inheritance:


