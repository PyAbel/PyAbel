.. _TransformMethods:

Transform Methods
=================

.. PyAbel includes a number of different methods for the numerical Abel transform. We recommend that these be used via the :meth:`abel.transform.Transform` class and changing the ``method`` argument.
..
.. A detailed comparison of the methods is found here: :doc:`Comparison of Abel Transform Methods <transform_methods/comparison>`
..
.. PyAbel included the following transform methods:
..
..     1. ``*`` The :doc:`basex <transform_methods/basex>` method of Dribinski and co-worker uses a Gaussian basis set to provide an efficient, robust transform. This is one of the de facto standard methods in photoelectron/photoion spectroscopy. :meth:`abel.basex.basex_transform`
..
..     2. The :doc:`hansenlaw <transform_methods/hansenlaw>` recursive method of Hansen and Law provides a fast transform with low centerline noise. :meth:`abel.hansenlaw.hansenlaw_transform`
..
..     3. The :doc:`direct <transform_methods/direct>` numerical integration of the analytical Abel transform equations is included mainly for completeness. While the forward Abel transform is useful, the inverse Abel transform requires very fine sampling of features (lots of pixels in the image) for good convergence to the analytical result. For the inverse Abel transform, other methods are generally more reliable. :meth:`abel.direct.direct_transform`
..
..     4. ``*`` The :doc:`three_point <transform_methods/three_point>` method of Dasch and co-workers provides an efficient and reliable transform. This method works well in a variety of situations. :meth:`abel.dasch.three_point_transform`
..
..     5. ``*`` The :doc:`two_point <transform_methods/two_point>` method (also published by Dasch) is a simpler approximation to the `three point` transform. :meth:`abel.dasch.two_point_transform`
..
..     6. ``*`` The :doc:`onion_peeling <transform_methods/onion_peeling>` onion-peeling deconvolution method described by Dash is one of the simpler and faster inversion methods. :meth:`abel.dasch.onion_peeling_transform`
..
..     7. The :doc:`onion_bordas <transform_methods/onion_bordas>` onion-peeling method of Bordas et al. is based on the MatLab code of Rallis and Wells *et al.* This method is reasonably slow, and is therefore not recommended for general use. :meth:`abel.onion_bordas.onion_bordas_transform`
..
..     8. ``*`` The :doc:`linbasex <transform_methods/linbasex>` 1D-spherical basis method of Gerber et al. evaluates 1D projections of velocity-map images in terms of 1D projections of spherical functions. The results produce directly the coefficients of the involved spherical functions, making the reconstruction of sliced Newton spheres unnecessary. This method makes additional assumptions about the symmetry of the data is not applicable to all situations! :meth:`abel.linbasex.linbasex_transform`
..
..     9. ``*`` The :doc:`rbasex <transform_methods/rbasex>` method is based on the pBasex method of Garcia et al. and basis functions developed by Ryazanov. Evaluates radial distributions of velocity-map images and transforms them to radial distributions of the reconstructed 3D distributions. This method makes additional assumptions about the symmetry of the data is not applicable to all situations! :meth:`abel.rbasrx.rbasex_transform`
..
..     ``*`` Methods marked with an asterisk generate basis sets and allow these basis sets to be saved to the disk to speed up future transforms.


Contents:

.. toctree::
   :maxdepth: 2
   
   transform_methods/comparison

   transform_methods/basex
   transform_methods/direct
   transform_methods/hansenlaw
   transform_methods/linbasex
   transform_methods/onion_bordas
   transform_methods/onion_peeling
   transform_methods/rbasex
   transform_methods/three_point
   transform_methods/two_point
   