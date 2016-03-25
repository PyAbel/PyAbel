Transform Methods
=================


The numerical Abel transform is computationally intensive, and a basic numerical integration of the analytical equations does not reliably converge. Consequently, numerous algorithms have been developed in order to approximate the Abel transform in a reliable and efficient manner. So far, PyAbel includes the following transform methods:


    1. ``*`` The :doc:`BASEX <transform_methods/basex>` method of Dribinski and co-workers, which uses a Gaussian basis set to provide a quick, robust transform. This is one of the de facto standard methods in photoelectron/photoion spectroscopy. 

    2. The :doc:`Hansen–Law <transform_methods/hansenlaw>` recursive method of Hansen and Law, which provides an extremely fast transform with low centerline noise. 

    3. The :doc:`Direct <transform_methods/direct>` numerical integration of the analytical Abel transform equations, which is implemented in Cython for efficiency. In general, while the forward Abel transform is useful, the inverse Abel transform requires very fine sampling of features (lots of pixels in the image) for good convergence to the analytical result, and is included mainly for completeness and for comparison purposes. For the inverse Abel transform, other methods are generally more reliable. 

    4. ``*`` The :doc:`Three Point <transform_methods/three_point>` method of Dasch and co-workers, which provides a fast and robust transform by exploiting the observation that underlying radial distribution is primarily determined from changes in the line-of-sight projection data in the neighborhood of each radial data point. This technique works very well in cases where the real difference between adjacent projections is much greater than the noise in the projections (i.e. where the raw data is not oversampled). 

    5. (Planned implementation) The :doc:`Fourier–Hankel <transform_methods/fh>` method, which is computationally efficient, but contains significant centerline noise and is known to introduce artifacts. 

    6. (Planned implementation) The :doc:`Onion Peeling <transform_methods/onion_peeling>` method. 

    7. (Planned implementation) The :doc:`POP <transform_methods/pop>` (polar onion peeling) method. POP projects the image onto a basis set of Legendre polynomial-based functions, which can greatly reduce the noise in the reconstruction. However, this method only applies to images that contain features at constant radii. I.e., it works for the spherical shells seen in photoelectron/ion spectra, but not for flames.

    ``*`` Methods marked with an asterisk require the generation of basis sets. The first time each method is run for a specific image size, a basis set must be generated, which can take several seconds or minutes. However, this basis set is saved to disk (generally to the current directory) and can be reused, making subsequent transforms very efficient. Users who are transforming numerous images using these methods will want to keep this in mind and specify the directory containing the basis sets.


Contents:

.. toctree::
   :maxdepth: 2
   
   transform_methods/comparison
   transform_methods/basex
   transform_methods/direct
   transform_methods/hansenlaw
   transform_methods/three_point
   transform_methods/onion_peeling
   transform_methods/pop
   transform_methods/fh
