# PyAbel
PyAbel is a Python package for performing Abel and (primarily) inverse Abel transforms. The Abel transform takes a 3D object and finds the 2D projection of that object. The more difficult problem -- the inverse Abel transform -- takes the 2D projection and finds the 3D object by assuming cylindrical symmetry in the vertical direction.

Currently, this package only uses the BASEX algorithm creared by Dribinski, Ossadtchi, Mandelshtam, and Reisler [[Rev. Sci. Instrum. 73 2634, (2002)](http://dx.doi.org/10.1063/1.1482156)], but we hope to include more algorithms for the inverse Abel transform in the future! 

The main function of this program is to perform the inverse Abel transform on a two-dimensional image. The inverse Abel transform takes a 2D projection of a cylindrically symmetric 3D image and return the central slice (2D) of the 3D distribution. The BASEX implementation uses Gaussian basis functions to find the transform instead of analytically solving the inverse Abel transform or applying the Fourier-Hankel method, as both the analytical solution and the Fourier-Hankel methods provide lower quality transforms when applied to real-world datasets (see the RSI paper). The BASEX implementation is quick, robust, and is probably the most common method used to transform velocity-map-imaging (VMI) datasets.

In this code, the axis of cylindrical symmetry is in assumed to be in the vertical direction. If this is not the case for your data, the `numpy.rot90` function may be useful.

### Installation notes

To install this module run,

    python setup.py install --user
	

### Example of use

Here is a quick example of how to use the program. See a more involved example in `examples/example_main.py`.


    from basex import BASEX
    import matplotlib.pyplot as plt
	
    filename = 'examples/data/Xenon_800_nm.tif'
    raw_data = plt.imread(filename)
    
    # Specify the center in x,y (horiz,vert) format
    center = (681, 491)

    print('Performing the inverse Abel transform:')

    # Load (or pre-calculate if needed) the basis set for a 1001x1001 image
    # using 500 basis function

    inv_ab = BASEX(n=1001, nbf=500, basis_dir='./',
            verbose=True, calc_speeds=True)

    # Calculate the inverse abel transform for the centered data
    recon, speed = inv_ab(raw_data, center, median_size=2,
                        gaussian_blur=0, post_median=0)

    #This makes the plots
    plt.imshow(recon,origin='lower')
    plt.show()

Have fun!
