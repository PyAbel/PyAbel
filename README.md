# pyBASEX
A Python implementation of the BASEX algorithm by Dribinski, Ossadtchi, Mandelshtam, and Reisler, Rev. Sci. Instrum. 73 2634, (2012)

The main function of the script is to perform the inverse Abel transform. The inverse Abel transform takes a 2D projection of a cylindrically symmetric 3D image and return the 2D slice of the 3D distribution. The BASEX implementation uses Gaussian basis functions to find the transform instead of directly solving the inverse Abel transform of applying the Fourier-Hankel method. The BASEX implementation is quick, robust, and is probably the most common method used to transform velocity-map-imaging datasets.

In this code, the axis of cylindrical symmetry is in assumed to be in the vertical direction. 

Here is a quick example of how to use the program. See a more involved example in the "main()" function in the BASEX.py file.

	import BASEX
	import matplotlib.pyplot as plt
	
	filename = 'example_data/Xenon_800_nm.tif'
    raw_data = plt.imread(filename)
    
    # Specify the center in x,y (horiz,vert) format
    center = (681,491)
    
    print 'Performing the inverse Abel transform:'
    # Transform the data
    recon,speeds = BASEX.center_and_transform(raw_data,center,median_size=2,gaussian_blur=0,
                                        post_median=0,verbose=True)
										
	plt.imshow(recon,origin='lower')
	plt.show()

Have fun!