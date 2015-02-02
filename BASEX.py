import numpy as np
from numpy.linalg import inv
from numpy import dot,transpose
import matplotlib.pyplot as plt
import Image, time, scipy.ndimage, scipy.misc

######################################################################
# PyBASEX - A Python BASEX implementation
# Dan Hickstein - University of Colorado Boulder
# danhickstein@gmail.com
#
# This is adapted from the BASEX Matlab code provided by the Reisler group.
#
# Please cite: "The Gaussian basis-set expansion Abel transform method" 
# V. Dribinski, A. Ossadtchi, V. A. Mandelshtam, and H. Reisler, 
# Review of Scientific Instruments 73, 2634 (2002).
#
# Version 1.2 - 2015-02-01
#   Added documentation 
# Version 1.1 - 2014-10-09
#   Adding a "center_and_transform" function to make things easier
# Versions 1.0 - 2012 
#   First port to Python
#
#
# To-Do list:
#
#   I took all of the linear algebra straight from the Matlab program. It's
#   a little hard to compare with the Rev. Sci. Instrum. paper. It would be 
#   nice to clean this up so that it's easier to follow along with the paper.
#
#   Currently, this program just uses the 1000x1000 basis set generated using 
#   the Matlab implementation of BASEX. It would be good to port the basis set 
#   generating functions as well. This would give people the flexibility to use 
#   different sized basis sets. For example, some image may need higher resolution
#   than 1000x1000, or, transforming larger quantities of low-resolution images
#   may be faster with a 100x100 basis set. 
# 
########################################################################


def main():
    # This function only executes if you run this file directly
    # This is not normally done, but it is a good way to test the transform
    # and also serves as a basic example of how to use the pyBASEX program.
    # In practice, you will probably want to write your own script and 
    # use "import BASEX" at the top of your script. 
    # Then, you can call, for example:
    # BASEX.center_and_transform('my_data.png',(500,500))
    
    # Load an image file as a numpy array:
    
    filename = 'example_data/Xenon_800_nm.tif'
    output_image = filename[:-4] + '_Abel_transform.png'
    output_text  = filename[:-4] + '_speeds.txt'
    output_plot  = filename[:-4] + '_comparison.pdf'
    
    print 'Loading ' + filename
    raw_data = plt.imread(filename)
    
    # Specify the center in x,y (horiz,vert) format
    center = (681,491)
    
    print 'Performing the inverse Abel transform:'
    # Transform the data
    recon,speeds = center_and_transform(raw_data,center,median_size=2,gaussian_blur=0,
                                        post_median=0,verbose=True)
    
    # # save the transform in 16-bits (requires pyPNG):
    # save16bitPNG('Xenon_800_transformed.png',recon)
    
    # save the transfrom in 8-bits:
    scipy.misc.imsave(output_image,recon)
    
    #save the speed distribution
    with open(output_text,'w') as outfile:
        outfile.write('Pixel\tIntensity\n')
        for pixel,intensity in enumerate(speeds):
            outfile.write('%i\t%f\n'%(pixel,intensity))
            
    # Set up some axes
    fig = plt.figure(figsize=(15,4))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    
    # Plot the raw data
    im1 = ax1.imshow(raw_data,origin='lower',aspect='auto')
    fig.colorbar(im1,ax=ax1,fraction=.1,shrink=0.9,pad=0.03)
    ax2.set_xlabel('x (pixels)')
    ax2.set_ylabel('y (pixels)')

    # Plot the 2D transform
    im2 = ax2.imshow(recon,origin='lower',aspect='auto')
    fig.colorbar(im2,ax=ax2,fraction=.1,shrink=0.9,pad=0.03)
    ax2.set_xlabel('x (pixels)')
    ax2.set_ylabel('y (pixels)')
    
    # Plot the 1D speed distribution
    ax3.plot(speeds)
    ax3.set_xlabel('Speed (pixel)')
    ax3.set_ylabel('Yield (log)')
    ax3.set_yscale('log')
    
    # Prettify the plot a little bit:
    plt.subplots_adjust(left=0.06,bottom=0.17,right=0.95,top=0.89,wspace=0.35,hspace=0.37)
    
    # Save a image of the plot
    plt.savefig(output_plot,dpi=150)
    
    plt.show()
    
    

def center_and_transform(data,center,median_size=0,gaussian_blur=0,post_median=0,
                         verbose=False,calc_speeds=True,symmetrize=False):
    # This is the main function that center the image, blurs the image (if desired)
    # and completes the BASEX transform. 
    #
    # Inputs:
    # data - a NxN numpy array where N is larger than 1000. 
    #        If N is smaller than 1000, zeros will we added to the edges on the image.
    # center - the center of the image in (x,y) format
    # median_size - size (in pixels) of the median filter that will be applied to the image before 
    #               the transform. This is crucial for emiminating hot pixels and other 
    #               high-frequency sensor noise that would interfere with the transform
    # gaussian_blur - the size (in pixels) of the gaussian blur applied before the BASEX tranform.
    #                 this is another way to blur the image before the transform. 
    #                 It is normally not used, but if you are looking at very broad features
    #                 in very noisy data and wich to apply an aggressive (large radius) blur
    #                 (i.e., a blur in excess of a few pixels) then the gaussian blur will 
    #                 provide better results than the median filter. 
    # post_median - this is the size (in pixels) of the median blur applied AFTER the BASEX transform
    #               it is not normally used, but it can be a good way to get rid of high-frequency 
    #               artifacts in the transformed image. For example, it can reduce centerline noise.
    # verbose - Set to True to see more output for debugging
    # calc_speeds - determines if the speed distribution should be calculated
    
    image = center_image(data,center=center)
    
    if symmetrize==True:
        image = apply_symmetry(image)
    
    if median_size>0:  
        image = scipy.ndimage.median_filter(image,size=median_size)
    if gaussian_blur>0: image = scipy.ndimage.gaussian_filter(image,sigma=gaussian_blur)
    
    #Do the actual transform
    recon,speeds = BASEX(image,calc_speeds=calc_speeds,verbose=verbose)
    
    if post_median > 0:
        recon = scipy.ndimage.median_filter(recon,size=post_median)
    return recon,speeds
    
    
def center_image(data,center):
    # This centers the image at the given center and makes it 1000 by 1000
    # We cannot use larger images without making new coefficients, which I don't know how to do
    H,W = np.shape(data)
    im=np.zeros((2000,2000))
    im[(1000-center[1]):(1000-center[1]+H),(1000-center[0]):(1000-center[0]+W)]=data
    im = im[499:1500,499:1500]
    return im


def BASEX(rawdata,verbose=False,calc_speeds=True):
    #This is the core funciton that does the actual transform
    # INPUTS:
    #  rawdata: a 1000x1000 numpy array of the raw image. 
    #       Must use this size, since this is what we have generated the coefficients for.
    #       If your image is larger you must crop or downsample.
    #       If smaller, pad with zeros outside. Just use the "center_image" function.
    #  verbose: Set to True to see more output for debugging
    #  calc_speeds: determines if the speed distribution should be calculated
    #
    # RETURNS:
    #  IM: The abel-transformed image, 1000x1000. 
    #      This is a slice of the 3D distribution
    #  speeds: a array of length=500 of the 1D distribution, integrated over all angles
    
     
    ### Loading basis sets ### 
    if verbose==True: print 'Loading basis sets...           ',; t1 = time.time()
    
    try:
        left,right,M,MT,Mc,McT = np.load('basisALL.npy0')
    except:
        print 'Basis sets not saved in numpy format. Loading text files.'
    
        try:
            M = np.loadtxt('basis1000pr_1.txt'); 
            Mc = np.loadtxt('basis1000_1.txt');
        except:
            M = np.loadtxt('./additional_files/basis1000pr_1.txt'); 
            Mc = np.loadtxt('./additional_files/basis1000_1.txt');
        
        np.save('basis1000.npy',(M,Mc))
        McT = transpose(Mc)
        MT  = transpose(M)
            
        left = dot( inv(dot(McT,Mc)) , McT) #Just making things easier to read
        q=1;
        NBF=np.shape(M)[1] # number of basis functions  
        E = np.identity(NBF)*q  # Creating diagonal matrix for regularization. (?)
        right = dot(M,inv(dot(MT,M)+E))
        np.save('basisALL.npy',(left,right,M,MT,Mc,McT))
        
    if verbose==True: print '%.2f seconds'%(time.time()-t1)
       
    # ### Reconstructing image  - This is where the magic happens###
    if verbose==True: print('Reconstructing image...         '),; t1 = time.time()
    
    Ci = dot(dot(left,rawdata),right)
    # P = dot(dot(Mc,Ci),MT) # This calculates the projection, which should recreate the original image
    IM = dot(dot(Mc,Ci),McT)
    
    if verbose==True: print '%.2f seconds'%(time.time()-t1)
    
    if calc_speeds==True:
        if verbose==True: print('Generating speed distribution...'),; t1 = time.time()
        nx,ny = np.shape(IM)
        xi = np.linspace(-100,100,nx)
        yi = np.linspace(-100,100,ny)
        X,Y = np.meshgrid(xi,yi)
    
        polarIM, ri, thetai = reproject_image_into_polar(IM)
    
        speeds = np.sum(polarIM,axis=1)
        speeds = speeds[:500] #Clip off the corners
        if verbose==True: print '%.2f seconds'%(time.time()-t1)
    else:
        speeds = np.zeros(500)
    
    return IM,speeds


# This section is to get the speed distribution. 
# The original matlab version used an analytical formula to get the speed distribution directly
# from the basis coefficients. But, the C version of BASEX uses a numerical method similar to 
# the one implemented here. The difference between the two methods is negligable. 

# I got these next two functions from a stackoverflow page and slightly modified them.
# http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
# It is possible that there is a faster way to get the speed distribution. 
# If you figure it out, pease let me know! (danhickstein@gmail.com)
def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)
    
    nr = r.max()
    nt = ny//2

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr)
    theta_i = np.linspace(theta.min(), theta.max(), nt)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = polar2cart(r_grid, theta_grid)
    X += origin[0] # We need to shift the origin
    Y += origin[1] # back to the lower-left corner...
    xi, yi = X.flatten(), Y.flatten() 
    coords = np.vstack((xi,yi)) # (map_coordinates requires a 2xn array)

    zi = scipy.ndimage.map_coordinates(data, coords)
    output = zi.reshape((nr,nt))
    return output, r_i, theta_i
    

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return r, theta
    
def polar2cart(r, theta):
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y
    

# The following functions are just conveinent functions for loading and saving images.
# Often you can just use plt.imread('my_file.png') to load a file.
# plt.imread also works for 16-bit tiff files.
def load_raw(filename,start=2,end=1440746,height=1038,width=1388):
    # This loads one of the raw VMI images from Vrakking's "VMI_Acquire" software
    # It ignores the first two values (which are just the dimensions of the image,
    # and not actual data) and cuts off about 10 values at the end.
    # I don't know why the files are not quite the right size, but this seems to work.
    
    # Load raw data
    A = np.fromfile(filename, dtype='int32', sep="")
    # Reshape into a numpy array
    return A[start:end].reshape([height, width])
    

def save16bitPNG(filename,data):
    # It's not easy to save 16-bit images in Python. Here is a way to save a 16-bit PNG
    # Again, this is thanks to stackoverflow: #http://stackoverflow.com/questions/25696615/can-i-save-a-numpy-array-as-a-16-bit-image-using-normal-enthought-python
    # This requires pyPNG
    
    import png
    with open(filename, 'wb') as f:
        writer = png.Writer(width=data.shape[1], height=data.shape[0], bitdepth=16, greyscale=True)
        data_list = data.tolist()
        writer.write(f, data_list)

if __name__ == '__main__':
    main() #


        
