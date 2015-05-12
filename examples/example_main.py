#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from BASEX import BASEX
from BASEX.io import load_raw
import scipy.misc


# This function only executes if you run this file directly
# This is not normally done, but it is a good way to test the transform
# and also serves as a basic example of how to use the pyBASEX program.
# In practice, you will probably want to write your own script and 
# use "import BASEX" at the top of your script. 
# Then, you can call, for example:
# BASEX.center_and_transform('my_data.png',(500,500))

# Load an image file as a numpy array:

# filename = 'example_data/Xenon_800_nm.tif'
filename = 'data/Xenon_800_nm.raw'

output_image = filename[:-4] + '_Abel_transform.png'
output_text  = filename[:-4] + '_speeds.txt'
output_plot  = filename[:-4] + '_comparison.pdf'

print('Loading ' + filename)
raw_data = load_raw(filename)
# raw_data = plt.imread(filename)

# Specify the center in x,y (horiz,vert) format
center = (681,491)

print('Performing the inverse Abel transform:')

# inv_ab = BASEX(n=1001, nbf=500, basis_dir='./',
#         verbose=True, calc_speeds=True)
        
inv_ab = BASEX(n=501, nbf=250, basis_dir='./',
        verbose=True, calc_speeds=True)

# Transform the data

recon, speeds = inv_ab(raw_data, center, median_size=2,
          gaussian_blur=0, post_median=0)

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

