#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from abel.basex import BASEX
from abel.io import load_raw
import scipy.misc

# This example demonstrates a BASEX transform of an image obtained using a 
# velocity map imaging (VMI) photoelecton spectrometer to record the 
# photoelectron angualar distribution resulting from above threshold ionization (ATI)
# in xenon gas using a ~40 femtosecond, 800 nm laser pulse. 
# This spectrum was recorded in 2012 in the Kapteyn-Murnane research group at 
# JILA / The University of Colorado at Boulder
# by Dan Hickstein and co-workers (contact DanHickstein@gmail.com)
# http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.073004
#
# Before you start your own transform, identify the central pixel of the image.
# It's nice to use a program like ImageJ for this. 
# http://imagej.nih.gov/ij/

# Specify the path to the file
filename = 'data/Xenon_ATI_VMI_800_nm_649x519.tif'

# Name the output files
output_image = filename[:-4] + '_Abel_transform.png'
output_text  = filename[:-4] + '_speeds.txt'
output_plot  = filename[:-4] + '_comparison.pdf'

# Step 1: Load an image file as a numpy array
print('Loading ' + filename)
raw_data = plt.imread(filename)

# Step 2: Specify the center in x,y (horiz,vert) format
center = (340,245)

# Step 3: perform the BASEX transform!
print('Performing the inverse Abel transform:')

recon, speeds = BASEX(raw_data, center, n=501, basis_dir='./',
                      verbose=True, calc_speeds=True)

# # save the transform in 16-bits (recommended, but requires pyPNG)
# save16bitPNG('Xenon_800_transformed.png',recon)

# save the transform in 8-bit format:
scipy.misc.imsave(output_image,recon)

# save the speed distribution
with open(output_text,'w') as outfile:
    outfile.write('Pixel\tIntensity\n')
    for pixel,intensity in enumerate(speeds):
        outfile.write('%i\t%f\n'%(pixel,intensity))

## Finally, plot the original image, the BASEX transform, and the radial distribution

# Set up some axes
fig = plt.figure(figsize=(15,4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

# Plot the raw data
im1 = ax1.imshow(raw_data,origin='lower',aspect='auto')
fig.colorbar(im1,ax=ax1,fraction=.1,shrink=0.9,pad=0.03)
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')

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

# Show the plots
plt.show()

# Hey, that was fun!
