# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import numpy as np
import matplotlib.pyplot as plt
import abel

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
filename = os.path.join('data', 'Xenon_ATI_VMI_800_nm_649x519.tif')

# Name the output files
output_image = filename[:-4] + '_Abel_transform.png'
output_text  = filename[:-4] + '_speeds.txt'
output_plot  = filename[:-4] + '_comparison.pdf'

# Step 1: Load an image file as a numpy array
print('Loading ' + filename)
raw_data = plt.imread(filename).astype('float64')

# Step 2: Specify the center in y,x (vert,horiz) format
center = (245,340)
# or, use automatic centering
# center = 'com'
# center = 'gaussian'

# Step 3: perform the BASEX transform!
print('Performing the inverse Abel transform:')

recon = abel.Transform(raw_data, direction='inverse', method='basex',
                       center=center, transform_options={'basis_dir':'./'},
                       verbose=True).transform
                      
speeds = abel.tools.vmi.angular_integration(recon)

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
im2 = ax2.imshow(recon,origin='lower',aspect='auto',clim=(0,2000))
fig.colorbar(im2,ax=ax2,fraction=.1,shrink=0.9,pad=0.03)
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')

# Plot the 1D speed distribution

ax3.plot(*speeds)
ax3.set_xlabel('Speed (pixel)')
ax3.set_ylabel('Yield (log)')
ax3.set_yscale('log')
#ax3.set_ylim(1e2,1e5)

# Prettify the plot a little bit:
plt.subplots_adjust(left=0.06,bottom=0.17,right=0.95,top=0.89,wspace=0.35,hspace=0.37)

# Show the plots
plt.show()
