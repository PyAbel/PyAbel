# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path

import numpy as np
import matplotlib.pyplot as plt

import abel

import scipy.misc
from scipy.ndimage.interpolation import shift
from scipy.ndimage import zoom

# This example demonstrates both Hansen and Law inverse Abel transform
# and basex for an image obtained using a velocity map imaging (VMI) 
# photoelecton spectrometer to record the photoelectron angular distribution 
# resulting  from photodetachement of O2- at 454 nm.
# This spectrum was recorded in 2010  
# ANU / The Australian National University
# J. Chem. Phys. 133, 174311 (2010) DOI: 10.1063/1.3493349
#
# Note the image zoomed to reduce calculation time

# Specify the path to the file
filename = os.path.join('data', 'O2-ANU1024.txt.bz2')

# Name the output files
base_dir, name = os.path.split(filename)
name  = name.split('.')[0]
output_image = name + '_inverse_Abel_transform_HansenLaw.png'
output_text  = name + '_speeds_HansenLaw.dat'
output_plot  = name + '_comparison_HansenLaw.png'

# Load an image file as a numpy array
print('Loading ' + filename)
im = np.loadtxt(filename)
print("scaling image to size 501 reduce the time of the basis set calculation")
im = zoom(im, 0.4892578125)
(rows, cols) = np.shape(im)
if cols%2 == 0:
    print ("Even pixel image cols={:d}, adjusting image centre\n",
           " center_image()".format(cols))
    im = abel.tools.center.center_image(im, center="slice", odd_size=True)
    # alternative
    #im = shift(im,(0.5,0.5))
    #im = im[:-1, 1::]  # drop left col, bottom row
    (rows,cols) = np.shape(im)

c2 = cols//2   # half-image width
r2 = rows//2   # half-image height
print ('image size {:d}x{:d}'.format(rows,cols))

# Hansen & Law inverse Abel transform
print('Performing Hansen and Law inverse Abel transform:')

# quad = (True ... => combine the 4 quadrants into one
reconH = abel.Transform(im, method="hansenlaw", direction="inverse", 
                        verbose=True, symmetry_axis=None).transform
rH, speedsH = abel.tools.vmi.angular_integration(reconH)

# Basex inverse Abel transform
print('Performing basex inverse Abel transform:')
reconB = abel.Transform(im, method="basex", direction="inverse", 
                        verbose=True, symmetry_axis=None).transform
rB, speedsB = abel.tools.vmi.angular_integration(reconB)

# plot the results - VMI, inverse Abel transformed image, speed profiles
# Set up some axes
fig = plt.figure(figsize=(15,4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

# Plot the raw data
im1 = ax1.imshow(im, origin='lower', aspect='auto')
fig.colorbar(im1, ax=ax1, fraction=.1, shrink=0.9, pad=0.03)
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
ax1.set_title('velocity map image: size {:d}x{:d}'.format(rows, cols))

# Plot the 2D transform
reconH2 = reconH[:,:c2]
reconB2 = reconB[:,c2:] 
recon = np.concatenate((reconH2,reconB2), axis=1)
im2 = ax2.imshow(recon, origin='lower', aspect='auto', vmin=0,
                 vmax=recon[:r2-50,:c2-50].max())
fig.colorbar(im2, ax=ax2, fraction=.1, shrink=0.9, pad=0.03)
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')
ax2.set_title('Hansen Law | Basex',x=0.4)

# Plot the 1D speed distribution - normalized
ax3.plot(rB, speedsB/speedsB[150:].max(), 'r-', label="Basex")
ax3.plot(rH, speedsH/speedsH[150:].max(), 'b-', label="Hansen Law")
ax3.axis(xmax=250, ymin=-0.1, ymax=1.5)
ax3.set_xlabel('speed (pixel)')
ax3.set_ylabel('intensity')
ax3.set_title('speed distribution')
ax3.legend(labelspacing=0.1, fontsize='small')

# Prettify the plot a little bit:
plt.subplots_adjust(left=0.06, bottom=0.17, right=0.95, top=0.89, wspace=0.35, hspace=0.37)

# Save a image of the plot
plt.savefig(output_plot, dpi=150)

# Show the plots
plt.show()
