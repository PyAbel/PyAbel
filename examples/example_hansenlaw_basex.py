#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path

import numpy as np
import matplotlib.pyplot as plt

from abel.hansenlaw import *
from abel.basex import BASEX 
from abel.io import load_raw
import scipy.misc
from scipy.ndimage.interpolation import shift

# This example demonstrates both Hansen and Law inverse Abel transform
# and basex for an image obtained using a velocity map imaging (VMI) 
# photoelecton spectrometer to record the photoelectron angular distribution 
# resulting  from photodetachement of O2- at 454 nm.
# This spectrum was recorded in 2010  
# ANU / The Australian National University
# J. Chem. Phys. 133, 174311 (2010) DOI: 10.1063/1.3493349

# Before you start, centre of the NxN numpy array should be the centre
#  of image symmetry
#   +----+----+
#   |    |    |
#   +----o----+
#   |    |    |
#   + ---+----+

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
(rows,cols) = np.shape(im)
if cols%2 != 1:
    print ("Even pixel image cols={:d}, adjusting image centre\n",
           " shift(im,(-0.5,-0.5)")
    imx = shift(im,(-0.5,-0.5))
    im  = imx[:-1,1:]  # drop left column, bottom row
    (rows,cols) = np.shape(im)
c2 = cols//2   # half-image width
r2 = rows//2   # half-image height
print ('image size {:d}x{:d}'.format(rows,cols))

# Hansen & Law inverse Abel transform
print('Performing Hansen and Law inverse Abel transform:')

# quad = (True ... => combine the 4 quadrants into one
reconH, speedsH = iabel_hansenlaw (im,calc_speeds=True,verbose=True)

# Basex inverse Abel transform
print('Performing basex inverse Abel transform:')
center = (r2,c2)
reconB, speedsB = BASEX (im, center, n=rows, basis_dir='./',
                             verbose=True, calc_speeds=True)

# plot the results - VMI, inverse Abel transformed image, speed profiles
# Set up some axes
fig = plt.figure(figsize=(15,4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

# Plot the raw data
im1 = ax1.imshow(im,origin='lower',aspect='auto')
fig.colorbar(im1,ax=ax1,fraction=.1,shrink=0.9,pad=0.03)
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
ax1.set_title('velocity map image')

# Plot the 2D transform
reconH2 = reconH[:,:c2]
reconB2 = reconB[:,c2:] 
recon = np.concatenate((reconH2,reconB2),axis=1)
im2 = ax2.imshow(recon,origin='lower',aspect='auto',vmin=0,vmax=recon[:r2-50,:c2-50].max())
fig.colorbar(im2,ax=ax2,fraction=.1,shrink=0.9,pad=0.03)
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')
ax2.set_title('Hansen Law | Basex',x=0.4)

# Plot the 1D speed distribution - normalized
ax3.plot(speedsB/speedsB[250:280].max(),'r-',label="Basex")
ax3.plot(speedsH/speedsH[250:280].max(),'b-',label="Hansen Law")
ax3.axis(xmax=c2-12,ymin=-0.1,ymax=1.5)
ax3.set_xlabel('Speed (pixel)')
ax3.set_ylabel('Intensity')
ax3.set_title('Speed distribution')
ax3.legend(labelspacing=0.1,fontsize='small')

# Prettify the plot a little bit:
plt.subplots_adjust(left=0.06,bottom=0.17,right=0.95,top=0.89,wspace=0.35,hspace=0.37)

# Save a image of the plot
plt.savefig(output_plot,dpi=150)

# Show the plots
plt.show()
