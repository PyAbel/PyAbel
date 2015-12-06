#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from abel.hansenlaw import *
from abel.basex import BASEX 
from abel.io import load_raw
from abel.tools import calculate_speeds,add_image_col,delete_image_col
import scipy.misc

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
filename = 'data/O2-ANU1024.txt.bz2'
#filename = 'data/Xenon_ATI_VMI_800_nm_649x519.tif'

# Name the output files
name = filename.split('.')[0].split('/')[1]
output_image = name + '_inverse_Abel_transform_HansenLaw.png'
output_text  = name + '_speeds_HansenLaw.dat'
output_plot  = name + '_comparison_HansenLaw.pdf'

# Load an image file as a numpy array
print('Loading ' + filename)
im = np.loadtxt(filename) if name[:2] == 'O2' else plt.imread(filename)
(n,m) = np.shape(im)
n2 = n//2   # half-image size
print ('image size {:d}x{:d}'.format(n,m))

# Hansen & Law inverse Abel transform
print('Performing Hansen and Law inverse Abel transform:')

reconH, speedsH = iabel_hansenlaw (im,verbose=True)

# Basex inverse Abel transform
# The basex basis functions require an odd pixel sized image
# add a centre pixel column to make the image width odd-size if needed
imb = add_image_col (im)
print('Performing basex inverse Abel transform:')
(nb,nb) = imb.shape
center = (nb//2,nb//2)
reconB = BASEX (imb, center, n=n, basis_dir='./',
                verbose=True, calc_speeds=False)
# remove centre column, return to original image shape
reconBx = delete_image_col (reconB) 
speedsB = calculate_speeds (reconBx)

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
reconB = reconB[:-1,:-1]  # fix temp basex (width+1,height+1) issue
reconH2 = reconH[:,:n2]
reconB2 = reconB[:,n2:] 
recon = np.concatenate((reconH2,reconB2),axis=1)
im2 = ax2.imshow(recon,origin='lower',aspect='auto',vmin=0,vmax=recon[:n2-50,:n2-50].max())
fig.colorbar(im2,ax=ax2,fraction=.1,shrink=0.9,pad=0.03)
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')
ax2.set_title('Hansen Law | Basex',x=0.42)

# Plot the 1D speed distribution - normalized to the band X(v'=2,v"=0)
ax3.plot(speedsB/speedsB[370:390].max(),'r-',label="Basex")
ax3.plot(speedsH/speedsH[370:390].max(),'b-',label="Hansen Law")
ax3.axis(xmax=n2-12,ymin=-0.1,ymax=1.5)
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
