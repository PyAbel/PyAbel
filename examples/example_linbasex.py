# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import os
import bz2

import matplotlib.pylab as plt

# This example demonstrates ``linbasex`` inverse Abel transform
# of a velocity-map image of photoelectrons from O2- photodetachment at 454 nm. 
# Measured at  The Australian National University
# J. Chem. Phys. 133, 174311 (2010) DOI: 10.1063/1.3493349

# Load image as a numpy array - numpy handles .gz, .bz2 
imagefile = bz2.BZ2File('data/O2-ANU1024.txt.bz2')
IM = np.loadtxt(imagefile)

if os.environ.get('READTHEDOCS', None) == 'True':
    IM = IM[::2,::2]
# the [::2, ::2] reduces the image size x1/2, decreasing processing memory load
# for the online readthedocs.org

# Image center should be mid-pixel and the image square, 
# `center=convolution` takes care of this

un = [0, 2]  # spherical harmonic orders
proj_angles = np.arange(0, 2*np.pi, np.pi/20) # projection angles
# adjust these parameter to 'improve' the look
smoothing = 0.9  # smoothing Gaussian 1/e width
threshold = 0.01 # exclude small amplitude Newton spheres
# no need to change these
radial_step = 1
clip = 0

# linbasex inverse Abel transform
LIM = abel.Transform(IM, method="linbasex", center="convolution",
                     center_options=dict(square=True),
                     transform_options=dict(basis_dir=None, return_Beta=True,
                                            legendre_orders=un,
                                            proj_angles=proj_angles,
                                            smoothing=smoothing,
                                            radial_step=radial_step, clip=clip,
                                            threshold=threshold)) 

# angular, and radial integration - direct from `linbasex` transform
# as class attributes
radial = LIM.radial
speed  = LIM.Beta[0]
anisotropy = LIM.Beta[1]

# normalize to max intensity peak i.e. max peak height = 1
speed /= speed[200:].max()  # exclude transform noise near centerline of image

# plots of the analysis
fig = plt.figure(figsize=(11, 5))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

# join 1/2 raw data : 1/2 inversion image
inv_IM = LIM.transform
cols = inv_IM.shape[1]
c2 = cols//2 
vmax = IM[:, :c2-100].max()
inv_IM *= vmax/inv_IM[:, c2+100:].max()
JIM = np.concatenate((IM[:, :c2], inv_IM[:, c2:]), axis=1)

# raw data
im1 = ax1.imshow(JIM, origin='upper', aspect='auto', vmin=0, vmax=vmax)
ax1.set_xlabel('column (pixels)')
ax1.set_ylabel('row (pixels)')
ax1.set_title('VMI, inverse Abel: {:d}x{:d}'.format(*inv_IM.shape),
              fontsize='small')

# Plot the 1D speed distribution and anisotropy parameter ("looks" better
# if multiplied by the intensity)
ax2.plot(radial, speed, label='speed')
ax2.plot(radial, speed*anisotropy, label=r'anisotropy $\times$ speed')
ax2.set_xlabel('radial pixel')
row, cols = IM.shape
ax2.axis(xmin=100*cols/1024, xmax=500*cols/1024, ymin=-1.5, ymax=1.8)
ax2.set_title("speed, anisotropy parameter", fontsize='small')
ax2.set_ylabel('intensity')
ax2.set_xlabel('radial coordinate (pixels)')

plt.legend(loc='best', frameon=False, labelspacing=0.1, fontsize='small')
plt.suptitle(
r'linbasex inverse Abel transform of O$_{2}{}^{-}$ electron velocity-map image',
             fontsize='larger')

# Save a image of the plot
plt.savefig("plot_example_linbasex.png", dpi=100)

# Show the plots
plt.show()
