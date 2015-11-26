#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

######################################################################
#
# example alternative inverse Abel transformation using the method of
#    Hansen and Law J. Opt. Soc. Am A2, 510-520 (1985) 
#
# Stephen Gibson - Australian National University, Australia
# Jason Gascooke - Flinders University, Australia
#
#  O-ANU2048.txt.bz2 is a bzipped 2048x2048 ascii velocity-map
#  image of O- photodetachment at 812.5 nm
######################################################################

from abel.HansenLaw import *  
from matplotlib.pylab import plt

# ascii image data: ANU O- photodetachment velocity-map image 2048x2048 pixel
im = np.loadtxt("./data/O-ANU2048.txt.bz2")
(n,m) = np.shape(im)

# Hansen and Law inverse Abel transformation
reconHL, speedHL = HansenLaw (im,mask=0b1111,verbose=True)

np.savetxt("speedHL.dat",speedHL)

# pretty plot
fig = plt.figure(figsize=(10,4))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
title = "Hansen & Law"
ax1.set_title (title,position=(0.4,1))
# vmax from image excluding centre line
ax1.imshow(reconHL,aspect=None,vmin=0,vmax=reconHL[:n//2-100,:n//2-100].max())
ax1.set_ylim(ax1.get_ylim()[::-1])
ax2.plot (speedHL/max(speedHL[500:]),'b-',label='Hansen&Law')
ax2.axis(ymin=-0.5,ymax=1.5)
ax2.set_title ("ANU O- photodetachment at 812.5 nm - speed")
plt.legend()
plt.savefig("example_HansenLaw.png")
plt.show()
