#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This example compares the available inverse Abel transform methods
# currently - direct, onion, hansenlaw, and basex 
# processing the O2- photoelectron velocity-map image
#
# Note it transforms only the Q0 (top-right) quadrant

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from abel.onion import iabel_onion_peeling
from abel.hansenlaw import iabel_hansenlaw_transform
from abel.basex import BASEX
from abel.direct import iabel_direct
from abel.three_point import iabel_three_point_transform
from abel.tools.vmi import calculate_speeds, find_image_center_by_slice
from abel.tools.symmetry import get_image_quadrants, put_image_quadrants

import collections
import matplotlib.pylab as plt
from time import time

# wrapper functions --------------------------------
#   will be removed once method calls are standardized

def iabel_onion_transform(Q0):
    # onion peel requires Q1 oriented image
    return iabel_onion_peeling(Q0[:,::-1])[:,::-1]

def iabel_basex_transform(Q0):
    # basex requires a whole image
    IM = put_image_quadrants((Q0, Q0, Q0, Q0), odd_size=True)
    print ("basex processed a complete image, reconstructed from Q0 shape ",IM.shape)
    rows, cols = IM.shape
    center = (rows//2+rows%2, cols//2+cols%2)
    AIM = BASEX (IM, center, n=rows, verbose=True)
    return get_image_quadrants(AIM)[0]  # only return Q0

# inverse Abel transform methods -----------------------------
#   dictionary of method: function()

transforms = {\
  "direct"      : iabel_direct,      
  "onion"       : iabel_onion_transform, 
  "hansenlaw"   : iabel_hansenlaw_transform,
  "basex"       : iabel_basex_transform,   
  "three_point" : iabel_three_point_transform, 
}
# sort dictionary 
transforms = collections.OrderedDict(sorted(transforms.items()))
ntrans = np.size(transforms.keys())  # number of transforms


# Image:   O2- VMI 1024x1024 pixel ------------------
IM = np.loadtxt('data/O2-ANU1024.txt.bz2')
# this is even size, all methods except 'onion' require an odd-size

# recenter the image to an odd size

IModd, offset = find_image_center_by_slice (IM, radial_range=(300,400))
#np.savetxt("O2-ANU1023.txt", IModd)

h, w = IModd.shape
print ("centered image 'data/O2-ANU2048.txt' shape = {:d}x{:d}".format(h,w))

Q0, Q1, Q2, Q3 = get_image_quadrants (IModd, reorient=True)
Q0fresh = Q0.copy()    # keep clean copy
print ("quadrant shape {}".format(Q0.shape))


# Intensity mask used for intensity normalization
#   quadrant image region of bright pixels 

mask = np.zeros(Q0.shape,dtype=bool)
mask[500:512, 358:365] = True   



# process Q0 quadrant using each method --------------------

iabelQ = []  # keep inverse Abel transformed image

for q, method in enumerate(transforms.keys()):

    Q0 = Q0fresh.copy()   # top-right quadrant of O2- image

    print ("\n------- {:s} inverse ...".format(method))  
    t0 = time()

    IAQ0 = transforms[method](Q0)   # inverse Abel transform using 'method'

    print ("                    {:.1f} sec".format(time()-t0))

    iabelQ.append(IAQ0)  # store for plot

    # polar projection and speed profile
    speed, radial = calculate_speeds(IAQ0, origin=(0,0))

    # normalize image intensity and speed distribution
    IAQ0  /= IAQ0[mask].max()  
    speed /= speed[radial>50].max()

    # plots    #121 whole image,   #122 speed distributions
    plt.subplot(121) 

    # method label for each quadrant
    annot_angle = -(45+q*90)*np.pi/180  # -ve because numpy coords from top
    if q > 3: 
       annot_angle += 50*np.pi/180    # shared quadrant - move the label  
    annot_coord = ( h/2+(h*0.9)*np.cos(annot_angle)/2, 
                    w/2+(w*0.9)*np.sin(annot_angle)/2 )
    plt.annotate(method, annot_coord, color="yellow")

    # plot speed distribution
    plt.subplot(122) 
    plt.plot (radial, speed, label=method)

# reassemble image, each quadrant a different method

# for < 4 images pad using a blank quadrant
blank = np.zeros(IAQ0.shape)  
for q in range(ntrans, 4):
    iabelQ.append(blank)

# more than 4, split quadrant
if ntrans == 5:
    # split last quadrant into 2 = upper and lower triangles
    tmp_img = np.tril(np.flipud(iabelQ[-2])) +\
              np.triu(np.flipud(iabelQ[-1]))
    iabelQ[3] = np.flipud(tmp_img)
# Fix me when > 5 images
 

im = put_image_quadrants ((iabelQ[0],iabelQ[1],iabelQ[2],iabelQ[3]), 
                           odd_size=True)

plt.subplot(121)
plt.imshow(im,vmin=0,vmax=0.8)

plt.subplot(122)
plt.axis(ymin=-0.05,ymax=1.1,xmin=50,xmax=450)
plt.legend(loc=0,labelspacing=0.1)
plt.tight_layout()
plt.savefig('example_all_O2.png',dpi=100)
plt.show()
