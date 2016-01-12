#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This example compares the available inverse Abel transform methods
# currently - direct, onion, hansenlaw, and basex 
# processing the O2- photoelectron velocity-map image
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abel.onion       import *
from abel.hansenlaw   import *
from abel.basex       import *
from abel.direct      import *
from abel.three_point import *
from abel.tools       import *

import collections
import matplotlib.pylab as plt
from time import time

# some wrapper functions --------------------------------
# will be removed once method calls are standardized

def iabel_onion_transform(Q0):
    # onion peel required Q1 oriented image
    return iabel_onion_peeling(Q0[::-1])[::-1]

def iabel_basex_transform(Q0):
    # basex requires a whole image
    IM = put_image_quadrants((Q0, Q0, Q0, Q0), odd_size=True)
    rows, cols = IM.shape
    center = (rows//2+rows%2, cols//2+cols%2)
    AIM = BASEX (IM, center, n=rows, verbose=False)
    return get_image_quadrants(AIM)[0]  # only return Q0

# inverse Abel transform methods -----------------------------
# dictionary of method: function()

transforms = {\
  "direct"      : iabel_direct,      
  #"onion"       : iabel_onion_transform, 
  "hansenlaw"   : iabel_hansenlaw_transform,
  "basex"       : iabel_basex_transform,   
  "three_point" : iabel_three_point_transform, 
}
# sort dictionary
transforms = collections.OrderedDict(sorted(transforms.items()))


# Image:   O2- VMI 1024x1024 pixel ------------------
IM = np.loadtxt('data/O2-ANU1024.txt.bz2')
# this is even size, all methods except 'onion' require an odd-size

# recenter the image to an odd size

IModd, offset = center_image_by_slice (IM, radial_range=(300,400))
np.savetxt("O2-ANU1023.txt", IModd)

h, w = IModd.shape
print ("centered image 'data/O2-ANU2048.txt' shape = {:d}x{:d}".format(h,w))

Q0, Q1, Q2, Q3 = get_image_quadrants (IModd, reorient=True)
print ("Image quadrant shape {}".format(Q0.shape))


# quadrant image region of bright pixels for intensity normalization

mask = np.zeros(Q0.shape,dtype=bool)
mask[500:512, 358:365] = True   



# process Q0 quadrant using each method --------------------

iabelQ = {}  # inverse Abel transformed image
speed = {}   # speed distribution
radial = {}  # radial coordinate for the speed distribution

Q0fresh = Q0.copy()
quad_tuple = ()
for q, method in enumerate(transforms.keys()):
    Q0 = Q0fresh.copy()   # top-right quadrant

    # inverse Abel transform of quadrant
    print ("\n{:s} inverse ...".format(method))  
    t0 = time()
    iabelQ[method] = iabelQ0 = transforms[method](Q0)

    print ("                    {:.1f} sec".format(time()-t0))

    # origin=(0,0), quadrant must be flipped up/down to generate correct speed profile
    speed[method], radial[method] = \
                      calculate_speeds(np.flipud(iabelQ0), origin=(0,0))

    # normalize image intensity and speed distribution
    iabelQ0       /= iabelQ0[mask].max()  
    speed[method] /= speed[method][radial[method]>50].max()
    quad_tuple += (iabelQ0,)

    # plots
    plt.subplot(121)  # combined image

    annot_angle = -(90*q+45)*np.pi/180
    annot_coord = (h/2+h*np.cos(annot_angle)/2, w/2+w*np.sin(annot_angle)/2)
    plt.annotate(method, xy=annot_coord, color="yellow")

    # speed plot
    plt.subplot(122)
    plt.plot (radial[method],speed[method],label=method)

# reassemble image, each quadrant a different method

qsize = np.shape(quad_tuple)[0]
if qsize < 4:
    blank = np.zeros(iabelQ0.shape)
    for i in range(qsize,4): quad_tuple += (blank,)

im = put_image_quadrants (quad_tuple, odd_size=True)

plt.subplot(121)
plt.imshow(im,vmin=0,vmax=1)

plt.subplot(122)
plt.axis(ymin=-0.05,ymax=1.1,xmin=50,xmax=450)
plt.legend(loc=0,labelspacing=0.1)
plt.tight_layout()
plt.savefig('example_all_O2.png',dpi=100)
plt.show()
