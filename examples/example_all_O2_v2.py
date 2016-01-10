#!/usr/bin/env python
# -*- coding: utf-8 -*-

# rough comparison of direct, onion, hansenlaw, and basex 
# inverse Abel transform
# for the O2- photoelectron velocity-map image
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

import matplotlib.pylab as plt
from time import time

# some wrapper functions
def iabel_direct_transform(Q0):
    rows, cols = Q0.shape
    return iabel_direct(Q0, center=(rows,0))

def iabel_onion_transform(Q0):
    return iabel_onion_peeling(Q0[::-1])[::-1]

def iabel_basex_transform(Q0):
    # needs whole image
    IM = put_image_quadrants((Q0,Q0,Q0,Q0), odd_size=True)
    rows, cols = IM.shape
    center = (rows//2+rows%2, cols//2+cols%2)
    # return only Q0, top-right quadrant
    return BASEX (IM, center, n=rows, verbose=False)[:center[0], center[1]:]     


transforms = {\
#"direct"      : iabel_direct_transform, 
#"onion"       : iabel_onion_transform, 
"hansenlaw"   : iabel_hansenlaw_transform, 
#"basex"       : iabel_basex_transform, 
#"three_point" : iabel_three_point_transform
}

# O2- VMI 1024x1024 pixel image
IM = np.loadtxt('data/O2-ANU1024.txt.bz2')
# this is even size, only usable by the onion method - center on a grid
# for the others the image is recentered on an odd size image - center on pixel

IModd, offset = center_image_by_slice (IM, radial_range=(300,400))

h, w = IModd.shape
print ("centered image 'data/O2-ANU2048.txt' shape = {:d}x{:d}".format(h,w))

Q = get_image_quadrants (IModd, reorient=True)
print ("Image quadrant shape {}".format(Q[0].shape))

mask = np.zeros(Q[0].shape,dtype=bool)
mask[500:512, 358:365] = True   # region of bright pixels for intensity normalization


iabelQ = {}
speed = {}
radial = {}
for method in transforms.keys():
    Q0 = Q[0].copy()   # top-right quadrant
    print ("\n{:s} inverse ...".format(method))  

    # inverse Abel transform of quadrant
    iabelQ[method] = iabelQ0 = transforms[method](Q0)

    t0 = time()
    print ("                    {:.1f} sec".format(time()-t0))

    # as of 10Jan16 origin=(0,0) not (512,0)?
    speed[method], radial[method] = \
                      calculate_speeds(iabelQ0, origin=(0,0))

    # normalize image intensity and speed distribution
    iabelQ0 /= iabelQ0[mask].max()
    speed[method] /= speed[method][radial[method]>50].max()


# reassemble image, each quadrant a different method
im = put_image_quadrants ((iabelQ0, iabelQ0, iabelQ0, iabelQ0), odd_size=True)

plt.subplot(121)
#plt.annotate("direct",(50,50),color="yellow")
#plt.annotate("onion",(870,210),color="yellow")
#plt.annotate("3pt",(600,60),color="yellow")
plt.annotate("hansenlaw",(700,950),color="yellow")
#plt.annotate("basex",(50,950),color="yellow")
plt.annotate("quadrant intensities normalized",
             (100,1200),color='b',annotation_clip=False)   
plt.annotate("speed intensity normalized $\\rightarrow$",
             (200,1400),color='b',annotation_clip=False)   
plt.imshow(im,vmin=0)

plt.subplot(122)
#plt.plot (radial["direct"],speed["direct"],'k',label="direct")
#plt.plot (radial["onion"],speed["onion"],'r--',label="onion")
#plt.plot (radial["basex"],speed["basex"],'g',label="basex")
plt.plot (radial["hansenlaw"],speed["hansenlaw"],'b',label="hansenlaw")
#plt.plot (radial["three_point"],speed["three_point"],'m',label="three_point")

plt.axis(ymin=-0.05,ymax=1.1,xmin=50,xmax=450)
plt.legend(loc=0,labelspacing=0.1)
plt.tight_layout()
plt.savefig('All.png',dpi=100)
plt.show()
