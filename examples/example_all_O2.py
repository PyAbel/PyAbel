#!/usr/bin/env python
# rough comparison of direct, onion, hansenlaw, and basex inverse Abel transform
# for the O2- data
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abel.onion import *
from abel.hansenlaw import *
from abel.basex import *
from abel.direct import *
import matplotlib.pylab as plt
from time import time

data = np.loadtxt('data/O2-ANU1024.txt.bz2')
im = data.copy()
h,w  = np.shape(data)
h2 = h//2
w2 = w//2
mask = np.zeros(data.shape,dtype=bool)
mask[h2-20:h2,140:160] = True   # region of bright pixels for intensity normalization

# direct ------------------------------
print ("direct inverse ...")  
t0 = time()
direct = iabel_direct (data[:,w2:])   # operates on 1/2 image, oriented [0,r]
print ("                    {:.1f} sec".format(time()-t0))
direct = np.concatenate((direct[:,::-1],direct),axis=1)  
direct_speed = calculate_speeds(direct)
directmax = direct[mask].max()
direct /= directmax

# onion  ------------------------------
print ("onion inverse ...")
data = im.copy()
t0 = time()
Q = get_image_quadrants(data,reorient=True)
AO = []
# flip quadrants to match pre 24 Dec 15 orientation definition 
for q in Q:
    AO.append(iabel_onion_peeling(q[:,::-1])[:,::-1]) #flip, flip-back
# only quadrant inversion available??
# reassemble
onion = put_image_quadrants((AO[0],AO[1],AO[2],AO[3]),odd_size=False)
print ("                   {:.1f} sec".format(time()-t0))
onion_speed = calculate_speeds(onion)
onionmax = onion[mask].max()  # excluding inner noise
onion /= onionmax

# hansenlaw  ------------------------------
print ("hansen law inverse ...")
data = im.copy()
t0 = time()
hl,hl_speed = iabel_hansenlaw(data,calc_speeds=True)
print ("                   {:.1f} sec".format(time()-t0))
hlmax = hl[mask].max()       # excluding inner noise
hl /= hlmax
hl[0:50,0:50] = 5  # tag image top-left corner

# basex  ------------------------------
centre = (h2,w2) 
print ("basex inverse ...")
data = im.copy()
t0 = time()
basex,basex_speed = BASEX (data,centre,n=h,calc_speeds=True,verbose=False)
print ("                   {:.1f} sec".format(time()-t0))
basexmax = basex[mask].max()
basex  /= basexmax

# reassemble image, each quadrant a different method
im = data.copy()
direct = direct[:h2,:w2]
im[:h2,:w2] = direct[:,:]   # Q1  
im[:h2,w2:] = onion[:h2,w2:]    # Q0
im[h2:,:w2] = basex[h2:-1,:w2]    # Q2
im[h2:,w2:] = hl[h2:,w2:]       # Q3

plt.subplot(121)
plt.annotate("direct",(50,50),color="yellow")
plt.annotate("onion",(800,50),color="yellow")
plt.annotate("hansenlaw",(700,950),color="yellow")
plt.annotate("basex",(50,950),color="yellow")
plt.annotate("quadrant intensities normalized",
             (100,1200),color='b',annotation_clip=False)   
plt.annotate("speed intensity normalized $\\rightarrow$",
             (200,1400),color='b',annotation_clip=False)   
plt.imshow(im,vmin=0,vmax=hlmax/2)

plt.subplot(122)
plt.plot (direct_speed/direct_speed[:-50].max(),'k',label='direct')
plt.plot (onion_speed/onion_speed[:-50].max(),'r',label='onion')
plt.plot (basex_speed/basex_speed[:-50].max(),'g',label='basex')
plt.plot (hl_speed/hl_speed[:-50].max(),'b',label='hansenlaw')

plt.axis(ymin=-0.05,ymax=1.1,xmin=50,xmax=450)
plt.legend(loc=0,labelspacing=0.1)
plt.tight_layout()
plt.savefig('All.png',dpi=100)
plt.show()
