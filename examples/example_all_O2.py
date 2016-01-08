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
from abel.dasch import *
from abel.tools import center_image_by_slice
import matplotlib.pylab as plt
from time import time

data = np.loadtxt('data/O2-ANU1024.txt.bz2')
im = data.copy()
h,w  = np.shape(data)
print ("image 'data/O2-ANU2048.txt' shape = {:d}x{:d}".format(h,w))
h2 = h//2
w2 = w//2
mask = np.zeros(data.shape,dtype=bool)
mask[h2-40:h2,140:160] = True   # region of bright pixels for intensity normalization

# direct ------------------------------
print ("direct inverse ...")  
t0 = time()
direct = iabel_direct (data[:,w2:])   # operates on 1/2 image, oriented [0,r]
print ("                    {:.1f} sec".format(time()-t0))
direct = np.concatenate((direct[:,::-1],direct),axis=1)  
direct_speed, dsr, _ = calculate_speeds(direct)
directmax = direct[mask].max()
direct /= directmax
direct_speed /= direct_speed[dsr>50].max()

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
onion_speed, osr, _ = calculate_speeds(onion)
onionmax = onion[mask].max()  # excluding inner noise
onion /= onionmax
onion_speed /= onion_speed[osr>50].max()

# hansenlaw  ------------------------------
print ("hansen law inverse ...")
data = im.copy()
t0 = time()
hl = iabel_hansenlaw(data)   # hansenlaw takes care of image center
hl_speed, hsr, _ = calculate_speeds(hl)
print ("                   {:.1f} sec".format(time()-t0))
hlmax = hl[mask].max()       # excluding inner noise
hl /= hlmax
#hl[0:50,0:50] = 5  # tag image top-left corner
hl_speed /= hl_speed[hsr>50].max()

# basex  ------------------------------
centre = (h2-0.5,w2+0.5) 
print ("basex inverse ...")
data = im.copy()
t0 = time()
basex = BASEX (data, centre, n=h)
basex_speed, bsr, _ = calculate_speeds(basex)
print ("                   {:.1f} sec".format(time()-t0))
basexmax = basex[mask].max()*2  # fix me! fudge factor
basex  /= basexmax
basex_speed /= basex_speed[bsr>50].max()

# three_point  ------------------------------
calc_dasch = False
if calc_dasch:
    im, _ = center_image_by_slice(im, r_range=(50,700))
    print ("dasch three_point inverse ...")
    data = im.copy()
    t0 = time()
    dasch = iabel_dasch_transform (data)
    dasch_speed, ddsr, _ = calculate_speeds(dasch)
    print ("                   {:.1f} sec".format(time()-t0))
    daschmax = dasch[mask].max()
    dasch  /= daschmax
    dasch_speed /= dasch_speed[ddsr>50]


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
plt.plot (dsr,direct_speed,'k',label='direct')
plt.plot (osr,onion_speed,'r',label='onion')
plt.plot (bsr,basex_speed,'g',label='basex')
plt.plot (hsr,hl_speed,'b',label='hansenlaw')
if calc_dasch: plt.plot (ddsr,dasch_speed,'c',label='3 point')

plt.axis(ymin=-0.05,ymax=1.1,xmin=50,xmax=450)
plt.legend(loc=0,labelspacing=0.1)
plt.tight_layout()
plt.savefig('All.png',dpi=100)
plt.show()
