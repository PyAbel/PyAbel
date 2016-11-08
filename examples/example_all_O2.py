# -*- coding: utf-8 -*-

# This example compares the available inverse Abel transform methods
# currently - direct, hansenlaw, and basex 
# processing the O2- photoelectron velocity-map image
#
# Note it transforms only the Q0 (top-right) quadrant
# using the fundamental transform code

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel

import collections
import matplotlib.pylab as plt
from time import time

# inverse Abel transform methods -----------------------------
#   dictionary of method: function()

transforms = {
  "basex": abel.basex.basex_transform,   
  "linbasex": abel.linbasex.linbasex_transform,
  "direct": abel.direct.direct_transform,      
  "hansenlaw": abel.hansenlaw.hansenlaw_transform,
  "onion_bordas": abel.onion_bordas.onion_bordas_transform,
  "onion_dasch": abel.dasch.onion_peeling_transform, 
  "three_point": abel.dasch.three_point_transform,
  "two_point" : abel.dasch.two_point_transform,
}
# sort dictionary 
transforms = collections.OrderedDict(sorted(transforms.items()))
ntrans = np.size(transforms.keys())  # number of transforms


# Image:   O2- VMI 1024x1024 pixel ------------------
IM = np.loadtxt('data/O2-ANU1024.txt.bz2')

# recenter the image to mid-pixel (odd image width)
IModd = abel.tools.center.center_image(IM, center="slice", odd_size=True)

h, w = IModd.shape
print("centered image 'data/O2-ANU2048.txt' shape = {:d}x{:d}".format(h, w))

# split image into quadrants
Q = abel.tools.symmetry.get_image_quadrants(IModd, reorient=True)

Q0 = Q[0]
Q0fresh = Q0.copy()    # keep clean copy
print ("quadrant shape {}".format(Q0.shape))

# Intensity mask used for intensity normalization
#   quadrant image region of bright pixels 
mask = np.zeros(Q0.shape, dtype=bool)
mask[500:512, 358:365] = True   

# process Q0 quadrant using each method --------------------

iabelQ = []  # keep inverse Abel transformed image
sp = []  # speed distributions
meth = []  # methods

for q, method in enumerate(transforms.keys()):

    Q0 = Q0fresh.copy()   # top-right quadrant of O2- image

    print ("\n------- {:s} inverse ...".format(method))  
    t0 = time()

    # inverse Abel transform using 'method'
    IAQ0 = transforms[method](Q0, direction="inverse", dr=0.1) 
    print ("                    {:.1f} sec".format(time()-t0))


    # polar projection and speed profile
    radial, speed = abel.tools.vmi.angular_integration(IAQ0, origin=(0, 0),
                                                       dr=0.1)

    # normalize image intensity and speed distribution
    IAQ0 /= IAQ0[mask].max()  
    speed /= speed[radial > 50].max()

    # keep data for plots
    iabelQ.append(IAQ0)  
    sp.append((radial, speed))
    meth.append(method)

# reassemble image, each quadrant a different method

# plot inverse Abel transformed image slices, and respective speed distributions
ax0 = plt.subplot2grid((1, 2), (0, 0))
ax1 = plt.subplot2grid((1, 2), (0, 1))

def ann_plt (quad, subquad, txt):
    # -ve because numpy coords from top
    annot_angle = -(30+30*subquad+quad*90)*np.pi/180
    annot_coord = (h/2+(h*0.8)*np.cos(annot_angle)/2,
                   w/2+(w*0.8)*np.sin(annot_angle)/2)
    ax0.annotate(txt, annot_coord, color="yellow", horizontalalignment='left')

# for < 4 images pad using a blank quadrant
r, c = Q0.shape
Q = np.zeros((4, r, c))

indx = np.triu_indices(iabelQ[0].shape[0])
iq = 0
for q in range(4):
    Q[q] = iabelQ[iq].copy()
    ann_plt(q, 0, meth[iq])
    ax1.plot(*(sp[iq]), label=meth[iq], alpha=0.3)
    iq += 1
    if iq < len(transforms):
        Q[q][indx] = np.triu(iabelQ[iq])[indx] 
        ann_plt(q, 1, meth[iq])
        ax1.plot(*(sp[iq]), label=meth[iq], alpha=0.3)
    iq += 1

# reassemble image from transformed (part-)quadrants
im = abel.tools.symmetry.put_image_quadrants((Q[0], Q[1], Q[2], Q[3]), 
                                              original_image_shape=IModd.shape)

ax0.axis('off')
ax0.set_title("inverse Abel transforms")
ax0.imshow(im, vmin=0, vmax=0.8)

ax1.set_title("speed distribution")
ax1.axis(ymin=-0.05, ymax=1.1, xmin=50, xmax=450)
ax1.legend(loc=0, labelspacing=0.1, fontsize=10, frameon=False)
plt.tight_layout()

# save a copy of the plot
plt.savefig('example_all_O2.png', dpi=100)
plt.show()
