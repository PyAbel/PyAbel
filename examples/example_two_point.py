# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""example_two_point.py.
"""

import numpy as np
import abel
import matplotlib.pyplot as plt

# Dribinski sample image size 501x501
IM = abel.tools.analytical.sample_image(n=501)

# split into quadrants
origQ = abel.tools.symmetry.get_image_quadrants(IM)

# speed distribution of original image
orig_speed = abel.tools.vmi.angular_integration(origQ[0], origin=(0,0))
scale_factor = orig_speed[1].max()

# forward Abel projection
fIM = abel.Transform(IM, direction="forward", method="hansenlaw").transform

# split projected image into quadrants
Q = abel.tools.symmetry.get_image_quadrants(fIM)
Q0 = Q[0].copy()

# two_point inverse Abel transform
tpQ0 = abel.two_point.two_point_transform(Q0)
# speed distribution
tp_speed = abel.tools.vmi.angular_integration(tpQ0, origin=(0,0))

plt.plot(orig_speed[0], orig_speed[1]/scale_factor, linestyle='dashed', 
         label="Dribinski sample")
plt.plot(tp_speed[0], tp_speed[1]*orig_speed[1][14]/tp_speed[1][14]/scale_factor, 
         label="two_point")
plt.axis(ymin=-0.1)
plt.legend(loc=0)
plt.savefig("example_two_point.png",dpi=100)
plt.show()
