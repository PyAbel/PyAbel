# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""example_dasch_methods.py.
"""

import numpy as np
import abel
import matplotlib.pyplot as plt

# Dribinski sample image size 501x501
n = 501
IM = abel.tools.analytical.sample_image(n)

# split into quadrants
origQ = abel.tools.symmetry.get_image_quadrants(IM)

# speed distribution of original image
orig_speed = abel.tools.vmi.angular_integration(origQ[0], origin=(0,0))
scale_factor = orig_speed[1].max()

plt.plot(orig_speed[0], orig_speed[1]/scale_factor, linestyle='dashed', 
         label="Dribinski sample")


# forward Abel projection
fIM = abel.Transform(IM, direction="forward", method="hansenlaw").transform

# split projected image into quadrants
Q = abel.tools.symmetry.get_image_quadrants(fIM)

dasch_transform = {\
"two_point": abel.dasch.two_point_transform,
"three_point": abel.dasch.three_point_transform, 
"onion_peeling": abel.dasch.onion_peeling_transform}

for method in dasch_transform.keys():
    Q0 = Q[0].copy()
# method inverse Abel transform
    AQ0 = dasch_transform[method](Q0)
# speed distribution
    speed = abel.tools.vmi.angular_integration(AQ0, origin=(0,0))

    plt.plot(speed[0], speed[1]*orig_speed[1][14]/speed[1][14]/scale_factor, 
             label=method)

plt.title("Dasch methods for Dribinski sample image $n={:d}$".format(n))
plt.axis(xmax=250, ymin=-0.1)
plt.legend(loc=0, frameon=False, labelspacing=0.1, fontsize='small')
plt.savefig("example_dasch_methods.png",dpi=100)
plt.show()
