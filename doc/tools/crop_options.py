from __future__ import print_function, division

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from copy import copy

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True,
                         figsize=(6, 1.8), frameon=False)

# origin
x0, y0 = 0.45, 0.4


def image(n, title, x1, y1, x2, y2):
    ax = axes[n]
    ax.set_title(title)
    ax.axis('off')
    ax.set_aspect('equal')
    patches = (Rectangle((0, 0), 1, 1, fc='gray'),
               Circle((x0, y0), 0.55, ec='black', fill=False, lw=2),
               Circle((x0, y0), 0.4,  ec='black', fill=False, lw=2),
               Circle((x0, y0), 0.25, ec='black', fill=False, lw=2))
    # background whole image
    clip = Rectangle((0, 0), 1, 1, visible=False)
    ax.add_patch(clip)
    for patch in patches:
        patch = copy(patch)
        ax.add_patch(patch)
        patch.set_alpha(0.2)
        patch.set_clip_path(clip)
    # foreground croped image
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(1, x2), min(1, y2)
    clip = Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1, visible=False)
    ax.add_patch(clip)
    for patch in patches:
        patch = copy(patch)
        ax.add_patch(patch)
        patch.set_clip_path(clip)
    # frame
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, ec='red', fill=False))
    # origin
    ax.plot(x0, y0, c='red', marker='+')


image(0, 'original data', 0, 0, 1, 1)

image(1, 'maintain_size',
      x0 - 0.5, y0 - 0.5,
      x0 + 0.5, y0 + 0.5)

hw, hh = min(x0, 1 - x0), min(y0, 1 - y0)
image(2, 'valid_region',
      max(0, x0 - hw), max(0, y0 - hh),
      min(1, x0 + hw), min(1, y0 + hh))

hw, hh = max(x0, 1 - x0), max(y0, 1 - y0)
image(3, 'maintain_data',
      min(0, x0 - hw), min(0, y0 - hh),
      max(0, x0 + hw), max(0, y0 + hh))

plt.subplots_adjust(left=0, bottom=0, right=1, top=0.88, wspace=0, hspace=0)

# plt.savefig('crop_options.svg')
plt.show()
