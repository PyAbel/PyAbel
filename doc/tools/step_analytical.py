# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from abel.tools.analytical import StepAnalytical

fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))

rmax = 7
r1 = 3
r2 = 5

for i in [0, 1]:
    ax = axs[i]
    ax.set_title(('source', 'projection')[i])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(('$r$', '$r$')[i])
    ax.set_xticks([0, r1, r2, rmax])
    ax.set_xticklabels(['0', 'r1', 'r2', 'r_max'])

    if i == 0:  # source
        ax.plot([0, r1, r1, r2, r2, rmax],
                [0, 0,  1,  1,  0,  0], color='red')

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', 'A0'])
    else:  # projection
        step = StepAnalytical(rmax * 20 + 1, rmax, r1, r2, symmetric=False)
        ax.plot(step.r, step.abel, color='red')

        ax.set_yticks([0])

    ax.set_xlim((0, rmax))
    ax.set_ylim(bottom=0)

plt.tight_layout()

#plt.savefig(profile + '.svg')
#plt.show()
