import numpy as np
import matplotlib.pyplot as plt

import abel.tools.transform_pairs

n = 100


def plot(profile):
    profile = 'profile' + str(profile)

    fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))

    # fig.suptitle(profile, weight='bold')  # figure title (not needed)

    eps = 1e-8  # (some profiles do not like exact 0 and 1)
    r = np.linspace(0 + eps, 1 - eps, n)
    f, a = getattr(abel.tools.transform_pairs, profile)(r)

    for i, p in enumerate([f, a]):
        ax = axs[i]
        ax.set_title(('source', 'projection')[i])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(('$r$', '$r$')[i])

        ax.plot(r, p, color='red')

        ax.set_xlim((0, 1.01))
        ax.set_ylim(bottom=0)

    plt.tight_layout()

    #plt.show()
    #plt.savefig(profile + '.svg')
