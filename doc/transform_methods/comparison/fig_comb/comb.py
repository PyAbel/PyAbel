import abel
from abel.tools.analytical import PiecewisePolynomial
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np

hw = 1     # peak half-width - Or is this the full-width?
step = 10  # center-to-center distance between peaks
n = 6      # number of peaks

rmax = int(n * step)


def peak(i):
    c = i * step
    if i:
        return [(c - hw, c, [1,  1], c, hw),
                (c, c + hw, [1, -1], c, hw)]
    else:
        return [(c, c + hw, [1, -1], c, hw)]


comb = PiecewisePolynomial(rmax + 1, rmax,
                           chain(*[peak(i) for i in range(1, n)]),
                           symmetric=False)

np.random.seed(4)
func = comb.abel + np.random.random(comb.abel.size)*1.2

transforms = [
  ("basex",          abel.basex.basex_transform,               '#006600'),
  ("basex (reg=10)", abel.basex.basex_transform,               '#006600'),
  ("daun (reg=5)",   abel.daun.daun_transform,                 '#880000'),
  ("daun (nonneg)",  abel.daun.daun_transform,                 '#880000'),
  ("direct",         abel.direct.direct_transform,             '#EE0000'),
  ("hansenlaw",      abel.hansenlaw.hansenlaw_transform,       '#CCAA00'),
  ("onion_bordas",   abel.onion_bordas.onion_bordas_transform, '#00AA00'),
  ("onion_peeling",  abel.dasch.onion_peeling_transform,       '#00CCFF'),
  ("three_point",    abel.dasch.three_point_transform,         '#0000FF'),
  ("two_point",      abel.dasch.two_point_transform,           '#CC00FF'),
]

ntrans = len(transforms)  # number of transforms

fig, axs = plt.subplots(ntrans, 1, figsize=(5, 10),
                        sharex=True, sharey=True)


def mysum(x,  dx=1, axis=1):
    # return np.trapz(x)
    return np.sum(x)


for num, (ax, (label, transFunc, color)) in enumerate(zip(axs.ravel(),
                                                          transforms)):
    print(label)
    if 'reg' in label:
        reg = label[label.rfind('=') + 1:-1]
        targs = dict(reg=float(reg))
    elif 'nonneg' in label:
        targs = dict(reg='nonneg')
    else:
        targs = dict()

    ax.plot(comb.r, comb.func, lw=1, color='#888888')

    recd = transFunc(func, **targs)
    ax.plot(comb.r, recd, lw=1, label=label, color=color, ms=2, marker='o')


def place_letter(letter, ax, color='k', offset=(0, 0)):
    ax.annotate(letter, xy=(0.02, 0.97), xytext=offset,
                xycoords='axes fraction', textcoords='offset points',
                color=color, ha='left', va='top', weight='bold')


for ax, letter in zip(axs.ravel(), 'abcdefghij'):
    ax.legend(loc='upper right', frameon=False, borderaxespad=0)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.2, 1.4)
    ax.set_yticks([0, 0.5, 1])

    ax.grid(alpha=0.2)

    place_letter(letter+')', ax)

axs[-1].set_xlabel('$r$ (pixels)')
axs[4].set_ylabel('$z$')


fig.tight_layout(pad=0)
plt.savefig('comb.svg')
plt.savefig('comb.pdf')
# plt.show()
