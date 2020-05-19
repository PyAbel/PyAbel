import numpy as np
import matplotlib.pyplot as plt
import abel
import bz2

transforms = [
  ("basex",         abel.basex.basex_transform,               '#880000'),
  ("direct",        abel.direct.direct_transform,             '#EE0000'),
  ("hansenlaw",     abel.hansenlaw.hansenlaw_transform,       '#CCAA00'),
  ("onion_bordas",  abel.onion_bordas.onion_bordas_transform, '#00AA00'),
  ("onion_peeling", abel.dasch.onion_peeling_transform,       '#00CCFF'),
  ("three_point",   abel.dasch.three_point_transform,         '#0000FF'),
  ("two_point",     abel.dasch.two_point_transform,           '#CC00FF'),
  ("linbasex",      abel.linbasex.linbasex_transform,         '#AAAAAA'),
  ("rbasex",        abel.rbasex.rbasex_transform,             '#AACC00'),
]

infile = bz2.BZ2File('../../../../examples/data/O2-ANU1024.txt.bz2')
IM = np.loadtxt(infile)

IModd = abel.tools.center.center_image(IM, origin="convolution",
                                       odd_size=True, square=True)

Q = abel.tools.symmetry.get_image_quadrants(IModd, reorient=True)
Q0 = Q[1]

h, w = np.shape(IM)

fig, axs = plt.subplots(3, 3, figsize=(7, 6.5), sharex=True, sharey=True)
fig1, axs1 = plt.subplots(3, 1, figsize=(6, 7))

for num, (ax, (label, transFunc, color), letter) in enumerate(zip(axs.ravel(),
                                                              transforms,
                                                              'abcdefghij')):
    print(label)

    if label == 'linbasex':
        targs = dict(proj_angles=np.arange(0, np.pi, np.pi/10))
    else:
        targs = dict()

    if label == 'basex':
        targs = dict(reg=200)

    if label == 'rbasex':
        trans = transFunc(Q0, direction="inverse", origin='ll')[0]

    else:
        trans = transFunc(Q0, direction="inverse", **targs)

    if label == 'linbasex':  # bugfix smoothing=0 transform offset by 1 pixel
        trans[:, 1:] = trans[:, :-1]

    r, inten = abel.tools.vmi.angular_integration(trans[::-1],
                                                  origin=(0, 0),
                                                  dr=0.1)

    inten /= 1e6

    im = ax.imshow(trans[::-1], cmap='gist_heat_r', origin='lower',
                   aspect='equal', vmin=0, vmax=5)

    ax.set_title(letter + ') ' + label,
                 x=0.05, y=0.93, ha='left', va='top',
                 weight='bold', color='k')

    pargs = dict(lw=1, color=color, zorder=-num)

    axs1[0].plot(r, inten,              **pargs)
    axs1[1].plot(r, inten, label=label, **pargs)
    axs1[2].plot(r, inten,              **pargs)


axc = fig.add_axes([0.93, 0.06, 0.01, 0.93])
cbar = plt.colorbar(im, orientation="vertical", cax=axc, label='Intensity')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')


for ax in axs.ravel():
    ax.set_xlim(0, 450)
    ax.set_ylim(0, 450)
    major = range(0, 500, 100)
    minor = range(50, 550, 100)
    ax.set_xticks(major)
    ax.set_xticklabels(major, fontdict={'fontsize': 6})
    ax.set_xticks(minor, minor=True)
    ax.set_yticks(major)
    ax.set_yticklabels(major, fontdict={'fontsize': 6},
                       rotation='vertical', verticalalignment='center')
    ax.set_yticks(minor, minor=True)

fig.subplots_adjust(left=0.05, bottom=0.06, right=0.92, top=0.99,
                    wspace=0.04, hspace=0.04)

for ax in axs[-1]:
    ax.set_xlabel('$r$ (pixels)')

for ax in axs[:, 0]:
    ax.set_ylabel('$z$ (pixels)')


for ax in axs1:
    ax.grid(color='k', alpha=0.1)

axs1[0].set_xlim(0, 512)
axs1[0].set_xticks(np.arange(0, 514, 20), minor=True)

axs1[1].set_xlim(355, 385)
axs1[1].set_xticks(np.arange(355, 385), minor=True)

axs1[2].set_xlim(80, 160)
axs1[2].set_xticks(np.arange(80, 160, 10), minor=True)
axs1[2].set_ylim(-0.004, 0.02)


def place_letter(letter, ax, color='k', offset=(0, 0)):
    ax.annotate(letter, xy=(0.02, 0.97), xytext=offset,
                xycoords='axes fraction', textcoords='offset points',
                color=color, ha='left', va='top', weight='bold')


for ax, letter in zip(axs1.ravel(), 'abc'):
    place_letter(letter+')', ax, color='k')

axs1[-1].set_xlabel('$r$ (pixels)')
fig1.tight_layout()
axs1[1].legend(fontsize=8)

fig.savefig('experiment.svg')
fig1.savefig('integration.svg')
# plt.show()
