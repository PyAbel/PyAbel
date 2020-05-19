# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

benchmark_dir = 'working'

transforms = [
    ("basex",         '#880000', {}),
    ("basex(var)",    '#880000', {'mfc': 'w'}),
    ("direct_C",      '#EE0000', {}),
    ("direct_Python", '#EE0000', {'mfc': 'w'}),
    ("hansenlaw",     '#CCAA00', {}),
    ("onion_bordas",  '#00AA00', {}),
    ("onion_peeling", '#00CCFF', {}),
    ("three_point",   '#0000FF', {}),
    ("two_point",     '#CC00FF', {}),
    ("linbasex",      '#AAAAAA', {}),
    ("rbasex",        '#AACC00', {}),
    ("rbasex(None)",  '#AACC00', {'mfc': 'w'}),
]

fig, axs = plt.subplots(2, 1, figsize=(5, 8))

for num, (meth, color, pargs) in enumerate(transforms):
    print(meth)
    fn = benchmark_dir + '/' + meth + '.dat'
    try:
        times = np.loadtxt(fn, unpack=True)
    except OSError:
        print('  cannot open')
        continue
    n = times[0]

    pargs.update(ms=3, lw=0.75, color=color, zorder=-num)

    # transform time
    axs[0].plot(n, times[1]*1e-3, 'o-', label=meth, **pargs)

    # basis time
    if times.shape[0] > 2:
        axs[0].plot(n, times[2]*1e-3, 's--', **pargs)

    # throughput
    axs[1].plot(n, n**2/times[1] * 1e3, 'o-', label=meth, **pargs)

axs[0].set_ylabel('Transform time (seconds)')
axs[0].set_xscale('log')
axs[0].set_yscale('log')

ns = np.logspace(np.log10(1e2), np.log10(1e5))
axs[0].plot(ns, 5e-13*ns**3, alpha=0.5, color='k', ls=':', lw=0.75, zorder=-99)

axs[1].set_xscale('log')
axs[1].set_yscale('log')

axs[1].set_xlabel('Image size ($n$, pixels)')
axs[1].set_ylabel('Throughput (pixels per second)')
axs[1].plot(1000, 30e6, '*', ms=8, mec='k', mfc='yellow')
axs[1].annotate(u'High-definition video\n(1000×1000 pixels, 30 fps)',
                xy=(1050, 34e6), xytext=(25, 50),
                textcoords='offset points', color='k', fontsize=9, ha='center',
                arrowprops=dict(color='k', arrowstyle='->'))

leg = axs[0].legend(fontsize=8, frameon=True, loc='lower right', numpoints=1,
                    labelspacing=0.1, ncol=2, columnspacing=0.5)

for line, text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

for ax in axs:
    ax.set_xlim(5, 1e5)

for ax in axs:
    ax.grid(alpha=0.1, color='k', ls='solid')
    ax.grid(alpha=0.05, color='k', ls='solid', which='minor')

fig.tight_layout(pad=0.1)

# plt.savefig('benchmarks.png', dpi=300)
plt.show()
