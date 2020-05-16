# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

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


def plot(directory, xlim, ylim, linex):
    plt.figure(figsize=(6, 6), frameon=False)

    plt.xlabel('Image size ($n$, pixels)')
    plt.xscale('log')
    plt.xlim(xlim)

    plt.ylabel('Basis-set generation time (seconds)')
    plt.yscale('log')
    plt.ylim(ylim)
    plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))

    plt.grid(which='both', color='#EEEEEE')
    plt.grid(which='minor', linewidth=0.5)

    plt.tight_layout(pad=0.1)

    # quadratic guiding line
    plt.plot(xlim, ylim[0] * (np.array(xlim) / linex)**2,
             color='#AAAAAA', ls=':')
    # its annotation (must be done after all layout for correct rotation)
    p = plt.gca().transData.transform(np.array([[1, 1**2], [2, 2**2]]))
    plt.text(linex, ylim[0], '\n      (quadratic scaling)', color='#AAAAAA',
             va='center', linespacing=2, rotation_mode='anchor',
             rotation=90 - np.degrees(np.arctan2(*(p[1] - p[0]))))

    # all timings
    for meth, color, pargs in transforms:
        try:
            times = np.loadtxt(directory + '/' + meth + '.dat', unpack=True)
        except OSError:
            continue
        if times.shape[0] < 3:
            continue
        n = times[0]
        t = times[2] * 1e-3  # in ms
        plt.plot(n, t, 'o-', label=meth, ms=5, color=color)

    plt.legend()

    # plt.show()


if __name__ == "__main__":
    plot('i7-9700_Linux', xlim=(5, 1e5), ylim=(3e-5, 2e4), linex=1e2)
    plt.savefig('basis_time.svg')
