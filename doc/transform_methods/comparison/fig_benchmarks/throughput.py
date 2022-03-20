# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

transforms = [
  ("basex",         '#006600', {}),
  ("basex(var)",    '#006600', {'mfc': 'w'}),
  ("daun",          '#880000', {}),
  ("daun(var)",     '#880000', {'mfc': 'w'}),
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


def plot(directory, xlim, ylim, va):
    plt.figure(figsize=(6, 6), frameon=False)

    # all timings
    for meth, color, pargs in transforms:
        pargs.update(color=color)
        if meth == 'two_point':
            ms = 3
        elif meth == 'three_point':
            ms = 5
        elif meth == 'onion_peeling':
            ms = 7
        else:
            ms = 5

        try:
            times = np.loadtxt(directory + '/' + meth + '.dat', unpack=True)
        except OSError:
            continue
        n = times[0]
        t = times[1] * 1e-3  # in ms
        plt.plot(n, n**2 / t, 'o-', label=meth, ms=ms, **pargs)

        # add an empty entry to end column 1 for more logical grouping
        if meth == 'daun(var)':
            plt.plot(np.NaN, np.NaN, 'o-', color='none', label=' ')

    plt.xlabel('Image size ($n$, pixels)')
    plt.xscale('log')
    plt.xlim(xlim)

    plt.ylabel('Throughput (pixels per second)')
    plt.yscale('log')
    plt.ylim(ylim)

    plt.grid(which='both', color='#EEEEEE')
    plt.grid(which='minor', linewidth=0.5)

    # HD video: 1 Mp * 30 Hz
    x = 1000
    y = 30 * x**2
    plt.plot(x, y, '+k')
    plt.annotate('HD video', (x, y), ha='center', va=va,
                 xytext=(0, -4 if va == 'top' else 2),
                 textcoords='offset points')

    plt.legend(ncol=3)

    plt.tight_layout(pad=0.1)

    plt.show()


if __name__ == "__main__":
    plot('i7-9700_Linux', xlim=(5, 1e5), ylim=(3e3, 2e9), va='top')
    plt.savefig('throughput.svg')
    plt.savefig('throughput.pdf')
