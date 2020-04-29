# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

directory = 'benchmarks_i7-9700_Linux_5.4.0-26-generic'

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

    times = np.loadtxt(directory + '/' + meth + '.dat', unpack=True)
    n = times[0]
    t = times[1] * 1e-3  # in ms
    plt.plot(n, n**2 / t, 'o-', label=meth, ms=ms, **pargs)

plt.xlabel('Image size ($n$, pixels)')
plt.xscale('log')
plt.xlim(5, 1e5)

plt.ylabel('Throughput (pixels per second)')
plt.yscale('log')
plt.ylim(2e3, 2e9)

plt.grid(which='both', color='#EEEEEE')
plt.grid(which='minor', linewidth=0.5)

plt.legend(ncol=3)

plt.tight_layout(pad=0.1)

plt.savefig('throughput.svg')
# plt.show()
