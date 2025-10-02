import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

transforms = [
  ("basex",         '#006600', {}),
  ("basex(var)",    '#006600', {'mfc': 'w'}),
  ("daun",          '#880000', {}),
  ("daun(var)",     '#880000', {'mfc': 'w'}),
  ("daun(nonneg)",  '#880000', {'mfc': 'w'}),
  ("direct_C",      '#EE0000', {}),
  ("direct_Python", '#EE0000', {'mfc': 'w'}),
  ("hansenlaw",     '#CCAA00', {}),
  ("onion_bordas",  '#00AA00', {}),
  ("onion_peeling", '#00CCFF', {'ms': 7}),
  ("three_point",   '#0000FF', {}),
  ("two_point",     '#CC00FF', {'ms': 3}),
  ("linbasex",      '#AAAAAA', {}),
  ("rbasex",        '#AACC00', {}),
  ("rbasex(None)",  '#AACC00', {'mfc': 'w'}),
]


def plot(directory, xlim, ylim, linex):
    plt.figure(figsize=(6, 6), frameon=False)

    plt.xlabel('Image size ($n$, pixels)')
    plt.xscale('log')
    plt.xlim(xlim)

    plt.ylabel('Transform time (seconds)')
    plt.yscale('log')
    plt.ylim(ylim)
    plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))

    plt.grid(which='both', color='#EEEEEE')
    plt.grid(which='minor', linewidth=0.5)

    plt.tight_layout(pad=0.1)

    # cubic guiding line
    plt.plot(xlim, ylim[0] * (np.array(xlim) / linex)**3,
             color='#AAAAAA', ls=':')
    # its annotation (must be done after all layout for correct rotation)
    p = plt.gca().transData.transform(np.array([[1, 1**3], [2, 2**3]]))
    plt.text(linex, ylim[0], '\n     (cubic scaling)', color='#AAAAAA',
             va='center', linespacing=2, rotation_mode='anchor',
             rotation=90 - np.degrees(np.arctan2(*(p[1] - p[0]))))

    # all timings
    for meth, color, pargs in transforms:
        try:
            times = np.loadtxt(directory + '/' + meth + '.dat', unpack=True)
        except OSError:
            continue
        n = times[0]
        t = times[1] * 1e-3  # in ms
        pargs = {'color': color, 'ms': 5} | pargs
        if meth != 'daun(nonneg)':
            plt.plot(n, t, 'o-', label=meth, **pargs)
        else:  # Daun with reg='nonneg' for the O2-ANU1024 example
            plt.plot(n, t, 's', **pargs)
            plt.annotate('daun(nonneg)\n\n', (n, t), ha='right', va='center',
                         xytext=(5, 0), textcoords='offset points')

    plt.legend()

    # plt.show()


if __name__ == "__main__":
    plot('i7-9700_Linux', xlim=(5, 1e5), ylim=(7e-7, 1e4), linex=1e2)
    plt.savefig('transform_time.svg')
    plt.savefig('transform_time.pdf')
