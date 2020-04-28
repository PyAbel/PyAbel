# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

benchmark_dir = 'benchmarks_i7-9700_Linux_5.4.0-26-generic'
'benchmarks_CPU_Linux_3.10.0-862.3.3.el7.x86_64'
'benchmarks_i7-6700_Linux_4.18.0-12-generic-MRvar'
# benchmark_dir = 'benchmarks_Xenon_Linux_3.10.0-862.3.3.el7.x86_64-MRvar'

transforms = [
  ("basex"        , '#880000', dict()),
  ("basex(var)"   , '#880000', dict(mfc='w')),
  ("rbasex"       , '#AACC00', dict()),
  ("rbasex(None)" , '#AACC00', dict(mfc='w')),
  ("direct_C"     , '#EE0000', dict()),
  ("direct_Python", '#EE0000', dict(mfc='w')),
  ("hansenlaw"    , '#CCAA00', dict()),
  ("onion_bordas" , '#00AA00', dict()),
  ("onion_peeling", '#00CCFF', dict()),
  ("three_point"  , '#0000FF', dict()),
  ("two_point"    , '#CC00FF', dict()),
  ("linbasex"     , '#BBBBBB', dict()),
]

fig, axs = plt.subplots(2,1,figsize=(3.37,7))

for num, (meth, color, pargs) in enumerate(transforms):
    print(meth)
    fn = benchmark_dir + '/' + meth + '.dat'
    try:
        times = np.loadtxt(fn, unpack=True)
    except:
        print('  cannot open')
        continue
    n = times[0]


    pargs.update(ms=3, lw=0.75, color=color, zorder=-num)

    axs[0].plot(n, times[1]*1e-3, 'o-', label=meth, **pargs)
    if times.shape[0] > 2:
        axs[0].plot(n, times[2]*1e-3, 's--', **pargs)

    axs[1].plot(n, n**2/times[1] * 1e3, 'o-', label=meth, **pargs)
        
    
axs[0].set_ylabel('Transform time (seconds)')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylim(1e-6, 9e3)
plt.setp(axs[0].get_xticklabels(), fontsize=8)
plt.setp(axs[0].get_yticklabels(), fontsize=8)

axs[0].plot(1054, 0.6,     'H', ms=5.5, mec='k', mfc='yellow', color='k', label='Rallis $et\;al.$')
axs[1].plot(1054, 1e6/0.6, 'H', ms=5.5, mec='k', mfc='yellow', color='k', label='Rallis $et\;al.$')
axs[0].plot(1000, 0.4,     'D', ms=4, mec='k', mfc='yellow', color='k', label='Harrison $et\;al.$')
axs[1].plot(1000, 1e6/0.4, 'D', ms=4, mec='k', mfc='yellow', color='k', label='Harrison $et\;al.$')


ns = np.logspace(np.log10(1e2), np.log10(1e5)) 
axs[0].plot(ns, 5e-13*ns**3, alpha=0.5, color='k', ls=':', lw=0.75, zorder=-99)

axs[1].set_xscale('log')
axs[1].set_yscale('log')
plt.setp(axs[1].get_xticklabels(), fontsize=8)
plt.setp(axs[1].get_yticklabels(), fontsize=8)

axs[1].set_xlabel('Image size ($n$, pixels)')
axs[1].set_ylabel('Throughput (pixels per second)')
axs[1].plot(1000, 30e6, '*', ms=8, mec='k', mfc='yellow')
axs[1].annotate(u'High-definition video\n(1000×1000 pixels, 30 fps)', xy=(1050, 34e6), xytext=(25,50),
                textcoords='offset points', color='k', fontsize=9, ha='center',
                arrowprops=dict(color='k', arrowstyle='->'))


leg = axs[1].legend(fontsize=8, frameon=True, loc=4, numpoints=1, labelspacing=0.1, ncol=2, columnspacing=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

axs[1].set_ylim(7e2, 5e9)

for ax in axs:
    ax.set_xlim(10, 1e5)

for ax in axs:
    ax.grid(alpha=0.1, color='k', ls='solid')
    ax.grid(alpha=0.05, color='k', ls='solid', which='minor')
    

def place_letter(letter, ax, color='k', offset=(0,0)):
    ax.annotate(letter, xy=(0.02, 0.97), xytext=offset, xycoords='axes fraction', textcoords='offset points', color=color,
                ha='left', va='top', weight='bold')
    
for ax, letter in zip(axs.ravel(), 'ab'):
    place_letter(letter+')', ax)
    
fig.tight_layout(pad=0.1)
    

plt.savefig('benchmarks.png', dpi=300)
plt.show()
