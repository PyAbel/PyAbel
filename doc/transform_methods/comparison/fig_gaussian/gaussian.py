import numpy as np
import matplotlib.pyplot as plt
import sys
import abel
import scipy.integrate

transforms = [
  ("basex"       , abel.basex.basex_transform),
  # ("linbasex"    , abel.linbasex.linbasex_transform),
  ("direct"      , abel.direct.direct_transform),
  ("hansenlaw"   , abel.hansenlaw.hansenlaw_transform),
  ("onion_bordas", abel.onion_bordas.onion_bordas_transform),
  ("onion_peeling" , abel.dasch.onion_peeling_transform),
  ("three_point" , abel.dasch.three_point_transform),
  ("two_point"   , abel.dasch.two_point_transform)]

ntrans = len(transforms)  # number of transforms

n     = 70
error_scale = 15 # factor to scale the error

case = 'gaussian'
# case = 'circ'

if case == 'gaussian':
    r_max = n
    sigma = n*0.25
    
    ref  = abel.tools.analytical.GaussianAnalytical(n, r_max, sigma, symmetric=False)
    func = ref.func
    proj = ref.abel 
    
    r    = ref.r
    dr   = ref.dr
    
if case == 'circ':
    r_max = 1.0
    def a(n, x):
        return np.sqrt(n*n - x*x)
    
    r = np.linspace(0, r_max, n)
    func = np.ones_like(r)
    proj = 2*np.sqrt(1-r**2)
    dr   = r[1]-r[0]


## add noise:
# proj = proj + 0.4*np.abs(np.random.random(np.shape(ref.abel)))
    
fig, axs = plt.subplots(ntrans, 1, figsize=(3.37,7), sharex=True, sharey=True)

for row, (label, transFunc) in enumerate(transforms):
    axs[row].plot(r, func, label='Analytical' if row == 0 else None, lw=1)

    inverse = transFunc(np.copy(proj),dr=dr, direction='inverse')
    
    
    rms = np.mean((inverse-func)**2)**0.5
    boldlabel = '$\\bf ' + label.replace('_', '\_')  +'$'
    axs[row].plot(r, inverse, 'o', ms=1.5, label=boldlabel+', RMSE=%.2f%%'%(rms*100))
    
    axs[row].plot(r, (inverse-func)*error_scale, 'o-', ms=1, color='r', alpha=0.7, lw=1, label='Error (x%i)'%error_scale if row == 0 else None)
    
    axs[row].axhline(0, color='k', alpha=0.3, lw=1)
    
    axs[row].legend(loc='upper right', frameon=False, fontsize=8)
    
    axs[row].grid(ls='solid', alpha=0.05, color='k')

    for label in axs[row].get_yticklabels():
        label.set_fontsize(7)

    
axs[-1].set_xlabel("$r$ (pixel)")
axs[3].set_ylabel('z')

for ax, letter in zip(axs, 'abcdefghi') :
    ax.grid(ls='solid', alpha=0.05, color='k')
    # ax.xaxis.set_tick_params(direction='in')
    # ax.yaxis.set_tick_params(direction='in')
    if case == 'gaussian':
        ax.set_ylim(-0.3,1.2)
        ax.set_xlim(0,n*0.74)
    else:
        ax.set_xlim(0,1)
    
    ax.annotate(letter + ')', xy=(0.02, 0.9), xytext=(0,0), textcoords='offset points', xycoords='axes fraction', fontsize=7)


fig.subplots_adjust(left=0.08, bottom=0.07, right=0.98, top=0.99, hspace=0.1)
fig.savefig('gaussian.svg', dpi=300)
# plt.show()
