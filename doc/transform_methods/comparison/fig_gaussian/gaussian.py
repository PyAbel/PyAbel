import numpy as np
import matplotlib.pyplot as plt
import abel

transforms = [
  ("basex",         abel.basex.basex_transform),
  ("direct",        abel.direct.direct_transform),
  ("hansenlaw",     abel.hansenlaw.hansenlaw_transform),
  ("onion_bordas",  abel.onion_bordas.onion_bordas_transform),
  ("onion_peeling", abel.dasch.onion_peeling_transform),
  ("three_point",   abel.dasch.three_point_transform),
  ("two_point",     abel.dasch.two_point_transform)]

ntrans = len(transforms)  # number of transforms

n = 70
error_scale = 15  # factor to scale the error

case = 'gaussian'
# case = 'circ'

if case == 'gaussian':
    r_max = n
    sigma = n*0.25

    ref = abel.tools.analytical.GaussianAnalytical(n, r_max, sigma,
                                                   symmetric=False)
    func = ref.func
    proj = ref.abel

    r = ref.r
    dr = ref.dr

if case == 'circ':
    r_max = 1.0

    def a(n, x):
        return np.sqrt(n*n - x*x)

    r = np.linspace(0, r_max, n)
    func = np.ones_like(r)
    proj = 2*np.sqrt(1-r**2)
    dr = r[1] - r[0]

fig, axs = plt.subplots(ntrans, 1, figsize=(5, 9), sharex=True, sharey=True)

for row, (label, transFunc) in enumerate(transforms):
    axs[row].plot(r, func, label='Analytical' if row == 0 else None, lw=1)

    inverse = transFunc(np.copy(proj), dr=dr, direction='inverse')

    rms = np.mean((inverse - func)**2)**0.5
    boldlabel = '$\\bf ' + label.replace('_', '\_') + '$'
    axs[row].plot(r, inverse, 'o', ms=1.5,
                  label=boldlabel + ', RMSE=%.2f%%' % (rms*100))

    axs[row].plot(r, (inverse-func)*error_scale, 'o-', ms=1, color='r',
                  alpha=0.7, lw=1,
                  label='Error (x%i)' % error_scale if row == 0 else None)

    axs[row].axhline(0, color='k', alpha=0.3, lw=1)

    axs[row].legend(loc='upper right', frameon=False)

    axs[row].grid(ls='solid', alpha=0.05, color='k')


axs[-1].set_xlabel("$r$ (pixels)")
axs[3].set_ylabel('$z$')

for ax, letter in zip(axs, 'abcdefghi'):
    ax.grid(ls='solid', alpha=0.05, color='k')
    if case == 'gaussian':
        ax.set_ylim(-0.25, 1.25)
        ax.set_xlim(0, n*0.74)
    else:
        ax.set_xlim(0, 1)

    ax.annotate(letter + ')', xy=(0.02, 0.88), xytext=(0, 0),
                textcoords='offset points', xycoords='axes fraction',
                weight='bold')


fig.tight_layout(pad=0)
fig.savefig('gaussian.svg')
# plt.show()
