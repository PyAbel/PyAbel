import numpy as np
import matplotlib.pyplot as plt
import abel 

n = 101
r_max = 4
r = np.linspace(-1*r_max, r_max, n)
dr = r[1]-r[0]

fig, axes = plt.subplots(ncols = 2)
fig.set_size_inches(8,4)

axes[0].set_xlabel("Lateral position, x")
axes[0].set_ylabel("F(x)")
axes[0].set_title("Original LOS signal")
axes[1].set_xlabel("Radial position, r")
axes[1].set_ylabel("f(r)")
axes[1].set_title("Inverted radial signal")

Mat = np.sqrt(np.pi)*np.exp(-1*r*r)
AnalyticAbelMat = np.exp(-1*r*r)
DaschAbelMat = abel.three_point.three_point_transform(Mat)[0]

axes[0].plot(r, Mat, 'r', label = r'$\sqrt{\pi}e^{-r^2}$')
axes[0].legend()

axes[1].plot(r, AnalyticAbelMat, 'k', lw = 1.5, alpha = 0.5, label = r'$e^{-r^2}$')
axes[1].plot(r, DaschAbelMat, 'r--', lw = 1.5, label = '3-pt Abel')

box = axes[1].get_position()
axes[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()