import numpy as np
import matplotlib.pyplot as plt
import abel 

n = 101
center = n//2
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
Left_Mat = Mat[:center+1][::-1]
Right_Mat = Mat[center:]
AnalyticAbelMat = np.exp(-1*r*r)
DaschAbelMat_Left = abel.three_point.three_point_transform(Left_Mat)[0]/dr
DaschAbelMat_Right = abel.three_point.three_point_transform(Right_Mat)[0]/dr

axes[0].plot(r, Mat, 'r', label = r'$\sqrt{\pi}e^{-r^2}$')
axes[0].legend()

axes[1].plot(r, AnalyticAbelMat, 'k', lw = 1.5, alpha = 0.5, label = r'$e^{-r^2}$')
axes[1].plot(r[:center+1], DaschAbelMat_Left[::-1], 'r--.', label = '3-pt Abel')
axes[1].plot(r[center:], DaschAbelMat_Right, 'r--.')

box = axes[1].get_position()
axes[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()