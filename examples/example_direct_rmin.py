import matplotlib.pyplot as plt
from abel.direct import direct_transform_new
from abel.tools.analytical import TransformPair

ref = TransformPair(n=100, profile=4)

# take the data for r > r_min only
r_min = 0.4
n_min = int(ref.n * r_min)
cut_r = ref.r[n_min:]
cut_abel = ref.abel[n_min:]

# inverse Abel transform for r > r_min
cut_res = direct_transform_new(cut_abel, r=cut_r, direction='inverse')

plt.figure(figsize=(6, 4))
plt.title('Inverse Abel transform of incomplete data')
plt.xlabel('Radius')
plt.ylabel('Intensity')
plt.plot(ref.r, ref.abel, 'C0:')
plt.plot(cut_r, cut_abel, 'C0', label='Data')
plt.plot(ref.r, ref.func, 'C1:')
plt.plot(cut_r, cut_res, 'C1', label='Transform')
plt.xlim(0, 1)
plt.ylim(0, None)
plt.legend()
plt.tight_layout()
plt.show()
