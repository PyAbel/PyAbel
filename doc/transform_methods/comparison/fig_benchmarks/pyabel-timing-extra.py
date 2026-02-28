import os
from time import time
import numpy as np

import pyabel

# The following should be used for Intel CPUs, since they work very slowly
# with subnormal floating-point numbers (see PyAbel issue #246):
try:
    import daz
    daz.set_ftz()
    daz.set_daz()
except ModuleNotFoundError:
    print('\nWarning! No daz module. Intel CPUs can show poor performance.\n')

benchmark_dir = 'working'
print('PyAbel method timings in the directory:\n\n'
      f'    {benchmark_dir}/method.dat\n')

try:
    os.mkdir(benchmark_dir)
except OSError:
    print('    directory exists, new timing will ovewrite existing file\n')


IM = np.loadtxt('../../../../examples/data/O2-ANU1024.txt.bz2')
print('Timing Daun transform of O2- sample data with reg="nonneg":\n')

t0 = time()
pyabel.Transform(IM, method='daun', symmetry_axis=0,
               transform_options=dict(reg='nonneg', verbose=True)
               ).transform
t = time() - t0
print(f'\n{t} s\n')

with open(os.path.join(benchmark_dir, f'daun(nonneg).dat'), 'w') as fp:
    fp.write(f'# {"daun(nonneg)":15s} {"iabel(ms)":20s}\n')
    fp.write(f'{np.shape(IM)[1]:5d}{t*1e3:20.5f}\n')
