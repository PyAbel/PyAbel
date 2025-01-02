"""
Print summary information about current OS, Python, and required packages.
"""
import platform
import sys
import os

print('Platorm:', platform.platform())
print('Python:', sys.version)

try:
    import numpy
    print('NumPy:', numpy.version.full_version)
except ImportError:
    print('NumPy not found')

try:
    import scipy
    print('SciPy:', scipy.version.full_version)
except ImportError:
    print('SciPy not found')

try:
    import Cython
    print('Cython:', Cython.__version__)
except ImportError:
    print('Cython not found')
