from pathlib import Path
import re
import sys
from setuptools import setup, Extension


# Set __version__ to the current package version
exec(Path('abel/_version.py').read_text('utf-8'))


# Use README as the project description on PyPI
long_description = Path('README.rst').read_text('utf-8')

# but remove CI badges
long_description = re.sub(
    r'^.+?https://(github\.com/.+/pytest.yml|ci\.appveyor\.com/).*?\n', '',
    long_description,
    flags=re.MULTILINE, count=4)  # limit to top 2 pairs of image + target

# and change all GH and RTD links to this specific PyAbel version
long_description = long_description.\
    replace('https://github.com/PyAbel/PyAbel/tree/master/',
            f'https://github.com/PyAbel/PyAbel/tree/v{__version__}/').\
    replace('https://pyabel.readthedocs.io/en/latest/',
            f'https://pyabel.readthedocs.io/en/v{__version__}/')


# Try to build Cython extensions
try:
    import numpy
    from Cython.Compiler import Options
    Options.annotate = False  # don't create HTML
    if sys.platform == 'win32':  # for MSVC
        libraries = []
        extra_compile_args = ['/Ox', '/fp:fast']
    else:  # for GCC and Clang
        libraries = ["m"]
        extra_compile_args = ['-O3', '-ffast-math', '-g0']
    ext_modules = [
        Extension('abel.lib.direct', ['abel/lib/direct.pyx'],
                  include_dirs=[numpy.get_include()],
                  libraries=libraries,
                  extra_compile_args=extra_compile_args)
    ]
except ImportError:
    ext_modules = None
    print(f'''\
{'=' * 75}
Warning: Cython extensions will not be built, thus the abel.direct
         C implementation will not be available.
         To build them, install Cython (and NumPy), then reinstall PyAbel
         using pip with the --no-build-isolation option.
{'=' * 75}''')


# Supply dynamic options to setuptools (static are taken from pyproject.toml;
# the license option in pyproject.toml is incompatible (!) between Setuptools
# <77 and >=77, thus it must be provided here; the license file is found
# automatically in both cases)
setup(version=__version__,
      long_description=long_description,
      ext_modules=ext_modules,
      license='MIT')
