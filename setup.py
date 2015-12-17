import sys
import re
import os.path
from setuptools import setup, find_packages, Extension
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
import numpy as np

try:
    from Cython.Distutils import build_ext
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = False
    _cython_installed = True
except ImportError:
    _cython_installed = False
    build_ext = object # just avoid a syntax error in TryBuildExt, this is not used anyway
    print('='*80)
    print('Warning: Cython extensions will not be built as Cython is not installed!\n'\
          '         This means that the abel.direct C implementation will not be available.')
    print('='*80)




# a define the version sting inside the package
# see https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package 
VERSIONFILE="abel/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


if sys.platform != 'win32':
    compile_args =  dict( extra_compile_args=['-O2', '-march=native'],
                 extra_link_args=['-O2', '-march=native'])
else:
    compile_args = {}

# Optional compilation of Cython modules adapted from
# https://github.com/bsmurphy/PyKrige which was itself adapted from a StackOverflow post


ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)

class TryBuildExt(build_ext):
    def build_extensions(self):
        try:
            build_ext.build_extensions(self)
        except ext_errors:
            print("**************************************************")
            print("WARNING: Cython extensions failed to build (used in abel.direct).\n"
                  "Typical reasons for this problem are:\n"
                  "  - a C compiler is not installed or not found\n"
                  "  - issues using mingw compiler on Windows 64bit (experimental support for now)\n"
                  "This only means that the abel.direct C implementation will not be available.\n")
            print("**************************************************")
            # continue the install
            pass
        except:
            raise


ext_modules=[
    Extension("abel.lib.direct",
             [os.path.join("abel","lib","direct.pyx")],
             libraries=["m"],
             **compile_args),
    ]

if _cython_installed:
    setup_args = {'cmdclass': {'build_ext': TryBuildExt},
                  'include_dirs': [ np.get_include() ],
                  'ext_modules': ext_modules}
else:
    setup_args = {}


setup(name='PyAbel',
      version=version,
      description='A Python package for forward and inverse Abel transforms',
      author='The PyAbel Team',
      url='https://github.com/PyAbel/PyAbel',
      license='GPLv3',
      packages=find_packages(),
      package_data={'abel': ['tests/data/*' ]},
      install_requires=[
              "numpy >= 1.6",
              "setuptools >= 16.0",
              "scipy >= 0.14",
              ],
      test_suite="abel.tests.run_cli",
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'Topic :: Software Development :: Libraries :: Python Modules'

          # Pick your license as you wish (should match "license" above)
           'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
      ],
      **setup_args
     )

