import sys
import re
import os
import os.path
from setuptools import setup, find_packages, Extension
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

# change behaviour of the setup.py on readthedocs.io
# https://read-the-docs.readthedocs.io/en/latest/faq.html#how-do-i-change-behavior-for-read-the-docs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    import numpy as np
    install_requires=[
          "numpy >= 1.6",
          "setuptools >= 16.0",
          "scipy >= 0.14",
          "six >= 1.10.0"
          ]
else:
    np = None
    install_requires=[]



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
    compile_args = dict( extra_compile_args=['-O2', '-march=native'],
                             extra_link_args=['-O2', '-march=native'])
    libraries = ["m"]
else:
    compile_args = dict( extra_compile_args=[])
    libraries = []

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
            if os.environ.get('CI'):
                # running on Travis CI or Appveyor CI
                if sys.platform == 'win32' and sys.version_info < (3, 0):
                    pass # Cython extensions are not built on Appveyor (Win) for PY2.7
                         # see PR #185
                else:
                    raise
            else:
                # regular install, Cython extensions won't be compiled
                pass
        except:
            raise

ext_modules=[
    Extension("abel.lib.direct",
             [os.path.join("abel","lib","direct.pyx")],
             libraries=libraries,
             **compile_args),
    ]

if _cython_installed and not on_rtd:
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
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      package_data={'abel': ['tests/data/*' ]},
      test_suite="abel.tests.run_cli",
      classifiers=[
      # How mature is this project? Common values are
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      'Development Status :: 3 - Alpha',

      # Indicate who your project is intended for
      'Intended Audience :: Science/Research',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Scientific/Engineering :: Physics',
      'Topic :: Scientific/Engineering :: Medical Science Apps.',
      'Topic :: Software Development :: Libraries :: Python Modules',

      # Pick your license as you wish (should match "license" above)
      'License :: OSI Approved :: MIT License',

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
