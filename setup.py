import os
import sys
import re
import os.path
from setuptools import setup, find_packages, Extension
from setuptools.errors import CCompilerError, ExecError, PlatformError

# define the version string inside the package, see:
# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
VERSIONFILE = "abel/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)

if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


# try to import numpy and Cython to build Cython extensions:
try:
    import numpy as np
    from Cython.Distutils import build_ext
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = False
    _cython_installed = True
except ImportError:
    _cython_installed = False
    setup_args = {}
    print('='*80)
    print('Warning: Cython extensions will not be built as Cython is not installed!\n'\
          '         This means that the abel.direct C implementation will not be available.')
    print('='*80)

if _cython_installed:  # if Cython is installed, we will try to build direct-C

    if sys.platform == 'win32':
        extra_compile_args = ['/Ox', '/fp:fast']
        libraries = []
    else:
        extra_compile_args = ['-Ofast', '-g0']
        libraries = ["m"]

    # Optional compilation of Cython modules adapted from
    # https://github.com/bsmurphy/PyKrige which was itself
    # adapted from a StackOverflow post

    class TryBuildExt(build_ext):
        """Class to  build the direct-C extensions."""

        def build_extensions(self):
            """Try to build the direct-C extension."""
            try:
                build_ext.build_extensions(self)
            except (CCompilerError, ExecError, PlatformError):
                print("**************************************************")
                print("WARNING: Cython extensions failed to build (used in abel.direct).\n"
                      "Typical reasons for this problem are:\n"
                      "  - a C compiler is not installed or not found\n"
                      "  - issues using mingw compiler on Windows 64bit (experimental support for now)\n"
                      "This only means that the abel.direct C implementation will not be available.\n")
                print("**************************************************")
            except:
                raise

    ext_modules = [
        Extension("abel.lib.direct",
                  [os.path.join("abel", "lib", "direct.pyx")],
                  include_dirs=[np.get_include()],
                  libraries=libraries, 
                  extra_compile_args=extra_compile_args)]

    setup_args = {'cmdclass': {'build_ext': TryBuildExt},
                  'include_dirs': [np.get_include()],
                  'ext_modules': ext_modules}


# use README as project description on PyPI:
with open('README.rst') as file:
    long_description = file.read()

# but remove CI badges
long_description = re.sub(
    r'^.+?https://(github\.com/.+/pytest.yml|ci\.appveyor\.com/).*?\n', '',
    long_description, flags=re.MULTILINE,
    count=4)  # limit to top 2 pairs of image + target

# and change GH and RTD links to specific PyAbel version
long_description = long_description.\
    replace('https://github.com/PyAbel/PyAbel/tree/master/',
            'https://github.com/PyAbel/PyAbel/tree/v' + version + '/').\
    replace('https://pyabel.readthedocs.io/en/latest/',
            'https://pyabel.readthedocs.io/en/v' + version + '/')


setup(name='PyAbel',
      version=version,
      description='A Python package for forward and inverse Abel transforms',
      author='The PyAbel Team',
      url='https://github.com/PyAbel/PyAbel',
      license='MIT',
      packages=find_packages(),
      # last versions available for Python 3.7 and tested
      install_requires=["numpy >= 1.21",
                        "scipy >= 1.7",
                        "setuptools >= 68.0"],
      package_data={'abel': ['tests/data/*']},
      long_description=long_description,
      long_description_content_type='text/x-rst',
      classifiers=[
          # How mature is this project? Common values are
          #  3 - Alpha
          #  4 - Beta
          #  5 - Production/Stable
          'Development Status :: 4 - Beta',

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
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3.13',
          ],
      **setup_args
      )
