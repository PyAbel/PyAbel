import sys
import re
import os.path
from setuptools import setup, find_packages, Extension
import numpy as np
from Cython.Distutils import build_ext
import Cython.Compiler.Options


Cython.Compiler.Options.annotate = False

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
    compile_args =  dict( extra_compile_args=['-O2', '-march=core2', '-mtune=native'],
                 extra_link_args=['-O2', '-march=core2', '-mtune=native'])
else:
    compile_args = {}

ext_modules=[
    Extension("abel.lib.direct",
             [os.path.join("abel","lib","direct.pyx")],
             libraries=["m"],
             **compile_args),
    ]


setup(name='PyAbel',
      version=version,
      description='A Python package for inverse Abel transforms',
      author='Dan Hickstein',
      packages=find_packages(),
      package_data={'abel': ['tests/data/*' ]},
      cmdclass= {'build_ext': build_ext},
      ext_modules= ext_modules,
      include_dirs=[ np.get_include() ],
      install_requires=[
              "numpy >= 1.6",
              "setuptools >= 16.0",
              "scipy >= 0.14",
              ],
      test_suite="abel.tests.run_cli"
     )

