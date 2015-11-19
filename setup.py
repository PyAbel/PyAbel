#!/usr/bin/python

from setuptools import setup, find_packages


setup(name='PyAbel',
      version='0.5.0',
      description='A Python package for inverse Abel transforms',
      author='Dan Hickstein',
      packages=find_packages(),
      package_data={'abel': ['tests/data/*' ]},
      install_requires=[
              "numpy >= 1.6",
              "setuptools >= 16.0",
              "scipy >= 0.14",
              ],
      test_suite="abel.tests.run_cli"
     )

