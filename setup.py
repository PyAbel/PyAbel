#!/usr/bin/python

from setuptools import setup, find_packages


setup(name='PyAbel',
      version='0.5.0',
      description='A Python package for inverse Abel transforms',
      author='Dan Hickstein',
      packages=find_packages(),
      package_data={'abel': ['abel/data/*','abel/tests/data/*' ]},
      #test_suite="BASEX.tests.run"
     )

