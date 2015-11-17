#!/usr/bin/python

from setuptools import setup


setup(name='PyAbel',
      version='0.5.0',
      description='A Python package for inverse Abel transforms',
      author='Dan Hickstein',
      packages=['abel'],
      package_data={'abel': ['abel/data/*','abel/tests/data/*' ]},
      #test_suite="BASEX.tests.run"
     )

