#!/usr/bin/python

from setuptools import setup


setup(name='BASEX',
      version='0.4',
      description='A Python implementation of the BASEX algorithm',
      author='Dan Hickstein',
      packages=['BASEX'],
      package_data={'BASEX': ['BASEX/data/*','BASEX/tests/data/*' ]},
      #test_suite="BASEX.tests.run"
     )

