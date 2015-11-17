#!/usr/bin/python

from setuptools import setup


setup(name='pyBASEX',
      version='0.5.0',
      description='A Python implementation of the BASEX algorithm',
      author='Dan Hickstein',
      packages=['basex'],
      package_data={'basex': ['basex/data/*','basex/tests/data/*' ]},
      #test_suite="BASEX.tests.run"
     )

