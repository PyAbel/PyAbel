Changelog
=========

2016-04-15 - Made changes to .travis.yml to have Travis CI automatically release new version of PyAbel to PyPi

2016-03-19 - onion-peeling algorithm now available (see #53)

2016-03-15 - Changed abel.transform to be a class rather than a function. 
	The previous syntax of: ::

	   abel.transform(IM)['transform']

	has been replaced with: ::

	    abel.transform(IM).transform
