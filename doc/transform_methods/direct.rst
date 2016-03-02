Direct
=====


Introduction
------------

This method attempts a direct integration of the Abel transform integral. It makes no assumptions about the data (apart from cylindrical symmetry), but it typically requires fine sampling to converge. Such methods are typically inefficient, but thanks to this Cython implementation (by Roman Yurchuk), this 'direct' method is competitive with the other methods.


How it works
------------

Information about the algorithm and the numerical optimizations is contained in `PR #52 <https://github.com/PyAbel/PyAbel/pull/52>`_

When to use it
--------------

When a robust forward transform is required, this method works quite well. It is not typically recommended for the inverse transform, but it can work well for smooth functions that are finely sampled.


How to use it
-------------

To complete the forward or inverse transform of a full image with the direct method, simply use the :func:`abel.transform` function: ::

	abel.transform(myImage, method='direct', direction='forward')
	abel.transform(myImage, method='direct', direction='inverse')
	

If you would like to access the Direct algorithm directly (to transform a right-side half-image), you can use :func:`abel.direct.direct_transform`.


Citation
--------
[1] `Dribinski et al, 2002 (Rev. Sci. Instrum. 73, 2634) <http://dx.doi.org/10.1063/1.1482156>`_, (`pdf <http://www-bcf.usc.edu/~reisler/assets/pdf/67.pdf>`_)