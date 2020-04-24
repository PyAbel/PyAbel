Comparison of Abel Transform Methods
====================================


Introduction
------------

Each new Abel transform method claims to be the best, providing a lower-noise, more accurate transform. The point of PyAbel is to provide an easy platform to try several Abel transform methods and determine which provides the best results for a specific dataset.

So far, we have found that for a high-quality dataset, all of the transform methods produce good results.


Speed benchmarks
----------------

The :class:`abel.benchmark.AbelTiming` class provides the ability to benchmark the speeds of the Abel transform algorithms.


Examples
^^^^^^^^

To give some sense of the relative and absolute speeds of each method, here we provide the results obtained on a system with a 3.0 GHz Intel i7-9700 processor and 32 GB RAM running GNU/Linux (see also our :ref:`publications <READMEcitation>` for the older 3.4 GHz Intel i7-6700 results).


Sustained transform speed
"""""""""""""""""""""""""

.. plot:: benchmarks/transform_time.py

.. plot:: benchmarks/throughput.py

* All method are benchmarked using their default parameters (exceptions are noted below).
* **basex(var)** means “variable regularization”, that is changing the regularization parameter for each transformed image.
* **direct_C** and **direct_Python** correspond to the “direct” method using its C (Cython) and Python backends respectively.
* **linbasex** and **rbasex** show whole-image (*n* × *n*) transforms, while all other methods show half-image (*n* rows, (*n* + 1)/2 columns) transforms.
* **rbasex(None)** means no output-image creation (only the transformed radial distributions).

Basis-set generation
""""""""""""""""""""

.. plot:: benchmarks/basis_time.py


General advice
^^^^^^^^^^^^^^

Most of the methods rely on matrix operations, and therefore their speed depends significantly on the performance of the underlying linear-algebra libraries. Different NumPy/SciPy distributions use different libraries by default, and some also provide a choice between several libraries. If the transform speed is important, it is advisable to run the benchmarks on all available configurations to select the fastest for the specific combination of the transform method, operating system and hardware.

Among the widely available options, the `Intel Math Kernel Library <https://en.wikipedia.org/wiki/Math_Kernel_Library>`_ (MKL) generally provides the best performance for Intel CPUs, although its installed size is rather huge and its performance on AMD CPUs is quite poor. It is used by default in `Anaconda Python <https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)>`_. `OpenBLAS <https://en.wikipedia.org/wiki/OpenBLAS>`_ generally provides the best performance for AMD CPUs and reasonably good performance for Intel CPUs. It is used by default in some distributions. AMD develops numerical libraries optimized for its own CPUs, but they are `not yet <https://github.com/numpy/numpy/issues/7372>`_ officially integrated with NumPy/SciPy.

Another important issue for modern Intel CPUs is that they suffer a severe performance degradation when `denormal numbers <https://en.wikipedia.org/wiki/Denormal_number>`_ are encountered, which sometimes happens in the intermediate calculations even if the input and output are “normal”. In this case, configuring the CPU to treat denormals as zeros does help. There is no official way to achieve this in NumPy/SciPy, but a third-party module `daz <https://github.com/chainer/daz>`_ can be used for this purpose. At least some modern AMD CPUs are less or not affected by this issue, although it's always better to run the tests to make sure.


Transform quality
-----------------

...coming soon! ...
