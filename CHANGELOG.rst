Changelog
=========

Unreleased
----------
* Added odd angular orders to tools.vmi.Distributions (PR #266).

v0.8.3 (2019-08-16)
-------------------
* Fixed the GUI examples (example_GUI.py and example_simple_GUI.py)
  so that they work with the lastest versions of tk (#269).
* New tools.vmi.Distributions class for extracting radial intensity and
  anisotropy distributions (PR #257).
* Dropped PyAbel version from basex cache files (PR #260).

v0.8.2 (2019-07-11)
-------------------
* Added forward transform to basex method (PR #240).
* Corrected tools.transform_pairs.profile4 (PR #241).
* Removed tools.transform_pairs.profile8, which was the same as profile6
  (PR #241).
* Major changes in benchmark.AbelTiming class (PR #244, #252).
* Corrected image shift in onion_bordas method (PR #248).
* Changed gradient calculations in hansenlaw method to first-order finite
  difference (PR #249).
* Corrected background width for Dribinski sample image in
  tools.analytical.SampleImage (PR #255).

v0.8.1 (2018-11-29)
-------------------
* Implemented Tikhonov regularization in basex method (PR #227).
* Improvements and changes in basis caching for methods using basis sets
  (PR #227, #232, #235).
* Improvements in basis generation for basex method, allow any sigma (PR #235).
* Added intensity correction to basex method (PR #235).
* New tools.polynomial module for analytical Abel transform of piecewise
  polynomials (PR #235).


(2016-04-20)
------------
* Lin-BASEX method of Thomas Gerber and co-workers available (PR #177)

(2016-03-20)
------------
* Dasch two-point, three-point (updated), and onion-peeling available
  (PR #155).
 
(2016-03-15)
------------
* Changed abel.transform to be a class rather than a function. The previous
  syntax of abel.transform(IM)['transform'] has been replaced with
  abel.Transform(IM).transform.
