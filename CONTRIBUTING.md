# Contributing to PyAbel




### Adding a new forward or inverse Abel implementation 

We are always looking for new implementation of forward or inverse Abel transform, therefore if you have an implementation that you would want to contribute to PyAbel, don't hesitate to do so. 


In order to allow a consistent user experience between different implementations, and insure an overall code quality, please consider the following points in your pull request.

##### Naming conventions

The implementation named `<implementation>`, located under `abel/<implementation>.py` should use the following naming system for top-level functions,

 -  `fabel_<implemenation>`  :  forward transform (when defined)
 -  `iabel_<implemenation>` :  inverse implementation (when defined)
 -  `_bs_<implementation>` :  function that generates  the basis sets (if necessary)
 -  `bs_<implementation>_cached` : function that loads the basis sets from disk, and generates them if they are not found (if necessary).


##### Unit tests

As to detect issues early and avoid regressions, the submitted implementation should have the following properties and pass the corresponding unit tests,

 1. The reconstruction has the same shape as the original image for the parity it support. When provided with a image size with a parity it does not support a clear exception should be raised).

 2. Given an array of 0 elements, the reconstruction should also be a 0 array.
  
 3. The implementation should be able to calculated the inverse (or forward) transform of a Gaussian function defined by a standard deviation `sigma`, with better then a `10 %` relative error with respect to the analytical solution for `0 > r > 2*sigma`.


The test suite can be run from within the PyAbel package with,
  
    nose -s  abel/tests/ --verbosity=2  --with-coverage --cover-package=abel

or, from any folder with,
    
   python  -c "import abel.tests; abel.tests.run_cli(coverage=True)"

which performs an equivalent call. 
  

##### Dependencies

The current list of dependencies can be found in [`setup.py`](https://github.com/PyAbel/PyAbel/blob/master/setup.py). Unless it cannot be avoided, refraining from adding new dependencies is preferred. 

##### Before merging

As to keep a clean git history, before merging your pull request please rebase your fork on the last master on PyAbel/. This could be done  [as explained in this post](https://stackoverflow.com/questions/7244321/how-to-update-a-github-forked-repository),
   
    # Add the remote, call it "upstream" (only the fist time)
    git remote add upstream git@github.com:PyAbel/PyAbel.git

    # Fetch all the branches of that remote into remote-tracking branches,
    # such as upstream/master:

    git fetch upstream

    # Make sure that you're on your master branch 
    # or any other branch your are working on

    git checkout master  # or your other working branch

    # Rewrite your master branch so that any commits of yours that
    # aren't already in upstream/master are replayed on top of that
    # other branch:

    git rebase upstream/master

See [this wiki](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request) for more information.
