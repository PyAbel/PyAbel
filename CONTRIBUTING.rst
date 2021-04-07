Contributing to PyAbel
======================


PyAbel is an open-source project, and we welcome improvements! Please let us know about any issues with the software, even if's just a typo. The easiest way to get started is to open a `new issue <https://github.com/PyAbel/PyAbel/issues>`_.

If you would like to make a Pull Request, the following information may be useful.


Change Log
----------

If the change is significant (more than just a typo-fix), please leave a short note about the change in `CHANGELOG.rst <https://github.com/PyAbel/PyAbel/blob/master/CHANGELOG.rst>`_


Unit tests
----------

Before submitting at Pull Request, be sure to run the unit tests. The test suite can be run from within the PyAbel package with ::
    
    pytest
    
For more detailed information, the following can be used::

    pytest abel/  -v  --cov=abel

Note that this requires that you have `pytest <https://docs.pytest.org/en/latest/>`_ and (optionally) `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ installed. You can install these with ::

    pip install pytest pytest-cov


Documentation
-------------

PyAbel uses Sphinx and `Napoleon <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html>`_ to process Numpy-style docstrings and is synchronized to `pyabel.readthedocs.io <http://pyabel.readthedocs.io>`_. To build the documentation locally, you will need `Sphinx <http://www.sphinx-doc.org/>`_, the `recommonmark <https://github.com/rtfd/recommonmark>`_ package, and the `sphinx_rtd_theme <https://github.com/snide/sphinx_rtd_theme/>`_. You can install them using ::

    pip install sphinx
    pip install recommonmark
    pip install sphinx_rtd_theme

Once you have these packages installed, you can build the documentation using ::

    cd PyAbel/doc/
    make html

Then you can open ``doc/_build/hmtl/index.html`` to look at the documentation. Sometimes you need to use ::

    make clean
    make html

to clear out the old documentation and get things to re-build properly.

When you get tired of typing ``make html`` every time you make a change to the documentation, it's nice to use use `sphix-autobuild <https://pypi.python.org/pypi/sphinx-autobuild>`_ to automatically update the documentation in your browser for you. So, install sphinx-autobuild using ::

    pip install sphinx-autobuild

Now you should be able to ::

    cd PyAbel/doc/
    make livehtml

which should launch a browser window displaying the docs. When you save a change to any of the docs, the re-build should happen automatically and the docs should update in a matter of a few seconds.

Alternatively, `restview <https://pypi.python.org/pypi/restview>`_ is a nice way to preview the ``.rst`` files.


Code Style
----------

We hope that the PyAbel code will be understandable, hackable, and maintainable for many years to come. So, please use good coding style, include plenty of comments, use docstrings for functions, and pick informative variable names.

PyAbel attempts to follow `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style whenever possible, since the PEP8 recommendations typically produces code that is easier to read. You can check your code using `pycodestyle <https://pypi.org/project/pycodestyle/>`_, which can be called from the command line or incorporated right into most text editors. Also, PyAbel is using automated pycodestyle checking of all Pull Requests using `pep8speaks <https://pep8speaks.com/>`_. However, `producing readable code <https://www.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds>`_ is the primary goal, so please go ahead and break the rules of PEP8 when doing so improves readability. For example, if a section of your code is easier to read with lines slightly longer than 79 characters, then use the longer lines.


Before merging
--------------

If possible, before merging your pull request please rebase your fork on the last master on PyAbel. This could be done `as explained in this post <https://stackoverflow.com/questions/7244321/how-to-update-a-github-forked-repository>`_::

    # Add the remote, call it "upstream" (only the fist time)
    git remote add upstream https://github.com/PyAbel/PyAbel.git

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

    # push the changes to your fork

    git push -f

See `this wiki <https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request>`_ for more information.


Adding a new forward or inverse Abel implementation
---------------------------------------------------

We are always looking for new implementation of forward or inverse Abel transform, therefore if you have an implementation that you would want to contribute to PyAbel, don't hesitate to do so.

In order to allow a consistent user experience between different implementations and insure an overall code quality, please consider the following points in your pull request.


Naming conventions
~~~~~~~~~~~~~~~~~~

The implementation named ``<implementation>``, located under ``abel/<implementation>.py``, should use the following naming system for top-level functions:

- ``<implemenation>_transform`` :  core transform (when defined)
- ``_bs_<implementation>`` :  function that generates  the basis sets (if necessary)


Unit tests
~~~~~~~~~~
To detect issues early, the submitted implementation should have the following properties and pass the corresponding unit tests:

1. The reconstruction has the same shape as the original image. Currently all transform methods operate with odd-width images and should raise an exception if provided with an even-width image.

2. Given an array with all 0 elements, the reconstruction should also be a 0 array.

3. The implementation should be able to calculated the inverse (or forward) transform of a Gaussian function defined by a standard deviation ``sigma``, with better than a 10 % relative error with respect to the analytical solution for ``0 < r < 2*sigma``.

Unit tests for a given implementation are located under ``abel/tests/test_<implemenation>.py``, which should contain at least the following 3 functions:

- ``test_<implementation>_shape``
- ``test_<implementation>_zeros``
- ``test_<implementation>_gaussian``

See ``abel/tests/test_basex.py`` for a concrete example.


Dependencies
------------

The current list of dependencies can be found in `setup.py <https://github.com/PyAbel/PyAbel/blob/master/setup.py>`_. Please refrain from adding new dependencies, unless it cannot be avoided.



Releasing on PyPi
-----------------

PyAbel should be automatically released on PyPi (see `PR #161 <https://github.com/PyAbel/PyAbel/pull/161>`_) whenever a new release is drafted on GitHub via the "Draft New Release" button on the `Releases page <https://github.com/PyAbel/PyAbel/releases>`_. But first, make a Pull Request that does the following:

- Increment the version number in abel/_version.py.
- Modify CHANGELOG.rst to include the new changes in the new version.


Citations
---------

Each version of PyAbel that is released triggers a new DOI on Zenodo, so that people can cite the project. If you would like you name added to the author list on Zenodo, please include it in ``.zenodo.json``. 
