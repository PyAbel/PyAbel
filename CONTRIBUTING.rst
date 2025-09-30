Contributing to PyAbel
======================

PyAbel is an open-source project, and we welcome improvements! Please let us know about any issues with the software, even if it's just a typo. The easiest way to get started is to open a `new issue <https://github.com/PyAbel/PyAbel/issues>`__.

If you would like to make a `pull request <https://github.com/PyAbel/PyAbel/pulls>`__, the following information may be useful.


Rebasing
--------

If possible, before submitting your pull request please `rebase <https://git-scm.com/book/en/v2/Git-Branching-Rebasing>`__ your fork on the last master on PyAbel::

    # Add the remote, call it "upstream" (only the first time)
    git remote add upstream https://github.com/PyAbel/PyAbel.git

    # Fetch all the branches of that remote into remote-tracking branches,
    # such as upstream/master:

    git fetch upstream

    # Make sure that you're on your master branch
    # or any other branch you are working on

    git checkout master  # or your other working branch

    # Rewrite your master branch so that any commits of yours that
    # aren't already in upstream/master are replayed on top of that
    # other branch:

    git rebase upstream/master

    # Push the changes to your fork

    git push -f


Code style
----------

We hope that the PyAbel code will be understandable, hackable, and maintainable for many years to come. So, please use good coding style, include plenty of comments, use docstrings for functions, and pick informative variable names.

PyAbel attempts to follow `PEP8 <https://peps.python.org/pep-0008/>`__ style whenever possible, since the PEP8 recommendations typically produce code that is easier to read. You can check your code using `pycodestyle <https://pypi.org/project/pycodestyle/>`__, which can be called from the command line or incorporated right into most text editors. Also, PyAbel is using automated pycodestyle checking of all pull requests using `pep8speaks <https://github.com/apps/pep8-speaks>`__. However, `producing readable code <https://peps.python.org/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds>`__ is the primary goal, so please go ahead and break the rules of PEP8 when doing so improves readability. For example, if a section of your code is easier to read with lines slightly longer than 79 characters, then use the longer lines.


Unit tests
----------

Before submitting a pull request, be sure to run the unit tests. The test suite can be run from within the PyAbel package with ::
    
    pytest
    
For more detailed information, the following can be used::

    pytest abel/  -v  --cov=abel

Note that this requires that you have `pytest <https://docs.pytest.org/en/latest/>`__ and (optionally) `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`__ installed. You can install these with ::

    pip install pytest pytest-cov


Documentation
-------------

PyAbel uses Sphinx and `Napoleon <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html>`__ to process Numpy-style docstrings and is synchronized to `pyabel.readthedocs.io <https://pyabel.readthedocs.io>`__. To build the documentation locally, you will need `Sphinx <https://www.sphinx-doc.org/>`__ and the `sphinx_rtd_theme <https://github.com/readthedocs/sphinx_rtd_theme>`__. You can install them using ::

    pip install sphinx
    pip install sphinx_rtd_theme

Once you have these packages installed, you can build the documentation using ::

    cd PyAbel/doc/
    make html

Then you can open ``doc/_build/html/index.html`` to look at the documentation. Sometimes you need to use ::

    make clean
    make html

to clear out the old documentation and get things to re-build properly.

When you get tired of typing ``make html`` every time you make a change to the documentation, it's nice to use `sphinx-autobuild <https://pypi.org/project/sphinx-autobuild/>`__ to automatically update the documentation in your browser for you. So, install sphinx-autobuild using ::

    pip install sphinx-autobuild

Now you should be able to ::

    cd PyAbel/doc/
    make livehtml

which should launch a browser window displaying the docs. When you save a change to any of the docs, the re-build should happen automatically and the docs should update in a matter of a few seconds.

Alternatively, `restview <https://pypi.org/project/restview/>`__ is a nice way to preview the ``.rst`` files.


Changelog
---------

If the change is significant (more than just a typo-fix), please leave a short note about the change in `CHANGELOG.rst <https://github.com/PyAbel/PyAbel/blob/master/CHANGELOG.rst>`__, at the bottom of the "Unreleased" section (the PR number can be added later).


Adding a new forward or inverse Abel implementation
---------------------------------------------------

We are always looking for new implementation of forward or inverse Abel transform, therefore if you have an implementation that you would want to contribute to PyAbel, don't hesitate to do so.

In order to allow a consistent user experience between different implementations and ensure an overall code quality, please consider the following points in your pull request.


Naming conventions
~~~~~~~~~~~~~~~~~~

The implementation named ``<implementation>``, located under ``abel/<implementation>.py``, should use the following naming system for top-level functions:

- ``<implementation>_transform`` — core transform (when defined)
- ``_bs_<implementation>`` — function that generates  the basis sets (if necessary)


Unit tests
~~~~~~~~~~
To detect issues early, the submitted implementation should have the following properties and pass the corresponding unit tests:

1. The reconstruction has the same shape as the original image. Currently all transform methods operate with odd-width images and should raise an exception if provided with an even-width image.

2. Given an array with all 0 elements, the reconstruction should also be a 0 array.

3. The implementation should be able to calculate the inverse (or forward) transform of a Gaussian function defined by a standard deviation ``sigma``, with better than a 10 % relative error with respect to the analytical solution for ``0 < r < 2*sigma``.

Unit tests for a given implementation are located under ``abel/tests/test_<implementation>.py``, which should contain at least the following 3 functions:

- ``test_<implementation>_shape``
- ``test_<implementation>_zeros``
- ``test_<implementation>_gaussian``

.. |test_basex.py| replace:: ``abel/tests/test_basex.py``
.. _test_basex.py: https://github.com/PyAbel/PyAbel/blob/master/abel/tests/test_basex.py

See |test_basex.py|_ for a concrete example.


Dependencies
------------

.. |setup.py| replace:: ``setup.py``
.. _setup.py: https://github.com/PyAbel/PyAbel/blob/master/setup.py

The current list of dependencies can be found in |setup.py|_. Please refrain from adding new dependencies, unless it cannot be avoided.


Citations
---------

Each version of PyAbel that is released triggers a new DOI on Zenodo, so that people can cite the project. If you would like your name added to the author list on Zenodo, please include it in ``.zenodo.json``.


----

For maintainers: Releasing a new version
----------------------------------------

First, make a pull request that does the following:

- Increment the version number in ``abel/_version.py``.
- Update ``CHANGELOG.rst`` by renaming the "Unreleased" section to the new version and adding the expected release date.
- Use the changelog to write version release notes that can be included as a comment in the PR and will be used later.
- Update copyright years in ``doc/conf.py``.

After the PR is merged:

- Press the "Draft a new release" button on the `Releases <https://github.com/PyAbel/PyAbel/releases>`__ page and create a new tag, matching the new version number (for example, "v1.2.3" for version "1.2.3").
- Copy and paste the release notes from the PR into the release notes.
- Release it!
- Check that the new version appears `on Zenodo <https://zenodo.org/record/594858>`__. If it does not, toggle the GitHub synchronization off and on in Zenodo (see Dan's `comment <https://github.com/PyAbel/PyAbel/issues/401#issuecomment-3315466954>`__).
- PyAbel source (sdist) and binary (wheels) distributions should be automatically built by the corresponding `GitHub actions <https://github.com/PyAbel/PyAbel/actions>`__ and published to PyPI (see `PR #395 <https://github.com/PyAbel/PyAbel/pull/395>`__ and `#403 <https://github.com/PyAbel/PyAbel/pull/403>`__).
- In parallel, Read the Docs should build the docs and activate the new version; check this `on Read the Docs <https://readthedocs.org/projects/pyabel/versions/>`__.
- Check that the new package is `on PyPI <https://pypi.org/project/PyAbel/#history>`__ (the "Example of use" output image in the project description will appear only after the new version is activated on Read the Docs).
- A bot should automatically make a PR on the `conda-forge repo <https://github.com/conda-forge/pyabel-feedstock>`__. This can take several hours and needs to be merged manually.
- Check that the new conda packages are `on Anaconda.org <https://anaconda.org/conda-forge/pyabel/files>`__.

Notes:

- The workflows to build sdist and wheels can also be run manually for testing the distributions. This also runs PyAbel tests on more platforms than routine PR tests and helps to catch errors before making a release.
- Running the "Publish to (Test)PyPI" workflow manually will publish the current (or selected) version `to TestPyPI <https://test.pypi.org/project/PyAbel/#history>`__. However, TestPyPI will reject attempts to publish a package with any version previously published on TestPyPI, even if it was deleted. Thus the version in ``abel/_version.py`` must be made unique (by using ``rc``, ``.post`` or ``.dev`` suffixes; see `Version specifiers <https://packaging.python.org/en/latest/specifications/version-specifiers/#version-scheme>`__), maybe in a separate branch, before running the workflow. **Do not create a new tag**, as this will initiate the actual release process.
- Pre-releases (with ``rc`` suffixes) do not trigger automatic Read the Docs updates, but the new version can be activated there manually. The conda-forge bot is triggered only by final releases ("latest version") on PyPI.
