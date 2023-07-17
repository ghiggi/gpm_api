Contributing guide
===========================

Hi! Thanks for taking the time to contribute to GPM-API.

You can contribute in many ways :

-  Join the
   `discussion <https://github.com/ghiggi/gpm_api/discussions>`__
- Report `issues <#issue-reporting-guidelines>`__
- Any others code improvements are welcome !


Before adding your contribution, please make sure to take a moment
and read through the following documnents :

- `Code of Conduct <https://github.com/ghiggi/gpm_api/blob/main/CODE_OF_CONDUCT.md>`__
- `Contributing environment setup <#contributing-environment-setup>`__
- `Contributing process <#contributing-process>`__
- `Code review checklist <#code-review-checklist>`__



Issue Reporting Guidelines
--------------------------

-  Always use one available `issue
   templates <https://github.com/ghiggi/gpm_api/issues/new/choose>`__
-  If you don’t find the required GitHub issue template, please ask for a new template.


GitHub
-----------------------

**We Develop with Github !**

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

We use `GitHub flow <https://docs.github.com/en/get-started/quickstart/github-flow>`__.
So all code changes happen through Pull Requests (PRs).




Contributing environment setup
-----------------------------------

**First Time Contributors ?**

Please follow the following steps to install your developing environment :

-  Setting up the development environment
-  Install pre-commit hooks

Setting up the development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You will need python to set up the development environment. See `the installation guide <https://gpm-api.readthedocs.io/en/latest/installation.html>`__
for further explanations.

Install pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~

After setting up your development environment, install the git pre-commit hook by executing the following command in the repository’s
root:

::

   pre-commit install


The pre-commit hooks are scripts executed automatically in every commit
to identify simple code quality issues. When an issue is identified
(the pre-commit script exits with non-zero status), the hook aborts the
commit and prints the error. Currently, GPM-API only tests that the
code to be committed complies with black's format style and the ruff linter.

In case that the commit is aborted, you only need to run black agains you code.
This can be done by running   ``black .``  or   ``ruff check .``

.. note::
	To maintain consistency, please use version and configuration defined into `.pre-commit-config.yaml`.



The can also be done with  ``pre-commit run --all-files``. This is recommended since it
indicates if the commit contained any formatting errors (that are automatically corrected).


More info on pre-commit and CI tools are provided in the Code quality and testing section  `Code quality and testing section <https://gpm-api.readthedocs.io/en/latest/contributors_guidelines.html#code-quality-control>`__



Contributing process
-----------------------

**How to contribute ?**


Here is a brief overview of the steps that each GPM-API developer must follow to contribute to the repository.

1. Fork the repository.
2. Create a new branch for each contribution.
3. Work on your changes.
4. Test your changes.
5. Push your local changes to your fork repository.
6. Create a new Pull Request in GitHub.


.. image:: /static/collaborative_process.png




Fork the repository
~~~~~~~~~~~~~~~~~~~

Once you have set the development environment (see `Setting up the development environment`_), the next step is creating
your local copy of the repository, where you will commit your
modifications. The steps to follow are:

1. Set up Git on your computer

2. Create a GitHub account (if you don’t have one)

3. Fork the repository in your GitHub.

4. Clone a local copy of your fork. For example:

::

   git clone https://github.com/<your-account>/gpm_api.git

Done! Now you have a local copy of the GPM-API repository.

Create a new branch
~~~~~~~~~~~~~~~~~~~

Each contribution should be made in a separate new branch of your forked repository.
Working on the main branch
is reserved for Core Contributors only. Core Contributors are developers
that actively work and maintain the repository. They are the only ones
who accept pull requests and push commits directly to the GPM-API
repository.

For more information on how to create and work with branches, see
`“Branches in a
Nutshell” <https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>`__
in the Git documentation.

Please define the name of your branch based on the scope of the contribution. Try to strictly stick to the following guidelines:

-  If you fix a bug: ``bugfix-<some_key>-<word>``
-  If you improve the documentation: ``doc-<some_key>-<word>``
-  If you add a new feature: ``feature-<some_key>-<word>``
-  If you refactor some code: ``refactor-<some_key>-<word>``
-  If you optimize some code: ``optimize-<some_key>-<word>``


\* Guidelines for the `data_source` :

- 	We use the institution name when campaign data spans more than 1 country (i.e. ARM, GPM)
- 	We use the country name when all campaigns (or sensor networks) are inside a given country.



Work on your changes
~~~~~~~~~~~~~~~~~~~~


We follow the pep8 and the python-guide writing style

-  `Code Style — The Hitchhiker's Guide to
   Python <https://docs.python-guide.org/writing/style/>`__

To ensure a minimal style consistency, we use
`black <https://black.readthedocs.io/en/stable/>`__ to auto-format
the source code. The black configuration used in the GPM-API project is
defined in the pyproject.toml, and it is automatically detected by
black (see above).



**Docstrings**

Every module, function, or class must have a docstring that describe its
purpose and how to use it. The docstrings follows the conventions
described in the `PEP
257 <https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings>`__
and the `Numpy’s docstrings
format <https://numpydoc.readthedocs.io/en/latest/format.html>`__.

Here is a summary of the most important rules:

-  Always use triple quotes for doctrings, even if it fits a single
   line.

-  For one-line docstring, end the phrase with a period.

-  Use imperative mood for all docstrings (“””Return some value.”””)
   rather than descriptive mood (“””Returns some value.”””).

Here is an example of a docstring:

::

    def adjust_lag2_corrcoef1(gamma_1, gamma_2):
       """
       A simple adjustment of lag-2 temporal autocorrelation coefficient to
       ensure that the resulting AR(2) process is stationary when the parameters
       are estimated from the Yule-Walker equations.

       Parameters
       ----------
       gamma_1 : float
         Lag-1 temporal autocorrelation coeffient.
       gamma_2 : float
         Lag-2 temporal autocorrelation coeffient.

       Returns
       -------
       out : float
         The adjusted lag-2 correlation coefficient.
       """


If you are using VS code, you can install the  `autoDocstring <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_ extension to automatically create such preformatted docstring.

You should configure VS code as follow :


.. image:: /static/vs_code_settings.png


The convention we adopt for our docstrings is the numpydoc string convention.


Code quality control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


To maintain a high code quality, Black and Ruff are defined in the ``.pre-commit-config.yaml`` file. These tools are run for every Pull Request on Github and can also be run locally.


+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+-------------------------------------------+
|  Tool                                                                                         | Aim                                                              | pre-commit | CI/CD | Version                                   |
+===============================================================================================+==================================================================+============+=======+===========================================+
| `Black <https://black.readthedocs.io/en/stable/>`__                                           | Python code formatter                                            | 👍         | 👍    | 22.8.0                                    |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+-------------------------------------------+
| `Ruff  <https://github.com/charliermarsh/ruff>`__                                             | Python linter                                                    | 👍         | 👍    | 0.0.2570                                  |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+-------------------------------------------+



**pre-commit**

To run pre-commit (black + Ruff) locally :

.. code-block:: bash

   pre-commit run --all-files


This is recommended since it
indicates if the commit contained any formatting errors (that are automatically corrected).




**Black**

To run Black locally :

.. code-block:: bash

	black .



.. note::
	To maintain consistency, make sure to stick to the version defined in the `.pre-commit-config.yaml` file. This version will be used in the CI.





**Ruff**

To run Ruff locally :

.. code-block:: bash

	ruff check .


.. note::
	To maintain consistency, make sure to stick to the version and the rule configuration defined in the `.pre-commit-config.yaml` file. This information is used in the CI.








In the table below, some CI tool are mentioned for your information, but does not need to be installed on your computer. They are automatically run when you push your changes to the main repository via a GitHub Pull Request.


+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
|  Tool                                                                                         | Aim                                                              | Python version                            |
+===============================================================================================+==================================================================+===========================================+
| `pre-commit.ci   <https://pre-commit.ci/>`__                                                  | Run pre-commit (as defined in `.pre-commit-config.yaml` )        |                                           |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
| `CodeBeat      <https://codebeat.co/>`__                                                      | Automated code review and analysis tools                         | all versions according to GitHub workflow |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
| `CodeScene <https://codescene.com/>`__                                                        | Automated code review and analysis tools                         | all versions according to GitHub workflow |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
| `CodeFactor <https://www.codefactor.io/>`__                                                   | Automated code review and analysis tools                         | all versions according to GitHub workflow |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+







Code testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Every code change must be tested !




**Pytest**

GPM-API tests are written using the third-party `pytest <https://docs.pytest.org>`_ package.



The tests located in the ``/gpm_api/tests`` folder are used to test various functions of the code and are automatically run when changes are pushed to the main repository through a GitHub Pull Request.

.. code-block:: bash

	pytest gpm_api/tests


The Continuous Integration (CI) on GitHub runs tests and analyzes code coverage. The following tools are used:


+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
|  Tool                                                                                         | Aim                                                              | Version                                   |
+===============================================================================================+==================================================================+===========================================+
| `Pytest  <https://docs.pytest.org>`__                                                         | Execute unit tests and functional tests                          |                                           |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
| Coverage                                                                                      | Measure the code coverage of the project's unit tests            | all versions according to GitHub workflow |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
| `CodeCov    <https://about.codecov.io/>`__                                                    | Uses the "coverage" package to generate a code coverage report.  | all versions according to GitHub workflow |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
| `Coveralls    <https://coveralls.io/>`__                                                      | Uses the "coverage" to track the quality of your code over time. | all versions according to GitHub workflow |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
| `CodeBeat      <https://codebeat.co/>`__                                                      | Automated code review and analysis tools                         | all versions according to GitHub workflow |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
| `CodeScene <https://codescene.com/>`__                                                        | Automated code review and analysis tools                         | all versions according to GitHub workflow |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+
| `CodeFactor <https://www.codefactor.io/>`__                                                   | Automated code review and analysis tools                         | all versions according to GitHub workflow |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------+



Push your changes to your fork repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During this process, pre-commit hooks will be run. Your commit will be
allowed only if quality requirements are fulfilled.

If you encounter errors, Black and Ruff can be run using the following command:

::

   pre-commit run --all-files

We follow a `commit message convention <https://www.conventionalcommits.org/en/v1.0.0/>`__, to have consistent git messages.
The goal is to increase readability and ease of contribution.



Create a new Pull Request in GitHub.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your code has been uploaded into your GPM-API fork, you can create
a Pull Request (PR) to the GPM-API main branch.

**Recommendation for the pull request**

-  Add screenshots or GIFs for any UI changes. This will help the person reviewing your code to understand what you’ve changed and how it
   works.

-  Please use the pertinent template for the pull request, and fill it out accurately.

-  It’s OK to have multiple small commits as you work on the PR - GitHub
   will automatically squash it before merging.

-  If adding a new feature:

   -  Add accompanying test case.
   -  Provide a convincing reason to add this feature. Ideally, you
      should open a suggestion issue first and have it approved before
      working on it.
   -  Present your issue in the ‘discussion’ part of this repo

-  If fixing bug:

   -  If you are resolving a special issue, add ``(fix #xxxx[,#xxxx])``
      (#xxxx is the issue id) in your PR title for a better release log,
      e.g. ``update entities encoding/decoding (fix #3899)``.
   -  Provide a detailed description of the bug in the PR. Live demo
      preferred.
   -  Add appropriate test coverage if applicable.

.. _section-1:

Code review checklist
---------------------

-  Ask to people to review your code:

   -  a person who knows the domain well and can spot bugs in the
      business logic;
   -  an expert in the technologies you’re using who can help you
      improve the code quality.

-  When you’re done with the changes after a code review, do another
   self review of the code and write a comment to notify the reviewer,
   that the pull request is ready for another iteration.
-  Resolve all the review comments, making sure they are all addressed before another review iteration.
-  Make sure you don’t have similar issues anywhere else in your pull
   request.
-  If you’re not going to follow a code review recommendations, please add a comment explaining why you think the reviewer suggestion is not relevant.
-  Avoid writing comment like “done” of “fixed” on each code review
   comment. Reviewers assume you’ll do all suggested changes, unless you
   have a reason not to do some of them.
-  Sometimes it’s okay to postpone changes — in this case you’ll need to
   add a ticket number to the pull request and to the code itself.

.. _section-2:


Credits
-------

Thank you to all the people who have already contributed to GPM-API repository!

If you have contributed data and/or code to GPM-API, add your name to the `AUTHORS.md <https://github.com/ghiggi/gpm_api/blob/main/AUTHORS.md>`__ file.
