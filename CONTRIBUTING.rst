Contributors Guidelines
===========================

Hi! Thanks for taking the time to contribute to GPM-API.

You can contribute in many ways:

- Join the `GitHub Discussions <https://github.com/ghiggi/gpm_api/discussions>`__
- Report `issues <#issue-reporting>`__
- Add new features
- Add new retrievals
- Add new visualization tools
- Any others code improvements are welcome !

**We Develop with GitHub !**

We use GitHub to host code, to track issues and feature requests, as well as accept Pull Requests.
We use `GitHub flow <https://docs.github.com/en/get-started/quickstart/github-flow>`__.
So all code changes happen through Pull Requests (PRs).

**First Time Contributors ?**

Before adding your contribution, please take a moment to read through the following sections:

- The :ref:`Installation for contributors <installation_contributor>` help you to set up the developing environment and the pre-commit hooks.
- The section `Contributing process <#contributing-process>`__ provides you with a brief overview of the steps that each GPM-API developer must follow to contribute to the repository.
- The `Code review checklist <#code-review-checklist>`__ enable to speed up the code review process.
- The `Code of conduct <https://github.com/ghiggi/gpm_api/blob/main/CODE_OF_CONDUCT.md>`__ details the expected behavior of all contributors.

Initiating a discussion about your ideas or proposed implementations is a vital step before starting your contribution !
Engaging with the community early on can provide valuable insights, ensure alignment with the project's goals, and prevent potential overlap with existing work.
Here are some guidelines to facilitate this process:

1. Start with a conversation

   Before start coding, open a `GitHub Discussion <https://github.com/ghiggi/gpm_api/discussions>`__, a `GitHub Feature Request Issue <https://github.com/ghiggi/gpm_api/issues/new/choose>`__ or
   just start a discussion in the `GPM-API Slack Workspace <https://join.slack.com/t/gpmapi/shared_invite/zt-28vkxzjs1-~cIYci2o3G0qEEoQJVMQRg>`__.
   These channels of communication provides an opportunity to gather feedback, understand the project's current state, and improve your contributions.

2. Seek guidance and suggestions

   Utilize the community's expertise. Experienced contributors and maintainers can offer guidance, suggest best practices, and help you navigate any complexities you might encounter.

3. Collaborate on the approach

   Discussing your implementation strategy allows for a collaborative approach to problem-solving.
   It ensures that your contribution is in line with the project's design principles and technical direction.

By following these steps, you not only enhance the quality and relevance of your contribution but also become an integral part of the project's collaborative ecosystem.

If you have any questions, please do not hesitate to ask in the `GitHub Discussions <https://github.com/ghiggi/gpm_api/discussions>`__ or in the
`GPM-API Slack Workspace <https://join.slack.com/t/gpmapi/shared_invite/zt-28vkxzjs1-~cIYci2o3G0qEEoQJVMQRg>`__.


Issue Reporting
-----------------

To facilitate and enhance the issue reporting process, it is important to utilize the predefined GitHub Issue Templates.
These templates are designed to ensure you provide all the essential information in your report, allowing for a faster and more effective response from the maintainers.
You can access and use these templates by visiting the `GitHub Issue Templates page here <https://github.com/ghiggi/gpm_api/issues/new/choose>`__.

However, if you find that the existing templates don't quite match the specifics of the issue you're encountering, please feel free to suggest a new template.
Your feedback is invaluable in refining our processes and ensuring we address a broader spectrum of concerns.
To do this, simply create a general issue in the repository, clearly stating that you're requesting a new template and include detailed suggestions about what this new template should entail.
This proactive approach helps us continuously evolve and better serve the needs of the project and its contributors.


Contributing process
-----------------------

In this section we explain the steps that each developer must follow to contribute to the GPM-API repository.

The collaborative process is illustrated in the following diagram:

.. image:: /static/collaborative_process.png

and each step is detailed in the following subsections:

.. contents::
   :depth: 1
   :local:


1. Fork the repository
~~~~~~~~~~~~~~~~~~~~~~~

If you do not have a GitHub account yet, please create one `here <https://github.com/join>`__.
If you do not have yet Git installed on your computer, please install it following `these instructions <https://github.com/git-guides/install-git>`__.
Then, please follow the guidelines in the :ref:`Installation for contributors <installation_contributor>` section
to create the local copy of the GPM-API repository, set up the developing environment and the pre-commit hooks.

Once you have have a local copy of the GPM-API repository on your machine, you are ready to
contribute to the project!


2. Create a new branch
~~~~~~~~~~~~~~~~~~~~~~~

Each contribution should be made in a separate new branch of your forked repository.
Working on the main branch is reserved for `Core Contributors` only.
Core Contributors are developers that actively work and maintain the repository.
They are the only ones who accept Pull Requests and push commits directly to the GPM-API repository.

For more information on how to create and work with branches, see
`“Branches in a Nutshell” <https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>`__
in the Git documentation.

Please define the name of your branch based on the scope of the contribution. Try to strictly stick to the following guidelines:

-  If you fix a bug: ``bugfix-<some_key>-<word>``
-  If you improve the documentation: ``doc-<some_key>-<word>``
-  If you add a new feature: ``feature-<some_key>-<word>``
-  If you refactor some code: ``refactor-<some_key>-<word>``
-  If you optimize some code: ``optimize-<some_key>-<word>``

For example, if you are adding a new feature, you can create a new branch with the following command:

::

   git checkout -b add-feature-<name>


3. Add your code
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can start working on your changes.
You can add new features, fix bugs, improve the documentation, refactor the code, or optimize the code.
When you are working on your changes, please stick with the repository's coding style and documentation rules.

**Code Style**

We follow the `PEP 8 <https://pep8.org/>`__ style guide for python code.
Another relevant style guide can be found in the `The Hitchhiker's Guide to Python <https://docs.python-guide.org/writing/style/>`__.

To ensure a minimal style consistency, we use `black <https://black.readthedocs.io/en/stable/>`__ to auto-format the source code.
The `black` configuration used in the GPM-API project is
defined in the `pyproject.toml <https://github.com/ghiggi/gpm_api/blob/main/pyproject.toml>`__.


**Code Documentation**

Every module, function, or class must have a docstring that describes its purpose and how to use it.
The docstrings follows the conventions described in the `PEP 257 <https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings>`__
and the `Numpy's docstrings format <https://numpydoc.readthedocs.io/en/latest/format.html>`__.

Here is a summary of the most important rules:

-  Always use triple quotes for doctrings, even if it fits a single
   line.

-  For one-line docstring, end the phrase with a period.

-  Use imperative mood for all docstrings (``“””Return some value.”””``)
   rather than descriptive mood (``“””Returns some value.”””``).

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
         Lag-1 temporal autocorrelation coefficient.
       gamma_2 : float
         Lag-2 temporal autocorrelation coefficient.

       Returns
       -------
       out : float
         The adjusted lag-2 correlation coefficient.
       """


If you are using VS code, you can install the  `autoDocstring <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_
extension to automatically create such preformatted docstring.

You should configure VS code as follow:


.. image:: /static/vs_code_settings.png


The convention we adopt for our docstrings is the numpydoc string convention.

.. _code_quality_control:

4. Check code quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~


Pre-commit hooks are automated scripts that run during each commit to detect basic code quality issues.
If a hook identifies an issue (signified by the pre-commit script exiting with a non-zero status), it halts the commit process and displays the error messages.

Currently, GPM-API tests that the code to be committed complies with `black's  <https://github.com/psf/black>`__ format style,
the `ruff <https://github.com/charliermarsh/ruff>`__ linter and the `codespell <https://github.com/codespell-project/codespell>`__ spelling checker.

+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+
|  Tool                                                                                         | Aim                                                              | pre-commit | CI/CD |
+===============================================================================================+==================================================================+============+=======+
| `Black <https://black.readthedocs.io/en/stable/>`__                                           | Python code formatter                                            | 👍         | 👍    |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+
| `Ruff  <https://github.com/charliermarsh/ruff>`__                                             | Python linter                                                    | 👍         | 👍    |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+
| `Codespell  <https://github.com/codespell-project/codespell>`__                               | Spelling checker                                                 | 👍         | 👍    |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+

The versions of the software used in the pre-commit hooks is specified in the `.pre-commit-config.yaml <https://github.com/ghiggi/gpm_api/blob/main/.pre-commit-config.yaml>`__ file.
This file serves as a configuration guide, ensuring that the hooks are executed with the correct versions of each tool, thereby maintaining consistency and reliability in the code quality checks.

If a commit is blocked due to these checks, you can manually correct the issues by running locally the appropriate tool: ``black .`` for Black, ``ruff check .`` for Ruff, or ``codespell`` for Codespell.
Alternatively, you can use the ``pre-commit run --all-files`` command to attempt automatic corrections of all formatting errors across all files.

The Continuous Integration (CI) tools integrated within GitHub employ the same pre-commit hooks to consistently uphold code quality for every Pull Request.

In addition to the pre-commit hooks, the Continuous Integration (CI) setup on GitHub incorporates an extended suite of tools.
These tools, which are not installable on a local setup, perform advanced code quality analyses and reviews after each update to a Pull Request.

Refer to the table below for a comprehensive summary of all CI tools employed to assess the code quality of a Pull Request.

+----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| Tool                                               | Aim                                                                                                                                 |
+====================================================+=====================================================================================================================================+
| `pre-commit.ci <https://pre-commit.ci/>`__         | Run pre-commit (as defined in `.pre-commit-config.yaml <https://github.com/ghiggi/gpm_api/blob/main/.pre-commit-config.yaml>`__)    |
+----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| `CodeBeat <https://codebeat.co/>`__                | Automated code review and analysis tools                                                                                            |
+----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| `CodeScene <https://codescene.com/>`__             | Automated code review and analysis tools                                                                                            |
+----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| `CodeFactor <https://www.codefactor.io/>`__        | Automated code review and analysis tools                                                                                            |
+----------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+


5. Check code functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every code change must be tested !

GPM-API tests are written using the third-party `pytest <https://docs.pytest.org>`_ package.

The tests are organized within the ``/gpm/tests`` directory and are structured to comprehensively assess various aspects of the code.

These tests are integral to the development process and are automatically triggered on GitHub upon any new commits or updates to a Pull Request.
The Continuous Integration (CI) on GitHub runs tests and analyzes code coverage using multiple versions of Python,
multiple operating systems, and multiple versions of dependency libraries. This is done to ensure that the code works in a variety of environments.

The following tools are used:

+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+
|  Tool                                                                                         | Aim                                                              |
+===============================================================================================+==================================================================+
| `Pytest  <https://docs.pytest.org>`__                                                         | Execute unit tests and functional tests                          |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+
| `Coverage <https://coverage.readthedocs.io/>`__                                               | Measure the code coverage of the project's unit tests            |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+
| `CodeCov    <https://about.codecov.io/>`__                                                    | Uses Coverage to track and analyze code coverage over time.      |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+
| `Coveralls    <https://coveralls.io/>`__                                                      | Uses Coverage to track and analyze code coverage over time.      |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+


For contributors interested in running the tests locally:

1. Ensure you have the :ref:`development environment <installation_contributor>` correctly set up. Make sure you also downloaded the additional test data.
2. Navigate to the GPM-API root directory.
3. Execute the following command to run the entire test suite:

.. code-block:: bash

	pytest

For more focused testing or during specific feature development, you may run subsets of tests.
This can be done by specifying either a sub-directory or a particular test module.

Run tests in a specific sub-directory:

.. code-block:: bash

    pytest gpm/tests/<test_subdirectory>/

Run a particular test module:

.. code-block:: bash

    pytest gpm/tests/<test_subdirectory>/test_<module_name>.py

These options provide flexibility, allowing you to efficiently target and validate specific components of the GPM-API software.

.. note::
   Each test module must be prefixed with ``test_`` to be recognized and selected by pytest.
   This naming pattern is a standard convention in pytest and helps in the automatic discovery of test files.

6. Push your changes
~~~~~~~~~~~~~~~~~~~~~~

Once you have finished working on your changes, you can push your local changes to your fork repository.

During this process, pre-commit hooks will be run. Your commit will be
allowed only if quality requirements are fulfilled.

If you encounter errors, you can attempt to fix the formatting errors with the following command:

::

   pre-commit run --all-files


7. Create a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Once your code has been uploaded into your GPM-API fork, you can create a GitHub Pull Request (PR) to the GPM-API main branch.

Recommendation for the Pull Requests:

-  Please fill it out accurately the Pull Request template.
-  It is perfectly fine to make many small commits as you work on a Pull Request. GitHub will automatically squash all the commits before merging the Pull Request.
-  If adding a new feature:

   -  Provide a convincing reason to add the new feature. Ideally, propose your idea through a `Feature Request Issue <https://github.com/ghiggi/gpm_api/issues/new/choose>`__ and obtain approval before starting work on it. Alternatively, you can present your ideas in the `GitHub Discussions <https://github.com/ghiggi/gpm_api/discussions>`__ or in the `GPM-API Slack Workspace <https://join.slack.com/t/gpmapi/shared_invite/zt-28vkxzjs1-~cIYci2o3G0qEEoQJVMQRg>`__.
   -  Implement unit tests to verify the functionality of the new feature. This ensures that your addition works as intended and maintains the quality of the codebase.

-  If fixing bug:

   -  Provide a comprehensive description of the bug within your Pull Request. This aids reviewers in understanding the issue and the impact of your fix.
   -  If your Pull Request addresses a specific issue, add ``(fix #xxxx)`` in your PR title to link the PR to the issue and enhance the clarity of release logs. For example, the title of a PR fixing issue ``#3899`` would be ``<your PR title> (fix #3899)``.
   -  If applicable, ensure that your fix includes appropriate tests. Adding tests for your bug fix helps prevent future regressions and maintains the stability of the software.


Contributing to test data
---------------------------

If your changes modify the structure of the GPM-API ``xarray.Dataset``,
you will likely need to update the test data in the ``gpm/tests/data/`` directory.

This directory functions as a separate git directory, with its own history and remote repository.
To update the test data, you need to first ask the maintainers to become a contributor on the
`gpm_api_test_data <https://github.com/ghiggi/gpm_api_test_data>`_ repository.
Then you can create a branch with the new test data and open a Pull Request which updates the GPM-API test data.

The GPM-API repository keeps track of the currently checked-out commit of the test-data repository.
When the checked-out commit changes, you can register this change in the GPM-API repository by running

.. code-block:: bash

    git add gpm/tests/data

and committing.


To submit your contribution that involves modifying test data, please follow this procedure.

(A: GPM-API repository, B: test-data repository)

1. Make a *feature branch* for B

.. code-block:: bash

    cd gpm/tests/data
    # Inside this directory, following git commands will apply to B
    git checkout -b my-feature-branch
    ...

2. Have A point to the *feature branch* of B

.. code-block:: bash

    # From the root of the GPM-API repository
    git add gpm/tests/data
    git commit
    ...

3. Make two PRs (for A and B) and get both accepted
4. Have the B’s PR merged into the B's *main branch*
5. Update A to point to B’s updated *main branch* (instead of the old *feature branch*)

.. code-block:: bash

    # Checkout the main branch of the test-data repository
    cd gpm/tests/data
    git checkout main
    git pull

    cd ../../..
    # From the root of the GPM-API repository, update the reference
    git add gpm/tests/data

6. Have A’s PR merged


Code review checklist
---------------------

-  Once your Pull Request is ready, ask the maintainers to review your code.
-  When you are done with the changes suggested by the reviewers, do another  self review of the code and write a comment to notify the reviewer,
   that the Pull Request is ready for another iteration.
-  Resolve all the review comments, making sure they are all addressed before another review iteration.
-  If you are not going to follow a code review recommendations, please add a comment explaining why you think the reviewer suggestion is not relevant.
-  Avoid writing comment like “done” of “fixed” on each code review comment.
   Reviewers assume you will do all suggested changes, unless you have a reason not to do some of them.


Credits
-------

Thank you to all the people who have already contributed to GPM-API repository!

If you have contributed code or documentation to GPM-API, add your name to the `AUTHORS.md <https://github.com/ghiggi/gpm_api/blob/main/AUTHORS.md>`__ file.
