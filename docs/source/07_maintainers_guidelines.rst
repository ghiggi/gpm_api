========================
Maintainers Guidelines
========================


The section is dedicated to the GPM-API core developers (maintainers).


Core Contributors
====================

* Current Release Manager : Ghiggi Gionata
* Testing Team : Ghiggi Gionata, Son Pham-Ba


Versions Guidelines
========================

GPM-API uses  `Semantic <https://semver.org/>`_ Versioning. Each release is associated with a git tag of the form X.Y.Z.

Given a version number in the MAJOR.MINOR.PATCH (eg., X.Y.Z) format, here are the differences in these terms:

- MAJOR version - make breaking/incompatible API changes
- MINOR version - add functionality in a backwards compatible manner.
- PATCH version - make backwards compatible bug fixes


Breaking vs. non-breaking changes
-----------------------------------

Since GPM-API is used by a broad ecosystem of both API consumers and implementers,
it needs a strict definition of what changes are “non-breaking” and are therefore allowed in MINOR and PATCH releases.

In the GPM-API spec, a breaking change is any change that requires either consumers or implementers to modify their code for it to continue to function correctly.

Examples of breaking changes include:

- Adding new functionalities to the GPM-API that affect the behavior of the software directly.


Examples of non-breaking changes include :

- Fix a bug.
- Adding new functionalities to GPM-API that don't affect the behavior of the API directly.
- Updating the documentation.
- Internal function refactoring that doesn't affect the behavior of the software directly.


Ongoing version support
-----------------------------------

GPM-API major releases aims to move the community forward, focusing on specifications stabilization and major feature additions, rather than backwards-compatibility.
GPM-API minor releases will be backwards compatible.
We strongly recommend adopting the latest release of GPM-API into production within 6 months for major releases, and 4 months for minor releases.

The maintainers do their best but does not guarantee any period of support or maintenance.

Releases that are 2 years or older may be considered as deprecated.


Documentation
========================

GPM-API's documentation is built using the powerful `Sphinx <https://www.sphinx-doc.org/en/master/>`_ framework,
styled with `Book Theme <https://sphinx-book-theme.readthedocs.io/en/stable/index.html>`_.

All documentation source files are neatly organized in the ``docs/`` directory within the project's repository.


Documentation Generation
--------------------------

To build the documentation locally, follow the next three steps.

1. Set up the python environment for building the documentation

	The python packages required to build the documentation are listed in the
	`environment.yaml <https://github.com/ghiggi/gpm_api/blob/main/docs/environment.yaml>`_ file.

	For an efficient setup, it is recommended to create a dedicated virtual environment.
	Navigate to the ``docs/`` directory and execute the following command.
	This will create a new environment and install the required packages:

	.. code-block:: bash

		conda create -f environment.yaml

2. Activate the virtual environment

	Once the environment is ready, activate it using:

	.. code-block:: bash

	   	conda activate build-doc-gpm-api


3. Generate the documentation

	With the environment set and activated, you're ready to generate the documentation.
	Execute:

	.. code-block:: bash

		make clean html

	This command will build the HTML version of the documentation.
	It first cleans previous builds (``make clean``) and then generates fresh documentation (``html``).

	.. note:: It's important to review the output of the command. Look out for warnings or errors and address them to ensure the documentation is accurate and complete.

By following these steps, you should have a local version of the GPM-API documentation
in the ``docs/build/html/`` directory, ready for review or deployment!

Documentation Deployment
--------------------------

A webhook is defined in the GitHub repository to trigger automatically the publication process to `ReadTheDocs <https://about.readthedocs.com/?ref=readthedocs.com>`__
after each Pull Request.

This webhook is linked to the GPM-API core developer.

.. image:: /static/documentation_release.png

Ghiggi Gionata owns the `ReadTheDocs <https://readthedocs.org/>`__ account.


Package Release
=================

A `GitHub Action <https://github.com/ghiggi/gpm_api/actions>`_ is configured to automate the packaging and uploading process to `PyPI <https://pypi.org/project/gpm-api/>`_.
This action, detailed `here <https://github.com/ghiggi/gpm_api/blob/main/.github/workflows/release_to_pypi.yaml>`_, triggers the packaging workflow depicted in the following image:

.. image:: /static/package_release.png

Upon the release of the package on PyPI, a conda-forge bot attempts to automatically update the `conda-forge recipe <https://github.com/conda-forge/gpm-api-feedstock/>`__.
Once the conda-forge recipe is updated, a new conda-forge package is released.

The PyPI project and the conda-forge recipes are collaboratively maintained by core contributors of the project.


Release Process
----------------

Before releasing a new version, the ``CHANGELOG.md`` file should be updated.

Execute ``git tag`` to identify the last version and determine the new ``X.Y.Z`` version number.
Then, run ``make changelog X.Y.Z`` to update the ``CHANGELOG.md`` file with the list of issues and pull requests that have been closed since the last release.
Manually edit the ``CHANGELOG.md`` if necessary.

Then, commit the new ``CHANGELOG.md`` file.

.. code-block:: bash

    git add CHANGELOG.md
    git commit -m "update CHANGELOG.md for version X.Y.Z"
    git push

Finally, create a new tag to trigger the release process.

.. code-block:: bash

    git tag -a vX.Y.Z -m "Version X.Y.Z"
    git push --tags

On GitHub, edit the release description to add the list of changes from the ``CHANGELOG.md`` file.


Reviewing Process
============================


The main branch is protected and requires at least one review before merging.

The review process is the following:

#. A PR is opened by a contributor
#. The CI pipeline is triggered and the status of the tests is reported in the PR.
#. A core contributor reviews the PR and request changes if needed.
#. The contributor updates the PR according to the review.
#. The core contributor reviews the PR again and merge it if the changes are ok.



Continuous Integration
==============================

Continuous Integration (CI) is a crucial practice in modern software development, ensuring that code changes are regularly integrated into the main codebase.
With CI, each commit or pull request triggers an automated process that verifies the integrity of the codebase, runs tests,
and performs various checks to catch issues early in the development lifecycle.

The table below summarizes the software tools utilized in our CI pipeline, describes their respective aims and project pages.

+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
|  Tools                                                                                             | Aim                                                              | Project page                                                                                 |
+====================================================================================================+==================================================================+==============================================================================================+
| `Pytest  <https://docs.pytest.org>`__                                                              | Execute unit tests and functional tests                          |                                                                                              |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| `Black <https://black.readthedocs.io/en/stable/>`__                                                | Python code formatter                                            |                                                                                              |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| `Ruff  <https://github.com/charliermarsh/ruff>`__                                                  | Python linter                                                    |                                                                                              |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| `pre-commit.ci   <https://pre-commit.ci/>`__                                                       | Run pre-commit as defined in `.pre-commit-config.yaml <https://github.com/ghiggi/gpm_api/blob/main/.pre-commit-config.yaml>`__                                  |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| `Coverage   <https://coverage.readthedocs.io/>`__                                                  | Measure the code coverage of the project's unit tests            |                                                                                              |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| `CodeCov    <https://about.codecov.io/>`__                                                         | Uses the "coverage" package to generate a code coverage report.  | `GPM-API  <https://app.codecov.io/gh/ghiggi/gpm_api>`__                                      |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| `Coveralls    <https://coveralls.io/>`__                                                           | Uses the "coverage" to track the quality of your code over time. | `GPM-API  <https://coveralls.io/github/ghiggi/gpm_api>`__                                    |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| `CodeScene <https://codescene.com/>`__                                                             | Automated code review and analysis tools                         | `GPM-API  <https://codescene.io/projects/36767/>`__                                          |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| `CodeFactor <https://www.codefactor.io/>`__                                                        | Automated code review and analysis tools                         | `GPM-API <https://www.codefactor.io/repository/github/ghiggi/gpm_api>`__                     |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| `Codacy      <https://www.codacy.com/>`__                                                          | Automated code review and analysis tools                         | `GPM-API <https://app.codacy.com/gh/ghiggi/gpm_api/dashboard>`__                             |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+


Earthdata Cloud integration test
--------------------------------

``gpm/tests/test_io/test_download_earthdata.py`` downloads a real IMERG-FR V07 granule
for 2017-01-01 20:00 to 20:30 into a temporary GPM archive. It checks the expected filename,
opens the HDF5 ``Grid`` group, and reads a grid cell from each variable.
The test is marked ``integration`` and runs when both Earthdata environment variables are set
or an Earthaccess credential file exists. This includes the local ``.netrc`` created by
``gpm.define_configs``. Earthaccess honors ``NETRC`` for a custom path and uses ``_netrc`` on Windows.
The test skips only when neither credential source is available.
When credentials are provided, authentication, network, or download failures fail the test.

Configure the GitHub repository secrets as follows:

#. Open the repository's **Settings**.
#. Select **Secrets and variables**, then **Actions**, and open the **Secrets** tab.
#. Click **New repository secret** and add ``EARTHDATA_USERNAME`` with your Earthdata Login username.
#. Add a second repository secret named ``EARTHDATA_PASSWORD`` with your Earthdata Login password.

The Linux and Windows test workflows expose these secrets only to their pytest steps:

.. code-block:: yaml

    env:
      EARTHDATA_USERNAME: ${{ secrets.EARTHDATA_USERNAME }}
      EARTHDATA_PASSWORD: ${{ secrets.EARTHDATA_PASSWORD }}

The normal CI test run includes the live download when both secrets are available.
Fork pull requests and Dependabot runs do not receive repository secrets, so the test skips there.
See `GitHub's documentation on Actions secrets
<https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets>`__.

To run the live test locally with an existing ``.netrc``:

.. code-block:: bash

    pytest gpm/tests/test_io/test_download_earthdata.py -m integration

Alternatively, supply credentials through environment variables:

.. code-block:: bash

    export EARTHDATA_USERNAME="your_username"
    export EARTHDATA_PASSWORD="your_password"
    pytest gpm/tests/test_io/test_download_earthdata.py -m integration

To exclude live integration tests, use ``pytest -m "not integration"``.


PPS integration test
----------------------

``gpm/tests/test_io/test_download_pps.py`` downloads the same IMERG-FR V07 granule
from PPS and verifies its archive path and readable ``Grid`` data.
Credentials are read from ``PPS_USERNAME`` and ``PPS_PASSWORD``, falling back to
``gpm.config.get("username_pps")`` and ``gpm.config.get("password_pps")``.
The test skips if either credential is unavailable.

For local use, configure PPS credentials with ``gpm.define_configs`` or export them:

.. code-block:: bash

    export PPS_USERNAME="your_pps_username"
    export PPS_PASSWORD="your_pps_password"
    pytest gpm/tests/test_io/test_download_pps.py -m integration

For CI, add the repository secrets ``PPS_USERNAME`` and ``PPS_PASSWORD`` under
**Settings**, **Secrets and variables**, **Actions**.
Both Linux and Windows test workflows export these secrets to their pytest steps.
Use the credentials for your registered PPS account; the usual PPS password is the
same email address as the username.
