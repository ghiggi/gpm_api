========================
Maintainers guidelines
========================


The section is dedicated to the GPM-API core developers (maintainers).


List of the core contributors
=================================

* Current Release Manager : Ghiggi Gionata
* Testing Team : Ghiggi Gionata



Versions guidelines
========================

GPM-API uses  `Semantic <https://semver.org/>`_ Versioning. Each release is associated with a git tag of the form X.Y.Z.

Given a version number in the MAJOR.MINOR.PATCH (eg., X.Y.Z) format, here are the differences in these terms:

- MAJOR version - make breaking/incompatible API changes
- MINOR version - add functionality in a backwards compatible manner.
- PATCH version - make backwards compatible bug fixes


Breaking vs. non-breaking changes
-----------------------------------

Since GPM-API is used by a broad ecosystem of both API consumers and implementers, it needs a strict definition of what changes are “non-breaking” and are therefore allowed in MINOR and PATCH releases.

In the GPM-API spec, a breaking change is any change that requires either consumers or implementers to modify their code for it to continue to function correctly.

Examples of breaking changes include:

- Adding new functionalities to the GPM-API that affect the behavior of the software directly.


Examples of non-breaking changes include :

- Fix a bug.
- Adding new functionalities to GPM-API that don’t affect the behavior of the API directly.
- Updating the documentation.
- Internal function refactoring that doesn’t affect the behavior of the software directly.




One implication of this policy is that clients should be prepared to ignore the presence of unexpected fields in responses and unexpected values for enums. This is necessary to preserve compatibility between PATCH versions within the same MINOR version range, since optional fields and enum values can be added as non-breaking changes.


Ongoing version support
-----------------------------------

GPM-API major releases aims to move the community forward, focusing on specifications stabilization and major feature additions, rather than backwards-compatibility. GPM-API minor releases will be backwards compatible. We strongly recommend adopting the latest release of GPM-API into production within 6 months for major releases, and 4 months for minor releases.

The `LTE <https://www.epfl.ch/labs/lte/>`_ does not guarantee any period of support or maintenance. Recommended versions are supported and maintained by the `LTE <https://www.epfl.ch/labs/lte/>`_  and our community – we provide updated guidance and documentation, track issues, and provide bug fixes and critical updates in the form of hotfixes for these versions. Releases that are 2 years or older may be considered as deprecated.

Refer to the list of Recommended Releases to see current releases and more details.




Documentation pipeline
========================

GPM-API’s documentation is built using Sphinx. All documentation lives in the ``docs/`` directory of the project repository.


Manual documentation creation
-----------------------------

After editing the source files there the documentation can be generated locally:


.. code-block:: bash

	sphinx-build -b html source build


The output of the previous command should be checked for warnings and errors. If the code is changed (new functions or classes) then the GPM-API documentation files should be regenerated before running the above command:

.. code-block:: bash

	sphinx-apidoc -f -o source/api .. ../setup.py


Automatic (Github) documentation creation
------------------------------------------


One webhook is defined in the repository to trigger the publication process to readthedoc.io.

This webhook is linked to the GPM-API core developer.

.. image:: /static/documentation_pipepline.png

Ghiggi Gionata owns the `ReadTheDoc <https://readthedocs.org/>`__ account.


Package releases pipeline
============================

One  `GitHub Action <https://github.com/ghiggi/gpm_api/actions>`_ is defined to trigger the packaging and the upload on `pypi.org <https://pypi.org/project/gpm-api/>`_.

.. image:: /static/package_pipepline.png

The `PyPi <https://pypi.org/>`__ project is shared beween the core contributors.



Reviewing process
============================


The main branch is protected and requires at least one review before merging.

The review process is the following:

#. A PR is opened by a contributor
#. The CI pipeline is triggered and the status of the tests is reported in the PR.
#. A core contributor reviews the PR and request changes if needed.
#. The contributor updates the PR according to the review.
#. The core contributor reviews the PR again and merge it if the changes are ok.



Continuous intergration (CI) testing tools
===========================================

Currently, on each Pull Request, GitHub Actions are configured as follow:


+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
|  Tools                                                                                             | Aim                                                              | Project page                                                                                 | Python version                            |
+====================================================================================================+==================================================================+==============================================================================================+===========================================+
| `Pytest  <https://docs.pytest.org>`__                                                              | Execute unit tests and functional tests                          |                                                                                              |                                           |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
| `Black <https://black.readthedocs.io/en/stable/>`__                                                | Python code formatter                                            |                                                                                              | No python version (Black version 22.8.0)  |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
| `Ruff  <https://github.com/charliermarsh/ruff>`__                                                  | Python linter                                                    |                                                                                              | (Ruff version 0.0.2570)                   |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
| `pre-commit.ci   <https://pre-commit.ci/>`__                                                       | Run pre-commit as defined in pre-commit-config.yaml              |                                                                                              |                                           |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
| Coverage                                                                                           | Measure the code coverage of the project's unit tests            |                                                                                              | all versions according to GitHub workflow |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
| `CodeCov    <https://about.codecov.io/>`__                                                         | Uses the "coverage" package to generate a code coverage report.  | `GPM-API  <https://app.codecov.io/gh/ghiggi/gpm_api>`__                                      | all versions according to GitHub workflow |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
| `Coveralls    <https://coveralls.io/>`__                                                           | Uses the "coverage" to track the quality of your code over time. | `GPM-API  <https://coveralls.io/github/ghiggi/gpm_api>`__                                    | all versions according to GitHub workflow |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
| `CodeBeat      <https://codebeat.co/>`__                                                           | Automated code review and analysis tools                         | `GPM-API <https://codebeat.co/projects/github-com-ghiggi/gpm_api>`__                         | all versions according to GitHub workflow |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
| `CodeScene <https://codescene.com/>`__                                                             | Automated code review and analysis tools                         |                                                                                              | all versions according to GitHub workflow |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
| `CodeFactor <https://www.codefactor.io/>`__                                                        | Automated code review and analysis tools                         | `GPM-API <https://www.codefactor.io/repository/github/ghiggi/gpm_api>`__                     | all versions according to GitHub workflow |
+----------------------------------------------------------------------------------------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------+-------------------------------------------+
