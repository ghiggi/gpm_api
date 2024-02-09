=========================
Installation
=========================


We define here two types of installation:

- `Installation for standard users`_ : for users who want to process data.

- `Installation for contributors`_: for contributors who want to enrich the project (eg. add a new features).

We recommend users and contributors to first set up a virtual environment where to install GPM-API.


.. _virtual_environment:

Virtual environment creation
===============================

While not mandatory, utilizing a virtual environment when installing GPM-API is recommended.

Using a virtual environment for installing packages provides isolation of dependencies,
easier package management, easier maintenance, improved security, and improved development workflow.

Here below we provide two options to set up a virtual environment,
using `venv <https://docs.python.org/3/library/venv.html>`__
or `conda <https://docs.conda.io/en/latest/>`__ (recommended).

**With conda:**

* Install `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
  or `anaconda <https://docs.anaconda.com/anaconda/install/>`__
  if you don't have it already installed.

* Create the `gpm-api-py311` (or any other custom name) conda environment:

.. code-block:: bash

	conda create --name gpm-api-py311 python=3.11 --no-default-packages

* Activate the `gpm-api-py311` conda environment:

.. code-block:: bash

	conda activate gpm-api-py311


**With venv:**

* Windows: Create a virtual environment with venv:

.. code-block:: bash

   python -m venv gpm-api-pyXXX
   cd gpm-api-pyXXX/Scripts
   activate


* Mac/Linux: Create a virtual environment with venv:

.. code-block:: bash

   virtualenv -p python3 gpm-api-pyXXX
   source gpm-api-pyXXX/bin/activate

.. _installation_standard:

Installation for standard users
==================================

The latest GPM-API stable version is available
on the `Python Packaging Index (PyPI) <https://pypi.org/project/gpm-api/>`__
and on the `conda-forge channel <https://anaconda.org/conda-forge/gpm-api>`__.

Therefore you can either install the package with pip or conda.
Installation with conda is recommended, as GPM-API depends on `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`__
and `GEOS <https://libgeos.org/>`_ libraries, which can be difficult to install with pip.

Please install the package in the virtual environment you created before !

**With conda:**

.. code-block:: bash

   conda install -c conda-forge gpm-api


.. note::
   In alternative to conda, if you are looking for a lightweight package manager you could use `micromamba <https://micromamba.readthedocs.io/en/latest/>`__.

**With pip:**

On Linux, prior to the installation of GPM-API, you can install GEOS using your package manager (e.g. ``apt install libgeos-dev``).
Then, install GPM-API with:

.. code-block:: bash

   pip install gpm-api


.. _installation_contributor:

Installation for contributors
================================

The latest GPM-API version is available on the GitHub repository `gpm_api <https://github.com/ghiggi/gpm_api>`_.
You can install the package in editable mode, so that you can modify the code and see the changes immediately.
Here below we provide the steps to install the package in editable mode.

Clone the repository from GitHub
......................................

According to the :ref:`contributors guidelines <contributor_guidelines>`,
you should first
`create a fork into your personal GitHub account <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`__.

Then create a local copy of the repository you forked with:

.. code-block:: bash

   git clone https://github.com/<your-account>/gpm_api.git
   cd gpm_api

Create the development environment
......................................

We recommend to create a dedicated conda environment for development purposes.
You can create a conda environment (i.e. with python 3.11) with:

.. code-block:: bash

	conda create --name gpm-api-dev-py311 python=3.11 --no-default-packages
	conda activate gpm-api-dev-py311

Install the GPM-API package dependencies
............................................

.. code-block:: bash

	conda install --only-deps gpm-api


Install the GPM-API package in editable mode
................................................

Install the GPM-API package in editable mode by executing the following command in the GPM-API repository's root:

.. code-block:: bash

	pip install -e ".[dev]"


Install pre-commit code quality checks
..............................................

Install the pre-commit hook by executing the following command in the GPM-API repository's root:

.. code-block:: bash

   pre-commit install


Pre-commit hooks are automated scripts that run during each commit to detect basic code quality issues.
If a hook identifies an issue (signified by the pre-commit script exiting with a non-zero status), it halts the commit process and displays the error messages.

.. note::
	The versions of the software used in the pre-commit hooks is specified in the `.pre-commit-config.yaml <https://github.com/ghiggi/gpm_api/blob/main/.pre-commit-config.yaml>`__ file. This file serves as a configuration guide, ensuring that the hooks are executed with the correct versions of each tool, thereby maintaining consistency and reliability in the code quality checks.

Further details about pre-commit hooks can be found in the Contributors Guidelines, specifically in the provided in the :ref:`Code quality control <code_quality_control>` section.


Run GPM-API on Jupyter Notebooks
=====================================

If you want to run GPM-API on a `Jupyter Notebook <https://jupyter.org/>`__,
you have to take care to set up the IPython kernel environment where GPM-API is installed.

For example, if your conda/virtual environment is named `gpm-api-dev`, run:

.. code-block:: bash

   python -m ipykernel install --user --name=gpm-api-dev

When you will use the Jupyter Notebook, by clicking on `Kernel` and then `Change Kernel`, you will be able to select the `gpm-api-dev` kernel.
