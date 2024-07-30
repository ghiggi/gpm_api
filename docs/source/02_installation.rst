=========================
Installation
=========================


We define here two types of installation:

- `Installation for standard users`_: for users who want to process data.

- `Installation for contributors`_: for contributors who want to enrich the project (eg. add a new features).

We recommend users and contributors first set up a virtual environment to install GPM-API.


.. _virtual_environment:

Virtual environment creation
===============================

While not mandatory, utilizing a virtual environment when installing GPM-API is recommended.

Using a virtual environment for installing packages provides isolation of dependencies,
easier package management, easier maintenance, improved security, and improved development workflow.

We provide two options to set up a virtual environment: using `venv <https://docs.python.org/3/library/venv.html>`__
or `conda <https://docs.conda.io/en/latest/>`__ (recommended).

**With conda:**

* Install `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
  or `anaconda <https://docs.anaconda.com/anaconda/install/>`__
  if you don't have it already installed.

* Create the *gpm-api-py311* (or any other custom name) conda environment:

.. code-block:: bash

	conda create --name gpm-api-py311 python=3.11 --no-default-packages

* Activate the *gpm-api-py311* conda environment:

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

Please install the package in the virtual environment you created before!

**With conda:**

.. code-block:: bash

   conda install -c conda-forge gpm-api

.. note::
   In an alternative to conda, if you are looking for a lightweight package manager you could use `micromamba <https://micromamba.readthedocs.io/en/latest/>`__.

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
The following steps guides to the package installation in editable mode.

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

We recommend creating a dedicated conda environment for development purposes.
You can create a conda environment (i.e. with python 3.11) with:

.. code-block:: bash

	conda create --name gpm-api-dev-py311 python=3.11 --no-default-packages
	conda activate gpm-api-dev-py311

Install the package dependencies
............................................

.. code-block:: bash

	conda install --only-deps gpm-api


Install the package in editable mode
................................................

Install the GPM-API package in editable mode by executing the following command in the GPM-API repository's root:

.. code-block:: bash

	pip install -e ".[dev]"


Install code quality checks
..............................................

Install the pre-commit hook by executing the following command in the GPM-API repository's root:

.. code-block:: bash

   pre-commit install


Pre-commit hooks are automated scripts that run during each commit to detect basic code quality issues.
If a hook identifies an issue (signified by the pre-commit script exiting with a non-zero status), it halts the commit process and displays the error messages.

.. note::
	The versions of the software used in the pre-commit hooks are specified in the `.pre-commit-config.yaml <https://github.com/ghiggi/gpm_api/blob/main/.pre-commit-config.yaml>`__ file. This file serves as a configuration guide, ensuring that the hooks are executed with the correct versions of each tool, thereby maintaining consistency and reliability in the code quality checks.

Further details about pre-commit hooks can be found in the Contributors Guidelines, specifically in the provided in the :ref:`Code quality control <code_quality_control>` section.

Download the test data
......................

Some of GPM-API's tests require additional data to be executed.
If you want to be able to run the full GPM-API test suite on your local machine, you also need to download such additional test data.
First, ensure you have your GitHub account ssh keys `set up correctly <https://docs.github.com/articles/adding-a-new-ssh-key-to-your-github-account>`_.
Then, from the within the ``gpm_api`` directory, run:

.. code-block:: bash

   git submodule update --init --recursive


Optional dependencies
=======================

Specific functionality in GPM-API may require additional optional dependencies.
To unlock the full functionalities offered by GPM-API, it is recommended to install also the packages detailed here below.

The following bash code allow to install all optional dependencies:

.. code-block:: bash

   conda install -c conda-forge jupyter spyder flox numbagg bottleneck opt-einsum python-graphviz ximage pyresample shapely geopandas xvec xoak scikit-learn pyvista trame trame-vuetify trame-vtk polars pyarrow xradar wradlib pyart


IDE Tools
..............

For an improved development experience, consider installing the intuitive `Jupyter <https://jupyter.org/>`_ and
`Spyder <https://www.spyder-ide.org/>`_ Python Integrated Development Environments (IDEs):

.. code-block:: bash

   conda install -c conda-forge jupyter spyder

Speed Up Xarray Computations
..........................................

To speed up arrays computations with xarray, install
`flox <https://flox.readthedocs.io/en/latest/>`_,
`numbagg <https://github.com/numbagg/numbagg>`_,
`bottleneck <https://bottleneck.readthedocs.io/en/latest/intro.html>`_ and
`opt-einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_:

.. code-block:: bash

   conda install -c conda-forge flox numbagg bottleneck opt-einsum

Dask Operations
......................

To visualize `Dask Task Graphs  <https://docs.dask.org/en/stable/10-minutes-to-dask.html>`_ and monitor
computations through the `Dask Dashboard <https://docs.dask.org/en/stable/dashboard.html>`_, please install:

.. code-block:: bash

   conda install -c conda-forge python-graphviz bokeh

Image Analysis
....................

To perform advanced image/volume manipulations, install `ximage <https://x-image.readthedocs.io/en/latest/index.html>`_:

.. code-block:: bash

   conda install -c conda-forge ximage

Geospatial Manipulation
.............................

To perform advanced geospatial manipulations, we recommend to install
`shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_,
`geopandas  <https://geopandas.org/en/stable/>`_,
`xvec <https://xvec.readthedocs.io/en/stable/>`_ and
`pyresample <https://pyresample.readthedocs.io/en/latest/>`_:

.. code-block:: bash

   conda install -c conda-forge shapely geopandas xvec pyresample

Cross-Section and Trajectories
.................................

To be able to extract radar cross-sections with gpm-api, install:

.. code-block:: bash

   conda install -c conda-forge xoak scikit-learn

3D Radar Visualization
...........................

To create interactive 3D radar visualization with gpm-api, please install
`pyvista  <https://docs.pyvista.org/>`_ and the associated dependencies:

.. code-block:: bash

      ``conda install -c conda-forge pyvista trame trame-vuetify trame-vtk``

Geographic Bucket Archives
............................

To create and analyse efficiently GPM satellite bucket archives with gpm-api, install
`polars <https://pola.rs/>`_ and
`pyarrow <https://arrow.apache.org/docs/python/index.html>`_:

.. code-block:: bash

      conda install -c conda-forge polars pyarrow

Spaceborne/Ground Radar Analysis
...................................

To perform spaceborne-ground radar calibration and validation, install
`xradar < https://docs.openradarscience.org/projects/xradar/en/stable/>`_,
`wradlib <https://docs.wradlib.org/en/latest/>`_ and
`pyart <https://arm-doe.github.io/pyart/>`_:

.. code-block:: bash

   conda install -c conda-forge xradar wradlib pyart


Run GPM-API on Jupyter Notebooks
=====================================

If you want to run GPM-API on a `Jupyter Notebook <https://jupyter.org/>`__,
you have to take care to set up the IPython kernel environment where GPM-API is installed.

For example, if your conda/virtual environment is named ``gpm-api-dev``, run:

.. code-block:: bash

   python -m ipykernel install --user --name=gpm-api-dev

When you will use the Jupyter Notebook, by clicking on ``Kernel`` and then ``Change Kernel``, you will be able to select the ``gpm-api-dev`` kernel.
