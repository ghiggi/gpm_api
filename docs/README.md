# GPM-API documentation

GPM-API’s documentation is built using the powerful [Sphinx](https://www.sphinx-doc.org/en/master/) framework,
styled with [Book Theme](https://sphinx-book-theme.readthedocs.io/en/stable/index.html).

All documentation source files are neatly organized in the `docs/` directory within the project's repository.

-----------------------------------------------------------------

### Build the documentation

To build the documentation locally, follow the next three steps.

**1. Set up the python environment for building the documentation**

The python packages required to build the documentation are listed in the [environment.yaml](https://github.com/ghiggi/gpm_api/blob/main/docs/environment.yaml) file.

For an efficient setup, we recommend creating a dedicated virtual environment.
Navigate to the `docs/` directory and execute the following command.
This will create a new environment and install the required packages:

```
conda create -f environment.yaml
```

**2. Activate the virtual environment**

Once the environment is ready, activate it using:

```
conda activate build-doc-gpm-api
```

**3. Generate the documentation**

With the environment set and activated, you're ready to generate the documentation.
Execute:

```
make clean html
```

This command will build the HTML version of the documentation.
It first cleans previous builds (`make clean`) and then generates fresh documentation (`html`).

**Note**: It's important to review the output of the command. Look out for warnings or errors and address them to ensure the documentation is accurate and complete.

By following these steps, you should have a local version of the GPM-API documentation in the ``docs/build/html/`` directory,
ready for review or deployment!
