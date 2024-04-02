# 📦 Welcome to GPM-API

|                      |                                                |
| -------------------- | ---------------------------------------------- |
| Deployment           | [![PyPI](https://badge.fury.io/py/gpm_api.svg?style=flat)](https://pypi.org/project/gpm_api/) [![Conda](https://img.shields.io/conda/vn/conda-forge/gpm-api.svg?logo=conda-forge&logoColor=white&style=flat)](https://anaconda.org/conda-forge/gpm-api) |
| Activity             | [![PyPI Downloads](https://img.shields.io/pypi/dm/gpm_api.svg?label=PyPI%20downloads&style=flat)](https://pypi.org/project/gpm_api/) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/gpm-api.svg?label=Conda%20downloads&style=flat)](https://anaconda.org/conda-forge/gpm-api) |
| Python Versions      | [![Python Versions](https://img.shields.io/badge/Python-3.9%20%203.10%20%203.11%20%203.12-blue?style=flat)](https://www.python.org/downloads/) |
| Supported Systems    | [![Linux](https://img.shields.io/github/actions/workflow/status/ghiggi/gpm_api/.github/workflows/tests.yml?label=Linux&style=flat)](https://github.com/ghiggi/gpm_api/actions/workflows/tests.yml) [![macOS](https://img.shields.io/github/actions/workflow/status/ghiggi/gpm_api/.github/workflows/tests.yml?label=macOS&style=flat)](https://github.com/ghiggi/gpm_api/actions/workflows/tests.yml) [![Windows](https://img.shields.io/github/actions/workflow/status/ghiggi/gpm_api/.github/workflows/tests_windows.yml?label=Windows&style=flat)](https://github.com/ghiggi/gpm_api/actions/workflows/tests_windows.yml) |
| Project Status       | [![Project Status](https://www.repostatus.org/badges/latest/active.svg?style=flat)](https://www.repostatus.org/#active) |
| Build Status         | [![Tests](https://github.com/ghiggi/gpm_api/actions/workflows/tests.yml/badge.svg?style=flat)](https://github.com/ghiggi/gpm_api/actions/workflows/tests.yml) [![Lint](https://github.com/ghiggi/gpm_api/actions/workflows/lint.yml/badge.svg?style=flat)](https://github.com/ghiggi/gpm_api/actions/workflows/lint.yml) [![Docs](https://readthedocs.org/projects/gpm_api/badge/?version=latest&style=flat)](https://gpm_api.readthedocs.io/en/latest/) |
| Linting              | [![Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat)](https://github.com/astral-sh/ruff) [![Codespell](https://img.shields.io/badge/Codespell-enabled-brightgreen?style=flat)](https://github.com/codespell-project/codespell) |
| Code Coverage        | [![Coveralls](https://coveralls.io/repos/github/ghiggi/gpm_api/badge.svg?branch=main&style=flat)](https://coveralls.io/github/ghiggi/gpm_api?branch=main) [![Codecov](https://codecov.io/gh/ghiggi/gpm_api/branch/main/graph/badge.svg?style=flat)](https://codecov.io/gh/ghiggi/gpm_api) |
| Code Quality         | [![Codefactor](https://www.codefactor.io/repository/github/ghiggi/gpm_api/badge?style=flat)](https://www.codefactor.io/repository/github/ghiggi/gpm_api) [![Codebeat](https://codebeat.co/badges/236abcf2-cbae-4ca9-8a2d-3b70495bb16b?style=flat)](https://codebeat.co/projects/github-com-ghiggi-gpm_api-main) [![Codacy](https://app.codacy.com/project/badge/Grade/bee842cb10004ad8bb9288256f2fc8af?style=flat)](https://app.codacy.com/gh/ghiggi/gpm_api/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![Codescene](https://codescene.io/projects/36767/status-badges/code-health?style=flat)](https://codescene.io/projects/36767) |
| License              | [![License](https://img.shields.io/github/license/ghiggi/gpm_api?style=flat)](https://github.com/ghiggi/gpm_api/blob/main/LICENSE) |
| Community            | [![Slack](https://img.shields.io/badge/Slack-gpm_api-green.svg?logo=slack&style=flat)](https://join.slack.com/t/gpmapi/shared_invite/zt-28vkxzjs1-~cIYci2o3G0qEEoQJVMQRg) [![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-green?logo=github&style=flat)](https://github.com/ghiggi/gpm_api/discussions) |
| Citation             | [![DOI](https://zenodo.org/badge/286664485.svg?style=flat)](https://doi.org/10.5281/zenodo.10255084) |

 [**Slack**](https://join.slack.com/t/gpmapi/shared_invite/zt-28vkxzjs1-~cIYci2o3G0qEEoQJVMQRg) | [**Docs**](https://gpm-api.readthedocs.io/en/latest/)

## 🚀 Quick start
GPM-API provides an easy-to-use python interface to download, read, process and visualize most
of the products of the Global Precipitation Measurement Mission (GPM) data archive.

The list of available products can be retrieved using:

```python
import gpm

gpm.available_products(product_types="RS")  # research products
gpm.available_products(product_types="NRT") # near-real-time products

```

Before starting using GPM-API, we highly suggest to save into a configuration file:
1. your credentials to access the [NASA Precipitation Processing System (PPS) servers](https://gpm.nasa.gov/data/sources/pps-research)
2. the directory on the local disk where to save the GPM dataset of interest.

To facilitate the creation of the configuration file, you can run the following script:

```python
import gpm

username_pps = "<your PPS username>" # likely your mail
password_pps = "<your PPS password>" # likely your mail
base_dir = "<path/to/directory/GPM"  # path to the directory where to download the data
gpm.define_configs(username_pps=username_pps,
                   password_pps=password_pps,
                   base_dir=base_dir)

# You can check that the config file has been correctly created with:
configs = gpm.read_configs()
print(configs)

```

-----------------------------------------------------------------------------------------
#### 📥 Download GPM data

Now you can either start to download GPM data within python:

```python
import gpm
import datetime

product = "2A-DPR"
product_type = "RS"
version = 7

start_time = datetime.datetime(2020,7, 22, 0, 1, 11)
end_time = datetime.datetime(2020,7, 22, 0, 23, 5)

gpm.download(product=product,
             product_type=product_type,
             version=version,
             n_threads=2,
             start_time=start_time,
             end_time=end_time)

```

or from the terminal using i.e. `download_daily_gpm_data <product> <year> <month> <day>`:

```bash
download_daily_gpm_data 2A-DPR 2022 7 22
```
-----------------------------------------------------------------------------------------
#### 💫 Open GPM files into xarray

A GPM granule can be opened in python using:

```python
import gpm

ds = gpm.open_granule(<path_to_granule>)

```

while multiple granules over a specific time period can be opened using:

```python
import gpm
import datetime

product = "2A-DPR"
product_type = "RS"
version = 7

start_time = datetime.datetime(2020,7, 22, 0, 1, 11)
end_time = datetime.datetime(2020,7, 22, 0, 23, 5)
ds = gpm.open_dataset(product=product,
                      product_type=product_type,
                      version=version
                      start_time=start_time,
                      end_time=end_time)
```

-----------------------------------------------------------------------------------------
#### 📖 Explore the GPM-API documentation

To discover all GPM-API download, manipulation, analysis and plotting utilities, or how to contribute your custom retrieval to GPM-API:
- please read the software documentation available at [https://gpm-api.readthedocs.io/en/latest/](https://gpm-api.readthedocs.io/en/latest/).
- dive into the Jupyter Notebooks [Tutorials](https://github.com/ghiggi/gpm_api/tree/main/tutorials).

-----------------------------------------------------------------------------------------

## 🛠️ Installation

### conda

GPM-API can be installed via [conda][conda_link] on Linux, Mac, and Windows.
Install the package by typing the following command in the terminal:

```bash
conda install gpm-api
```

In case conda-forge is not set up for your system yet, see the easy to follow instructions on [conda-forge][conda_forge_link].

[conda_link]: https://docs.conda.io/en/latest/miniconda.html
[conda_forge_link]: https://github.com/conda-forge/gpm-api-feedstock#installing-gpm-api

### pip

GPM-API can be installed also via [pip][pip_link] on Linux, Mac, and Windows.
On Windows you can install [WinPython][winpy_link] to get Python and pip running.
Prior installation of GPM-API, try to install to `cartopy>=0.21.0` package to ensure there are not [GEOS](https://libgeos.org/) library version incompatibilities.
If you can't solve the problems and install cartopy with pip, you should install at least cartopy with conda using `conda install cartopy>=0.21.0`.

Then, install the GPM-API package by typing the following command in the terminal:

```bash
pip install gpm-api
```

To install the latest development version via pip, see the [documentation][dev_install_link].

[pip_link]: https://pypi.org/project/gpm-api
[winpy_link]: https://winpython.github.io/
[dev_install_link]: https://gpm-api.readthedocs.io/en/latest/02_installation.html#installation-for-contributors

## 💭 Feedback and Contributing Guidelines

If you aim to contribute your data or discuss the future development of GPM-API,
we highly suggest to join the [**GPM-API Slack Workspace**](https://join.slack.com/t/gpmapi/shared_invite/zt-28vkxzjs1-~cIYci2o3G0qEEoQJVMQRg)

Feel free to also open a [GitHub Issue](https://github.com/ghiggi/gpm_api/issues) or a [GitHub Discussion](https://github.com/ghiggi/gpm_api/discussions) specific to your questions or ideas.

## Citation

If you are using GPM-API in your publication please cite our Zenodo repository:

> Ghiggi Gionata. ghiggi/gpm_api. Zenodo. [![https://doi.org/10.5281/zenodo.7753488](https://zenodo.org/badge/286664485.svg?style=flat)](https://doi.org/10.5281/zenodo.7753488)

If you want to cite a specific software version, have a look at the [Zenodo site](https://doi.org/10.5281/zenodo.7753488).

## License

The content of this repository is released under the terms of the [MIT license](LICENSE).


### Tutorials

The documentation also includes some [tutorials][tut_link], showing the most important use cases of GPM-API.
These tutorial are also available as Jupyter Notebooks and in Google Colab:

- 1. Download the GPM products [[Notebook][tut1_download_link]][[Colab][colab1_download_link]]
- 2. Introduction to the IMERG products [[Notebook][tut2_imerg_link]][[Colab][colab2_imerg_link]]
- 2. Introduction to the PMW 1B and 1C products [[Notebook][tut2_pmw1bc_link]][[Colab][colab_pmw1bc_link]]
- 2. Introduction to the PMW 2A products [[Notebook][tut2_pmw2a_link]][[Colab][colab2_pmw2a_link]]
- 2. Introduction to the RADAR 2A products [[Notebook][tut2_radar_2a_link]][[Colab][colab2_radar_2a_link]]
- 2. Introduction to the CORRA 2B products [[Notebook][tut2_corra_2b_link]][[Colab][colab2_corra_2b_link]]
- 2. Introduction to the Latent Heating products [[Notebook][tut2_lh_link]][[Colab][colab2_lh_link]]
- 2. Introduction to the ENVironment products [[Notebook][tut2_env_link]][[Colab][colab2_env_link]]
- 3. Introduction to image labeling and patch extraction [[Notebook][tut3_label_link]][[Colab][colab3_label_link]]
- 3. Introduction to image patch extraction [[Notebook][tut3_patch_link]][[Colab][colab3_patch_link]]

The associated python scripts are also provided in the `tutorial` folder.

[tut1_download_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab1_download_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

[tut2_imerg_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab2_imerg_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

[tut2_pmw1bc_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab_pmw1bc_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

[tut2_pmw2a_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab2_pmw2a_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

[tut2_radar_2a_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab2_radar_2a_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

[tut2_corra_2b_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab2_corra_2b_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

[tut2_lh_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab2_lh_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

[tut2_env_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab2_env_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

[tut3_label_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab3_label_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

[tut3_patch_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials
[colab3_patch_link]: https://github.com/ghiggi/gpm_api/tree/master/tutorials

## Requirements:

- [xarray](https://docs.xarray.dev/en/stable/)
- [dask](https://www.dask.org/)
- [cartopy](https://scitools.org.uk/cartopy/docs/latest/)
- [pycolorbar](https://pycolorbar.readthedocs.io/en/latest/)
- [h5py](https://github.com/h5py/h5py)
- [curl](https://curl.se/)


### Optional

- [zarr](https://zarr.readthedocs.io/en/stable/)
- [ximage](https://github.com/ghiggi/ximage)
- [pyresample](https://pyresample.readthedocs.io/en/latest/)
- [pyvista](https://docs.pyvista.org/version/stable/)
- [wget](https://www.gnu.org/software/wget/)
