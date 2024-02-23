===========
Quick Start
===========

GPM-API allows to download data from the NASA Precipitation Processing System (PPS) and NASA GES DISC Archives.
To download the data, it is necessary to first create two accounts and successively create the GPM-API configuration file.


Register to the NASA GES DISC
-------------------------------

To access the GPM data through the GES DISC Data Archive you need to have an `Earthdata Login` account.
If you don't have one, you can register on the `EarthData Portal <https://urs.earthdata.nasa.gov/>`__.

Once you have an EarthData account, to access the GES DISC Data Archive you need to authorize
the application `"NASA GESDISC DATA ARCHIVE"` by following the
instructions available at the following `link <https://disc.gsfc.nasa.gov/earthdata-login>`__.


Register to the NASA PPS
---------------------------

To register to the PPS servers, please follow the instructions available at the following `link  <https://registration.pps.eosdis.nasa.gov/registration/>`__.

If you plan to use Near-Real Time (NRT) data stored on PPS, make sure to check the box stating that you are interested in the NRT products.
NRT products includes for example the IMERG Early and Late Runs products.


Create the GPM-API configuration file
---------------------------------------

The GPM-API configuration file stores the credentials to access the PPS and GES DISC servers
as well as other parameters such as the directory on your local machine where to
save the GPM dataset of interest.

To facilitate the creation of the configuration file, you can adapt and run the following script in Python.
The configuration file will be created in the user's home directory under the name `.config_gpm_api.yaml`.

.. code-block:: python

    import gpm_api

    username_pps = "<your PPS username>"  # likely your mail, all in lowercase
    password_pps = "<your PPS password>"  # likely your mail, all in lowercase
    username_earthdata = "<your EarthData username>"
    password_earthdata = "<your EarthData password>"
    base_dir = "<path/to/a/local/directory/>"  # where to download all GPM data
    gpm_api.define_configs(
        username_pps=username,
        password_pps=password,
        username_earthdata=username_earthdata,
        password_earthdata=password_earthdata,
        base_dir=base_dir,
    )

    # You can check that the config file has been correctly created with:
    configs = gpm_api.read_configs()
    print(configs)



Search the product
--------------------

The products are organized in different categories, such as 'research' (RS) and 'Near-Real-Time' (NRT) products.
Please note that the **NRT products are only available through the PPS server**!!!.

To list the available "RS" and "NRT" products, you can use the following command:

.. code-block:: python

    import gpm_api

    gpm_api.available_products(product_types="RS")  # research products
    gpm_api.available_products(product_types="NRT")  # near-real-time products



You can also search for a specific category of products:

.. code-block:: python

    gpm_api.available_products(product_categories="PMW")  # Passive Microwave
    gpm_api.available_products(product_categories="RADAR")
    gpm_api.available_products(product_categories="CMB")  # Combined products
    gpm_api.available_products(product_categories="IMERG")



specific product levels:

.. code-block:: python

    gpm_api.available_products(product_levels="1C")
    gpm_api.available_products(product_levels=["1B", "1C"])
    gpm_api.available_products(product_levels="2A")

    gpm_api.available_products(product_levels="2A", product_categories="RADAR")
    gpm_api.available_products(product_levels="2A", product_categories="PMW")


specific time periods:

.. code-block:: python

    gpm_api.available_products(end_time="1995-01-31")  # from the start of the mission to 1995-01-31
    gpm_api.available_products(start_time="2014-01-01", end_time="2016", product_categories="PMW")
    gpm_api.available_products(start_time="2019-01-01")  # from 2019-01-01 to the present



specific sensors or satellites:

.. code-block:: python

    gpm_api.available_products(satellites="GPM")
    gpm_api.available_products(satellites="TRMM")
    gpm_api.available_products(satellites="GPM", product_categories="PMW")
    gpm_api.available_products(satellites="TRMM", product_categories="RADAR")

    gpm_api.available_products(sensors="SSMIS")
    gpm_api.available_products(sensors="SSMI")


A list of available satellites and sensors can be retrieved using:

.. code-block:: python

    gpm_api.available_satellites()
    gpm_api.available_sensors()


Download the data
--------------------

With the GPM-API you can either download the data from the command line or from Python.

To download the data in Python, you can adapt the following code snippet:

.. code-block:: python

    import gpm_api
    import datetime

    product = "2A-DPR"
    product_type = "RS"
    version = 7
    storage = "pps"  # or "ges_disc"

    start_time = datetime.datetime(2020, 7, 22, 1, 10, 11)
    end_time = datetime.datetime(2020, 7, 22, 2, 30, 5)

    # Download data over specific time periods
    gpm_api.download(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        storage=storage,
    )

    # Download data over a specific day
    gpm_api.download_daily_data(
        year=2022,
        month=1,
        day=1,
        product=product,
        product_type=product_type,
        version=version,
        storage=storage,
    )

    # Download data over a specific month
    download_monthly_data(
        year=2022,
        month=1,
        product=product,
        product_type=product_type,
        version=version,
        storage=storage,
    )

From the command line, you can download the data using similar commands.
For example, to download all data of a given product over a specific day, you can use:

.. code-block:: bash

    download_gpm_daily_data 2A-DPR 2022 7 22

and to download data over a specific period, you can use:

.. code-block:: bash

    download_gpm_data 2A-DPR --start-time "2022-07-22 00:01:11" --end-time "2022-07-22 00:23:05"

For more information on the available options, you can use the following commands:

.. code-block:: bash

    download_gpm_data --help
    download_gpm_daily_data --help
    download_gpm_monthly_data --help


Open the data
----------------

Within the GPM-API, we use the name "granule" to refer to a single file,
while the name "dataset" is used to refer to a collection of granules.

GPM-API enables to open single or multiple granules into an xarray Dataset, an object designed for working with labeled multi-dimensional arrays.

The `gpm_api.open_granule(filepath)` opens a single file into xarray by providing the path of the file of interest.

The `gpm_api.open_dataset` function enable to open a collection of granules over a period of interest.

The following example shows how to download and open a dataset over a specific time period:

.. code-block:: python

    import gpm_api
    import datetime

    product = "1C-GMI"
    product_type = "RS"
    version = 7
    storage = "pps"  # or "ges_disc"

    start_time = datetime.datetime(2020, 7, 22, 1, 10, 11)
    end_time = datetime.datetime(2020, 7, 22, 2, 30, 5)

    # Download data over a specific time period
    gpm_api.download(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        storage=storage,
    )

    # Open the dataset over a specific time period
    ds = gpm_api.open_dataset(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
    )

    # Plot a specific variable of the dataset
    ds["Tc"].gpm_api.plot_map()


You are now ready to explore the various :ref:`tutorials <tutorials>` available in the documentation and learn more about the GPM-API functionalities.

If you are not familiar with xarray, we highly suggest to first have a look also at the `xarray documentation <http://xarray.pydata.org/en/stable/>`__.
