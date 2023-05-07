#!/usr/bin/env python3
"""
Created on Thu Oct 13 16:48:22 2022

@author: ghiggi
"""

##----------------------------------------------------------------------------.
import datetime
import os

from gpm_api.io.checks import check_base_dir
from gpm_api.io.products import (
    GPM_1B_RS_products,
    GPM_1C_NRT_products,
    GPM_CMB_2B_RS_products,
    GPM_CMB_RS_products,
    # CMB NRT?
    GPM_IMERG_NRT_products,
    GPM_IMERG_RS_products,
    GPM_NRT_products,
    GPM_PMW_1A_RS_products,
    GPM_PMW_1B_NRT_products,
    GPM_PMW_1C_RS_products,
    GPM_PMW_2A_GPROF_NRT_products,
    GPM_PMW_2A_GPROF_RS_products,
    GPM_PMW_2A_PRPS_NRT_products,
    GPM_PMW_2A_PRPS_RS_products,
    GPM_PMW_NRT_products,
    GPM_PMW_RS_products,
    # GPM_RADAR_1B_NRT_products,
    GPM_RADAR_2A_NRT_products,
    GPM_RADAR_2A_RS_products,
    GPM_RADAR_NRT_products,
    # CMB NRT?
    GPM_RADAR_RS_products,
    GPM_RS_products,
)

####--------------------------------------------------------------------------.
####################
#### LOCAL DISK ####
####################


def get_disk_dir_pattern(product, product_type, version):
    """
    Defines the local (disk) repository base pattern where data are stored and searched.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.

    Returns
    -------

    pattern : str
        Directory base pattern.
        If product_type == "RS": GPM/RS/V<version>/<product_category>/<product>
        If product_type == "NRT": GPM/NRT/<product_category>/<product>
        Product category are: RADAR, PMW, CMB, IMERG

    """

    if product_type == "RS":
        if product in GPM_PMW_RS_products():
            product_category = "PMW"
        elif product in GPM_IMERG_RS_products():
            product_category = "IMERG"
        elif product in GPM_RADAR_RS_products():
            product_category = "RADAR"
        elif product in GPM_CMB_RS_products():
            product_category = "CMB"
        else:
            raise ValueError("If this error appear, BUG when checking product.")
    elif product_type == "NRT":
        if product in GPM_PMW_NRT_products():
            product_category = "PMW"
        elif product in GPM_IMERG_NRT_products():
            product_category = "IMERG"
        elif product in GPM_RADAR_NRT_products():
            product_category = "RADAR"
        elif product in GPM_CMB_RS_products():
            product_category = "CMB"
        else:
            raise ValueError("If this error appear, BUG when checking product.")
    else:
        raise ValueError("Valid product_type: 'RS' and 'NRT'.")

    # Define pattern
    if product_type == "NRT":
        dir_structure = os.path.join("GPM", product_type, product_category, product)
    else:  # product_type == "RS"
        version_str = "V0" + str(int(version))
        dir_structure = os.path.join("GPM", product_type, version_str, product_category, product)
    return dir_structure


def get_disk_directory(base_dir, product, product_type, date, version):
    """
    Provide the disk repository path where the requested GPM data are stored/need to be saved.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product name. See: gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.

    Returns
    -------

    dir_path : str
        Directory path where data are located.
        If product_type == "RS": <base_dir>/GPM/RS/V0<version>/<product_category>/<product>/<YYYY>/<MM>/<DD>
        If product_type == "NRT": <base_dir>/GPM/NRT/<product_category>/<product>/<YYYY>/<MM>/<DD>
        <product_category> are: RADAR, PMW, CMB, IMERG.

    """
    base_dir = check_base_dir(base_dir)
    dir_structure = get_disk_dir_pattern(product, product_type, version)

    dir_path = os.path.join(
        base_dir,
        dir_structure,
        date.strftime("%Y"),
        date.strftime("%m"),
        date.strftime("%d"),
    )
    return dir_path


####--------------------------------------------------------------------------.
####################
#### PPS SERVER ####
####################


def get_pps_nrt_product_dir(product, date):
    """
    Retrieve the NASA PPS server directory stucture where NRT data are stored.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products() .
    date : datetime.date
        Single date for which to retrieve the data.
        Note: this is currently only needed when retrieving IMERG data.
    """
    ####----------------------------------------------------------------------.
    #### Check product validity
    if product not in GPM_NRT_products():
        raise ValueError("Please specify a valid NRT product. See GPM_NRT_products().")

    ####----------------------------------------------------------------------.
    #### Retrieve NASA server folder name for NRT
    #### - GPM PMW 1B
    if product in GPM_PMW_1B_NRT_products():
        folder_name = "GMI1B"
    # ------------------------------------------------------------------------.
    #### - GPM PMW 1C
    elif product in GPM_1C_NRT_products():
        if product == "1C-GMI":
            folder_name = "1C/GMI"
        elif product in ["1C-SSMI-F16", "1C-SSMI-F17", "1C-SSMI-F18"]:
            folder_name = "1C/SSMIS"
        elif product == "1C-ASMR2-GCOMW1":
            folder_name = "1C/AMSR2"
        elif product == "1C-SAPHIR-MT1":
            folder_name = "1C/SAPHIR"
        elif product in ["1C-MHS-METOPB", "1C-MHS-METOPC", "1C-MHS-NOAA19"]:
            folder_name = "1C/MHS"
        elif product in ["1C-ATMS-NOAA20", "1C-ATMS-NPP"]:
            folder_name = "1C/ATMS"
        else:
            raise ValueError("BUG - Some product option is missing.")
    # ------------------------------------------------------------------------.
    #### - GPM PMW 2A GPROF
    elif product in GPM_PMW_2A_GPROF_NRT_products():
        if product in ["2A-GMI"]:
            folder_name = "GPROF/GMI"
        elif product in ["2A-TMI"]:
            folder_name = "GPROF/TMI"
        elif product in ["2A-SSMIS-F16", "2A-SSMIS-F17", "2A-SSMIS-F18"]:
            folder_name = "GPROF/SSMIS"
        elif product in ["2A-ASMR2-GCOMW1"]:
            folder_name = "GPROF/AMSR2"
        elif product in ["2A-MHS-METOPB", "2A-MHS-METOPC", "2A-MHS-NOAA19"]:
            folder_name = "GPROF/MHS"
        elif product in ["2A-ATMS-NOAA20", "2A-ATMS-NPP"]:
            folder_name = "GPROF/ATMS"
        elif product in ["2A-SAPHIR-MT1"]:
            folder_name = "GPROF/SAPHIR"
        else:
            raise ValueError("BUG - Some product option is missing.")
    # ------------------------------------------------------------------------.
    #### - GPM PMW 2A PPRS
    elif product in GPM_PMW_2A_PRPS_NRT_products():
        folder_name = "PRPS"
    # ------------------------------------------------------------------------.
    #### - GPM DPR 2A
    elif product in GPM_RADAR_2A_NRT_products():
        if product == "2A-Ku":
            folder_name = "radar/KuL2"
        elif product == "2A-Ka":
            folder_name = "radar/KaL2"
        elif product == "2A-DPR":
            folder_name = "radar/DprL2"
        else:
            raise ValueError("BUG - Some product option is missing.")
    # ------------------------------------------------------------------------.
    #### - GPM CMB 2B
    elif product in GPM_RADAR_2A_NRT_products():
        if product == "2B-GPM-CMB":
            folder_name = "combine"
        else:
            raise ValueError("BUG - Some product option is missing.")

    # ------------------------------------------------------------------------.
    #### - GPM IMERG NRT
    elif product in GPM_IMERG_NRT_products():
        if product == "IMERG-ER":
            folder_name = "imerg/early"
        elif product == "IMERG-LR":
            folder_name = "imerg/late"
        else:
            raise ValueError("BUG - Some product option is missing.")

        #### Specify the directory structure for the daily list of IMERG NRT products
        dir_structure = os.path.join(folder_name, datetime.datetime.strftime(date, "%Y%m"))
        return dir_structure
    # ------------------------------------------------------------------------.
    else:
        raise ValueError("BUG - Some product option is missing.")

    #### Specify the directory structure for daily list of NRT data (exect IMERG)
    dir_structure = folder_name

    ####----------------------------------------------------------------------.
    return dir_structure


def get_pps_rs_product_dir(product, date, version):
    """
    Retrieve the NASA PPS server directory stucture where RS data are stored.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products() .

    date : datetime.date
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    """

    ####----------------------------------------------------------------------.
    #### Check product validity
    if product not in GPM_RS_products():
        raise ValueError("Please specify a valid NRT product. See GPM_RS_products().")

    ####----------------------------------------------------------------------.
    #### Retrieve NASA server folder name for RS
    #### - GPM DPR 1B (and GMI)
    if product in GPM_1B_RS_products():
        folder_name = "1B"
    # ------------------------------------------------------------------------.
    #### - GPM DPR 2A
    elif product in GPM_RADAR_2A_RS_products():
        folder_name = "radar"
    # ------------------------------------------------------------------------.
    #### - GPM PMW 1A
    elif product in GPM_PMW_1A_RS_products():
        folder_name = "1A"
    # ------------------------------------------------------------------------.
    #### -  GPM PMW 1C
    elif product in GPM_PMW_1C_RS_products():
        folder_name = "1C"
    # ------------------------------------------------------------------------.
    #### -  GPM PMW 2A PRPS
    elif product in GPM_PMW_2A_PRPS_RS_products():
        folder_name = "prps"
    # ------------------------------------------------------------------------.
    #### -  GPM PMW 2A GPROF
    elif product in GPM_PMW_2A_GPROF_RS_products():
        folder_name = "gprof"
    # ------------------------------------------------------------------------.
    #### -  GPM CMB 2B
    elif product in GPM_CMB_2B_RS_products():
        folder_name = "radar"
    # ------------------------------------------------------------------------.
    #### -  GPM IMERG
    elif product == "IMERG-FR":
        folder_name = "imerg"
    else:
        raise ValueError("BUG - Some product is missing.")
    # ------------------------------------------------------------------------.
    #### Specify the directory structure for current RS version
    if version == 7:
        dir_structure = os.path.join(
            "gpmdata",
            datetime.datetime.strftime(date, "%Y/%m/%d"),
            folder_name,
        )
    #### Specify the directory structure for old RS version
    elif version in [4, 5, 6]:
        version_str = "V0" + str(int(version))
        dir_structure = os.path.join(
            "gpmallversions",
            version_str,
            datetime.datetime.strftime(date, "%Y/%m/%d"),
            folder_name,
        )
    else:
        raise ValueError("Please specify either version 4, 5, 6 or 7.")
    # ------------------------------------------------------------------------.
    # Return the directory structure
    return dir_structure


def get_pps_directory(product, product_type, date, version):
    """
    Retrieve the NASA PPS server directory paths where the GPM data are listed and stored.

    The data list is retrieved using https.
    The data stored are retrieved using ftps.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products() .
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.

    Returns
    -------
    url_data_server : str
        url of the NASA PPS server where the data are stored.
    url_data_list: list
        url of the NASA PPS server where the data are listed.

    """
    ##------------------------------------------------------------------------.
    ### NRT data
    if product_type == "NRT":
        #### Specify servers
        url_server_text = "https://jsimpsonhttps.pps.eosdis.nasa.gov/text"
        url_data_server = "ftps://jsimpsonftps.pps.eosdis.nasa.gov"
        # url_data_server = 'ftps://jsimpsonftps.pps.eosdis.nasa.gov'

        # Retrieve directory structure
        dir_structure = get_pps_nrt_product_dir(product, date)

        # Define url where data are listed
        url_data_list = os.path.join(url_server_text, dir_structure)

    ##------------------------------------------------------------------------.
    #### RS data
    elif product_type == "RS":

        ## Specify servers
        url_server_text = "https://arthurhouhttps.pps.eosdis.nasa.gov/text"
        url_data_server = "ftps://arthurhouftps.pps.eosdis.nasa.gov"

        # Retrieve directory structure
        dir_structure = get_pps_rs_product_dir(product, date, version)

        # Define url where data are listed
        url_data_list = os.path.join(url_server_text, dir_structure)

    ##------------------------------------------------------------------------.
    return (url_data_server, url_data_list)


####--------------------------------------------------------------------------.
