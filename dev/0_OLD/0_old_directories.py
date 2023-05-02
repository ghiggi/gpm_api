#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:53:39 2022

@author: ghiggi
"""

import datetime

##----------------------------------------------------------------------------.
import os

from gpm_api.io.products import (  # CMB NRT?; GPM_DPR_1B_NRT_products,
    GPM_1B_RS_products,
    GPM_1C_NRT_products,
    GPM_CMB_2B_RS_products,
    GPM_CMB_RS_products,
    GPM_DPR_2A_NRT_products,
    GPM_DPR_2A_RS_products,
    GPM_DPR_NRT_products,
    GPM_DPR_RS_products,
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
    GPM_RS_products,
)

# TODO:
# GPM/version/<RS/NRT>/<PRODUCT>
# Search files without internet connection


# -----------------------------------------------------------------------------.
###############################
### File directory queries ####
###############################
def get_GPM_disk_directory(base_dir, product, product_type, date, version=6):
    """
    Provide the disk repository path where the requested GPM data are stored/need to be saved.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product name. See: GPM_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
        GPM data readers are currently implemented only for GPM V06.

    Returns
    -------
    None.

    """
    GPM_folder_name = "GPM_V" + str(version)
    if product_type == "RS":
        if product in GPM_PMW_RS_products():
            product_type_folder = "PMW_RS"
        elif product in GPM_IMERG_RS_products():
            product_type_folder = "IMERG_RS"
        elif product in GPM_DPR_RS_products():
            product_type_folder = "DPR_RS"
        elif product in GPM_CMB_RS_products():
            product_type_folder = "CMB_RS"
        else:
            raise ValueError("If this error appear, BUG when checking product")
    elif product_type == "NRT":
        if product in GPM_PMW_NRT_products():
            product_type_folder = "PMW_NRT"
        elif product in GPM_IMERG_NRT_products():
            product_type_folder = "IMERG"
        elif product in GPM_DPR_NRT_products():
            product_type_folder = "DPR_NRT"
        elif product in GPM_CMB_RS_products():
            product_type_folder = "CMB_NRT"
        else:
            raise ValueError("If this error appear, BUG when checking product")
    else:
        raise ValueError("If this error appear, BUG when checking product_type")

    DIR = os.path.join(
        base_dir,
        GPM_folder_name,
        product_type_folder,
        product,
        date.strftime("%Y"),
        date.strftime("%m"),
        date.strftime("%d"),
    )
    return DIR


def get_GPM_PPS_directory(product, product_type, date, version=6):
    """
    Provide the NASA PPS server directory path where the requested GPM data are stored.

    Parameters
    ----------
    product : str
        GPM product name. See: GPM_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
        GPM data readers are currently implemented only for GPM V06.

    Returns
    -------
    url_data_server : str
        url of the NASA PPS server from which to retrieve the data .
    url_file_list: list
        url of the NASA PPS server from which to retrieve the list of daily files.

    """
    ##------------------------------------------------------------------------.
    ### NRT data
    if product_type == "NRT":
        if product not in GPM_NRT_products():
            raise ValueError("Please specify a valid NRT product: GPM_NRT_products()")
        ## Specify servers
        url_server_text = "https://jsimpsonhttps.pps.eosdis.nasa.gov/text"
        url_data_server = "https://jsimpsonhttps.pps.eosdis.nasa.gov"
        # url_data_server = 'ftps://jsimpsonftps.pps.eosdis.nasa.gov'

        ## Retrieve NASA server folder name for NRT
        # GPM PMW 1B
        if product in GPM_PMW_1B_NRT_products():
            folder_name = "GMI1B"
        # GPM PMW 1C
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
        # GPM PMW 2A GPROF
        elif product in GPM_PMW_2A_GPROF_NRT_products():
            if product in ["2A-GMI"]:
                folder_name = "GPROF/GMI"
            elif product in ["2A-TMI"]:
                folder_name = "GPROF/TMI"
            elif product in ["2A-SSMI-F16", "2A-SSMI-F17", "2A-SSMI-F18"]:
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
        # GPM PMW 2A PPRS
        elif product in GPM_PMW_2A_PRPS_NRT_products():
            folder_name = "PRPS"
        # GPM DPR 2A
        elif product in GPM_DPR_2A_NRT_products():
            if product == "2A-Ku":
                folder_name = "radar/KuL2"
            elif product == "2A-Ka":
                folder_name = "radar/KaL2"
            elif product == "2A-DPR":
                folder_name = "radar/DprL2"
            else:
                raise ValueError("BUG - Some product option is missing.")
        # GPM CMB 2B
        elif product in GPM_DPR_2A_NRT_products():
            if product == "2B-GPM-CMB":
                folder_name = "combine"
            else:
                raise ValueError("BUG - Some product option is missing.")

        # GPM IMERG NRT
        elif product in GPM_IMERG_NRT_products():
            if product == "IMERG-ER":
                folder_name = "imerg/early"
            elif product == "IMERG-LR":
                folder_name = "imerg/late"
            else:
                raise ValueError("BUG - Some product option is missing.")
            # Specify the url to retrieve the daily list of IMERG NRT products
            url_file_list = (
                url_server_text
                + "/"
                + folder_name
                + "/"
                + datetime.datetime.strftime(date, "%Y%m")
                + "/"
            )
            return url_file_list
        else:
            raise ValueError("BUG - Some product option is missing.")
        # Specify the url to retrieve the daily list of NRT data
        url_file_list = url_server_text + "/" + folder_name + "/"
    ##------------------------------------------------------------------------.
    #### RS data
    elif product_type == "RS":
        if product not in GPM_RS_products():
            raise ValueError("Please specify a valid NRT product: GPM_RS_products()")
        ## Specify servers
        url_server_text = "https://arthurhouhttps.pps.eosdis.nasa.gov/text"
        url_data_server = "https://arthurhouhttps.pps.eosdis.nasa.gov"
        #  url_data_server = 'ftp://arthurhouftps.pps.eosdis.nasa.gov'

        ## Retrieve NASA server folder name for RS
        # GPM DPR 1B (and GMI)
        if product in GPM_1B_RS_products():
            folder_name = "1B"
        # GPM DPR 2A
        elif product in GPM_DPR_2A_RS_products():
            folder_name = "radar"
        # GPM PMW 1A
        elif product in GPM_PMW_1A_RS_products():
            folder_name = "1A"
        # GPM PMW 1C
        elif product in GPM_PMW_1C_RS_products():
            folder_name = "1C"
        # GPM PMW 2A PRPS
        elif product in GPM_PMW_2A_PRPS_RS_products():
            folder_name = "prps"
        # GPM PMW 2A GPROF
        elif product in GPM_PMW_2A_GPROF_RS_products():
            folder_name = "gprof"
        # GPM CMB 2B
        elif product in GPM_CMB_2B_RS_products():
            folder_name = "radar"
        # GPM IMERG
        elif product == "IMERG-FR":
            folder_name = "imerg"
        else:
            raise ValueError("BUG - Some product is missing.")
        # Specify the url where to retrieve the daily list of GPM RS data
        if version == 7:
            url_file_list = (
                url_server_text
                + "/gpmdata/"
                + datetime.datetime.strftime(date, "%Y/%m/%d")
                + "/"
                + folder_name
                + "/"
            )
        elif version in [4, 5, 6]:
            version_str = "V0" + str(int(version))
            url_file_list = (
                url_server_text
                + "/gpmallversions/"
                + version_str
                + "/"
                + datetime.datetime.strftime(date, "%Y/%m/%d")
                + "/"
                + folder_name
                + "/"
            )
        else:
            raise ValueError("Please specify either version 4, 5, 6 or 7.")
    ##------------------------------------------------------------------------.
    return (url_data_server, url_file_list)


##-----------------------------------------------------------------------------.
