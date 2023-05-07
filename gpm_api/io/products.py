#!/usr/bin/env python3
"""
Created on Thu Oct 13 11:13:15 2022

@author: ghiggi
"""
import numpy as np

from gpm_api.io.patterns import (
    GPM_1B_NRT_pattern_dict,
    GPM_1B_RS_pattern_dict,
    GPM_2A_NRT_pattern_dict,
    GPM_2A_RS_pattern_dict,
    GPM_2B_NRT_pattern_dict,
    GPM_2B_RS_pattern_dict,
    GPM_CMB_2B_NRT_pattern_dict,
    GPM_CMB_2B_RS_pattern_dict,
    GPM_CMB_NRT_pattern_dict,
    GPM_CMB_RS_pattern_dict,
    GPM_IMERG_NRT_pattern_dict,
    GPM_IMERG_pattern_dict,
    GPM_IMERG_RS_pattern_dict,
    GPM_PMW_1A_RS_pattern_dict,
    GPM_PMW_1B_NRT_pattern_dict,
    GPM_PMW_1B_RS_pattern_dict,
    GPM_PMW_1C_NRT_pattern_dict,
    GPM_PMW_1C_RS_pattern_dict,
    GPM_PMW_2A_GPROF_NRT_pattern_dict,
    GPM_PMW_2A_GPROF_RS_pattern_dict,
    GPM_PMW_2A_PRPS_NRT_pattern_dict,
    GPM_PMW_2A_PRPS_RS_pattern_dict,
    GPM_PMW_NRT_pattern_dict,
    GPM_PMW_RS_pattern_dict,
    GPM_RADAR_1B_RS_pattern_dict,
    GPM_RADAR_2A_NRT_pattern_dict,
    GPM_RADAR_2A_RS_pattern_dict,
    GPM_RADAR_NRT_pattern_dict,
    GPM_RADAR_RS_pattern_dict,
)


####--------------------------------------------------------------------------.
###########################
### Available products ####
###########################
##----------------------------------------------------------------------------.
#### RADAR
def GPM_RADAR_1B_RS_products():
    """Provide a list of available GPM DPR 1B-level RS data for download."""
    product_list = list(GPM_RADAR_1B_RS_pattern_dict().keys())
    return product_list


def GPM_RADAR_1B_NRT_products():
    """Provide a list of available GPM DPR 1B-level NRT data for download."""
    raise ValueError("NRT data for GPM DPR 1B not available !")


def GPM_RADAR_2A_RS_products():
    """Provide a list of available GPM DPR 2A-level RS data for download."""
    product_list = list(GPM_RADAR_2A_RS_pattern_dict().keys())
    return product_list


def GPM_RADAR_2A_NRT_products():
    """Provide a list of available GPM DPR 2A-level NRT data for download."""
    product_list = list(GPM_RADAR_2A_NRT_pattern_dict().keys())
    return product_list


def GPM_RADAR_2A_ENV_RS_products():
    """Provide a list of available GPM DPR 2A-level ENV RS data for download."""
    product_list = ["2A-ENV-DPR", "2A-ENV-Ka", "2A-ENV-Ku"]
    return product_list


def GPM_RADAR_2A_ENV_NRT_products():
    """Provide a list of available GPM DPR 2A-level ENV NRT data for download."""
    raise ValueError("NRT data for GPM DPR 2A-ENV not available !")


def GPM_RADAR_RS_products():
    """Provide a list of available GPM DPR RS data for download."""
    product_list = list(GPM_RADAR_RS_pattern_dict().keys())
    return product_list


def GPM_RADAR_NRT_products():
    """Provide a list of available GPM DPR NRT data for download."""
    product_list = list(GPM_RADAR_NRT_pattern_dict().keys())
    return product_list


####--------------------------------------------------------------------------.
#### PMW
def GPM_PMW_1A_RS_products():
    """Provide a list of available GPM PMW 1A-level RS data for download."""
    product_list = list(GPM_PMW_1A_RS_pattern_dict().keys())
    return product_list


def GPM_PMW_1B_RS_products():
    """Provide a list of available GPM PMW 1B-level RS data for download."""
    product_list = list(GPM_PMW_1B_RS_pattern_dict().keys())
    return product_list


def GPM_PMW_1B_NRT_products():
    """Provide a list of available GPM PMW 1B-level NRT data for download."""
    product_list = list(GPM_PMW_1B_NRT_pattern_dict().keys())
    return product_list


def GPM_PMW_1C_RS_products():
    """Provide a list of available GPM PMW 1C-level RS data for download."""
    product_list = list(GPM_PMW_1C_RS_pattern_dict().keys())
    return product_list


def GPM_PMW_1C_NRT_products():
    """Provide a list of available GPM PMW 1C-level NRT data for download."""
    product_list = list(GPM_PMW_1C_NRT_pattern_dict().keys())
    return product_list


def GPM_PMW_2A_GPROF_RS_products():
    """Provide a list of available GPM PMW 2A-level GPROF RS data for download."""
    product_list = list(GPM_PMW_2A_GPROF_RS_pattern_dict().keys())
    return product_list


def GPM_PMW_2A_GPROF_NRT_products():
    """Provide a list of available GPM PMW 2A-level GPROF NRT data for download."""
    product_list = list(GPM_PMW_2A_GPROF_NRT_pattern_dict().keys())
    return product_list


def GPM_PMW_2A_PRPS_RS_products():
    """Provide a list of available GPM PMW 2A-level PRPS RS data for download."""
    product_list = list(GPM_PMW_2A_PRPS_RS_pattern_dict().keys())
    return product_list


def GPM_PMW_2A_PRPS_NRT_products():
    """Provide a list of available GPM PMW 2A-level PRPS NRT data for download."""
    product_list = list(GPM_PMW_2A_PRPS_NRT_pattern_dict().keys())
    return product_list


def GPM_PMW_RS_products():
    """Provide a list of available RS GPM PMW data for download."""
    product_list = list(GPM_PMW_RS_pattern_dict().keys())
    return product_list


def GPM_PMW_NRT_products():
    """Provide a list of available NRT GPM PMW data for download."""
    product_list = list(GPM_PMW_NRT_pattern_dict().keys())
    return product_list


####--------------------------------------------------------------------------.
#### CMB
def GPM_CMB_2B_RS_products():
    """Provide a list of available GPM CMB 2B-level RS data for download."""
    product_list = list(GPM_CMB_2B_RS_pattern_dict().keys())
    return product_list


def GPM_CMB_2B_NRT_products():
    """Provide a list of available GPM CMB 2B-level NRT data for download."""
    product_list = list(GPM_CMB_2B_NRT_pattern_dict().keys())
    return product_list


def GPM_CMB_RS_products():
    """Provide a list of available RS GPM CMB data for download."""
    product_list = list(GPM_CMB_RS_pattern_dict().keys())
    return product_list


def GPM_CMB_NRT_products():
    """Provide a list of available NRT GPM CMB data for download."""
    product_list = list(GPM_CMB_NRT_pattern_dict().keys())
    return product_list


####--------------------------------------------------------------------------.
#### IMERG
def GPM_IMERG_NRT_products():
    """Provide a list of available GPM IMERG NRT data for download."""
    product_list = list(GPM_IMERG_NRT_pattern_dict().keys())
    return product_list


def GPM_IMERG_RS_products():
    """Provide a list of available GPM IMERG RS data for download."""
    product_list = list(GPM_IMERG_RS_pattern_dict().keys())
    return product_list


def GPM_IMERG_products():
    """Provide a list of available GPM IMERG data for download."""
    product_list = list(GPM_IMERG_pattern_dict().keys())
    return product_list


####--------------------------------------------------------------------------.
#### LEVELS
### 1B, 1C, 2A, 2B levels
def GPM_1B_RS_products():
    """Provide a list of available GPM 1B-level RS data for download."""
    product_list = list(GPM_1B_RS_pattern_dict().keys())
    return product_list


def GPM_1B_NRT_products():
    """Provide a list of available GPM 1B-level NRT data for download."""
    product_list = list(GPM_1B_NRT_pattern_dict().keys())
    return product_list


def GPM_1C_RS_products():
    """Provide a list of available GPM PMW 1C-level RS data for download."""
    product_list = list(GPM_PMW_1C_RS_pattern_dict().keys())
    return product_list


def GPM_1C_NRT_products():
    """Provide a list of available GPM PMW 1C-level NRT data for download."""
    product_list = list(GPM_PMW_1C_NRT_pattern_dict().keys())
    return product_list


def GPM_2A_RS_products():
    """Provide a list of available GPM 2A-level RS data for download."""
    product_list = list(GPM_2A_RS_pattern_dict().keys())
    return product_list


def GPM_2A_NRT_products():
    """Provide a list of available GPM 2A-level NRT data for download."""
    product_list = list(GPM_2A_NRT_pattern_dict().keys())
    return product_list


def GPM_2B_RS_products():
    """Provide a list of available GPM 2B-level RS data for download."""
    product_list = list(GPM_2B_RS_pattern_dict().keys())
    return product_list


def GPM_2B_NRT_products():
    """Provide a list of available GPM 2B-level NRT data for download."""
    product_list = list(GPM_2B_NRT_pattern_dict().keys())
    return product_list


####---------------------------------------------------------------------------.
#### RS vs. NRT


def GPM_RS_products():
    """Provide a list of available GPM RS data for download."""
    return (
        GPM_RADAR_RS_products()
        + GPM_PMW_RS_products()
        + GPM_CMB_RS_products()
        + GPM_IMERG_RS_products()
    )


def GPM_NRT_products():
    """Provide a list of available GPM NRT data for download."""
    return (
        GPM_RADAR_NRT_products()
        + GPM_PMW_NRT_products()
        + GPM_CMB_NRT_products()
        + GPM_IMERG_NRT_products()
    )


####--------------------------------------------------------------------------.
#### PRODUCTS


def available_products(product_type=None):
    """
    Provide a list of all/NRT/RS GPM data for download.

    Parameters
    ----------
    product_type : str, optional
        If None (default), provide all products (RS and NRT).
        If 'RS', provide a list of all GPM RS data for download.
        If 'NRT', provide a list of all GPM NRT data for download.
    Returns
    -------
    List
        List of GPM products name.

    """
    if product_type is None:
        return list(np.unique(GPM_RS_products() + GPM_NRT_products()))
    else:
        if product_type == "RS":
            return GPM_RS_products()
        if product_type == "NRT":
            return GPM_NRT_products()
        else:
            raise ValueError("Please specify 'product_type' either 'RS' or 'NRT'")
    return None


####--------------------------------------------------------------------------.
