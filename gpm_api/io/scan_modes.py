#!/usr/bin/env python3
"""
Created on Thu Oct 13 11:26:27 2022

@author: ghiggi
"""
from gpm_api.io.products import (
    GPM_IMERG_products,
    GPM_PMW_2A_GPROF_RS_products,
    GPM_PMW_2A_PRPS_RS_products,
)


def available_scan_modes(product, version):

    ##---------------------------------------------.
    #### TRMM Radar
    if product in ["1B-PR"]:
        if version <= 6:
            scan_modes = ["NS"]
        else:  # V7 onward
            scan_modes = ["FS"]
    elif product in ["2A-PR"]:
        if version <= 6:
            scan_modes = ["NS"]
        else:  # V7 onward
            scan_modes = ["FS"]
    elif product in ["2A-ENV-PR"]:
        if version <= 6:
            scan_modes = ["NS"]
        else:  # V7 onward
            scan_modes = ["FS"]
    ##---------------------------------------------.
    #### GPM Radar
    # Ku
    elif product in ["1B-Ku", "2A-Ku", "2A-ENV-Ku"]:
        if version <= 6:
            scan_modes = ["NS"]
        else:  # V7 onward
            scan_modes = ["FS"]
    # Ka
    elif product in ["1B-Ka"]:
        scan_modes = ["HS", "MS"]  # this still holds in V7
    elif product in ["2A-Ka", "2A-ENV-Ka"]:
        if version <= 6:
            scan_modes = ["HS", "MS"]
        else:  # V7 onward
            scan_modes = ["HS", "FS"]
    # DPR
    elif product in ["2A-DPR"]:
        if version <= 6:
            scan_modes = ["NS", "MS", "HS"]
        else:  # V7 onward
            scan_modes = ["FS", "HS"]
    elif product in ["2A-ENV-DPR"]:
        if version <= 6:
            scan_modes = ["NS", "MS", "HS"]
        else:  # V7 onward
            scan_modes = ["FS", "HS"]
    ##---------------------------------------------.
    # CSH, SLH
    elif product in ["2A-GPM-SLH", "2A-TRMM-SLH", "2B-GPM-CSH", "2B-TRMM-CSH"]:
        scan_modes = ["Swath"]
    ##---------------------------------------------.
    # CORRA
    elif product in ["2B-GPM-CORRA"]:
        scan_modes = ["KuKaGMI", "KuGMI"]
    elif product in ["2B-TRMM-CORRA"]:
        scan_modes = ["KuTMI"]
    ##---------------------------------------------.
    # IMERG
    elif product in GPM_IMERG_products():
        scan_modes = ["Grid"]
    ##---------------------------------------------.
    # L1A PMW
    elif product in ["1A-GMI"]:
        scan_modes = ["S1", "S2", "S4", "S5"]  # gmi1aHeader, "S3" not implemented
    elif product in ["1A-TMI"]:
        scan_modes = ["S1", "S2", "S3"]  # tmi1aHeader
    # L1B PMW
    elif product in ["1B-TMI", "1C-TMI"]:
        scan_modes = ["S1", "S2", "S3"]
    elif product in ["1B-GMI", "1C-GMI"]:
        scan_modes = ["S1", "S2"]
    ##---------------------------------------------.
    # L1C PMW
    elif product.find("1C-SSMI") == 0 or product.find("1C-ATMS") == 0:
        scan_modes = ["S1", "S2", "S3", "S4"]
    elif product.find("1C-MHS") == 0:
        scan_modes = ["S1"]
    elif product.find("1C-GMI") == 0:
        scan_modes = ["S1", "S2"]
    elif product.find("1C-ATMS") == 0:
        scan_modes = ["S1", "S2", "S3", "S4"]
    elif product.find("1C-AMSR2") == 0:
        scan_modes = ["S1", "S2", "S3", "S4", "S5", "S6"]
    elif product.find("1C-AMSRE") == 0:
        scan_modes = ""  # TODO
    elif product.find("1C-AMSUB") == 0:
        scan_modes = ["S1"]
    elif product.find("1C-SAPHIR") == 0:
        scan_modes = ["S1"]
    ##---------------------------------------------.
    # L2 GPROF and PRPS
    elif product in GPM_PMW_2A_GPROF_RS_products():
        scan_modes = ["S1"]  # GprofHeadr
    elif product in GPM_PMW_2A_PRPS_RS_products():
        scan_modes = ["S1"]
    else:
        raise ValueError(f"Retrievals for product {product} not yet implemented.")
    return scan_modes
