#!/usr/bin/env python3
"""
Created on Thu Oct 13 11:12:31 2022

@author: ghiggi
"""
# ----------------------------------------------------------------------------.
#################################
### File Pattern dictionary  ####
#################################
### RADAR
def GPM_RADAR_1B_RS_pattern_dict():
    """Return the filename pattern* associated to GPM DPR 1B products."""
    GPM_dict = {"1B-PR": "1B.TRMM.PR*", "1B-Ka": "GPMCOR_KAR*", "1B-Ku": "GPMCOR_KUR*"}
    return GPM_dict


def GPM_RADAR_2A_RS_pattern_dict():
    """Return the filename pattern* associated to GPM DPR 2A RS products."""
    GPM_dict = {
        "2A-PR": "2A.TRMM.PR.V\\d-*",  # to distinguish from SLH
        "2A-DPR": "2A.GPM.DPR.V\\d-*",  # to distinguish from SLH
        "2A-Ka": "2A.GPM.Ka.V*",
        "2A-Ku": "2A.GPM.Ku.V*",
        "2A-ENV-PR": "2A-ENV.TRMM.PR.V*",
        "2A-ENV-DPR": "2A-ENV.GPM.DPR.V*",
        "2A-ENV-Ka": "2A-ENV.GPM.Ka.V*",
        "2A-ENV-Ku": "2A-ENV.GPM.Ku.V*",
        "2A-GPM-SLH": "2A.GPM.DPR.GPM-SLH*",
        "2A-TRMM-SLH": "2A.TRMM.PR.TRMM-SLH*",
    }
    return GPM_dict


def GPM_RADAR_2A_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM DPR 2A NRT products."""
    GPM_dict = {
        "2A-DPR": "2A.GPM.DPR.V\\d-*",  # to distinguish from SLH
        "2A-Ka": "2A.GPM.Ka.V*",
        "2A-Ku": "2A.GPM.Ku.V*",
    }
    return GPM_dict


def GPM_RADAR_RS_pattern_dict():
    """Return the filename pattern* associated to GPM DPR RS products."""
    GPM_dict = GPM_RADAR_1B_RS_pattern_dict()
    GPM_dict.update(GPM_RADAR_2A_RS_pattern_dict())
    return GPM_dict


def GPM_RADAR_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM DPR NRT products."""
    GPM_dict = GPM_RADAR_2A_NRT_pattern_dict()
    return GPM_dict


##----------------------------------------------------------------------------.
#### CMB
def GPM_CMB_2B_RS_pattern_dict():
    """Return the filename pattern* associated to GPM 2B RS products."""
    GPM_dict = {
        "2B-GPM-CORRA": "2B.GPM.DPRGMI.CORRA*",
        "2B-TRMM-CORRA": "2B.TRMM.PRTMI.CORRA*",
        "2B-GPM-CSH": "2B.GPM.DPRGMI.2HCSHv*",
        "2B-TRMM-CSH": "2B.TRMM.PRTMI.2HCSHv*",
    }
    return GPM_dict


def GPM_CMB_2B_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM 2B NRT products."""
    GPM_dict = {"2B-GPM-CORRA": "2B.GPM.DPRGMI.CORRA*"}
    return GPM_dict


def GPM_CMB_RS_pattern_dict():
    """Return the filename pattern* associated to GPM CMB RS products."""
    GPM_dict = GPM_CMB_2B_RS_pattern_dict()
    return GPM_dict


def GPM_CMB_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM CMB NRT products."""
    GPM_dict = GPM_CMB_2B_NRT_pattern_dict()
    return GPM_dict


##----------------------------------------------------------------------------.
### PMW
def GPM_PMW_1A_RS_pattern_dict():
    """Return the filename pattern* associated to GPM PMW 1A products."""
    GPM_dict = {"1A-TMI": "1A.TRMM.TMI.*", "1A-GMI": "1A.GPM.GMI.*"}
    return GPM_dict


def GPM_PMW_1B_RS_pattern_dict():
    """Return the filename pattern* associated to GPM PMW 1B products."""
    GPM_dict = {"1B-TMI": "1B.TRMM.TMI.*", "1B-GMI": "1B.GPM.GMI.*"}
    return GPM_dict


def GPM_PMW_1B_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM PMW 1B products."""
    GPM_dict = {"1B-GMI": "1B.GPM.GMI.*"}
    return GPM_dict


def GPM_PMW_1C_RS_pattern_dict():
    """Return the filename pattern* associated to GPM PMW 1C products."""
    # Common calibrated brightness temperatures
    GPM_dict = {
        "1C-TMI": "1C.TRMM.TMI.*",
        "1C-GMI": "1C.GPM.GMI.*",
        "1C-SSMI-F08": "1C.F08.SSMI.*",
        "1C-SSMI-F10": "1C.F10.SSMI.*",
        "1C-SSMI-F11": "1C.F11.SSMI.*",
        "1C-SSMI-F13": "1C.F13.SSMI.*",
        "1C-SSMI-F14": "1C.F14.SSMI.*",
        "1C-SSMI-F15": "1C.F15.SSMI.*",
        "1C-SSMIS-F16": "1C.F16.SSMIS.*",
        "1C-SSMIS-F17": "1C.F17.SSMIS.*",
        "1C-SSMIS-F18": "1C.F18.SSMIS.*",
        "1C-SSMIS-F19": "1C.F19.SSMIS.*",
        "1C-AMSR2-GCOMW1": "1C.GCOMW1.AMSR2.*",
        "1C-AMSRE-AQUA": "1C.AQUA.AMSRE.*",
        "1C-AMSUB-NOAA15": "1C.NOAA15.AMSUB.*",
        "1C-AMSUB-NOAA16": "1C.NOAA16.AMSUB.*",
        "1C-AMSUB-NOAA17": "1C.NOAA17.AMSUB.*",
        "1C-SAPHIR-MT1": "1C.MT1.SAPHIR.*",
        "1C-MHS-METOPA": "1C.METOPA.MHS.*",
        "1C-MHS-METOPB": "1C.METOPB.MHS.*",
        "1C-MHS-METOPC": "1C.METOPC.MHS.*",
        "1C-MHS-NOAA18": "1C.NOAA18.MHS.*",
        "1C-MHS-NOAA19": "1C.NOAA19.MHS.*",
        "1C-ATMS-NOAA20": "1C.NOAA20.ATMS.*",
        "1C-ATMS-NPP": "1C.NPP.ATMS.*",
    }
    return GPM_dict


def GPM_PMW_1C_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM PMW 1C products."""
    # Common calibrated brightness temperatures
    GPM_dict = {
        "1C-GMI": "1C.GPM.GMI.*",
        "1C-SSMIS-F16": "1C.F16.SSMIS.*",
        "1C-SSMIS-F17": "1C.F17.SSMIS.*",
        "1C-SSMIS-F18": "1C.F18.SSMIS.*",
        "1C-AMSR2-GCOMW1": "1C.GCOMW1.AMSR2.*",
        "1C-SAPHIR-MT1": "1C.MT1.SAPHIR.*",
        "1C-MHS-METOPB": "1C.METOPB.MHS.*",
        "1C-MHS-METOPC": "1C.METOPC.MHS.*",
        "1C-MHS-NOAA19": "1C.NOAA19.MHS.*",
        "1C-ATMS-NOAA20": "1C.NOAA20.ATMS.*",
        "1C-ATMS-NPP": "1C.NPP.ATMS.*",
    }
    return GPM_dict


def GPM_PMW_2A_GPROF_RS_pattern_dict():
    """Return the filename pattern* associated to GPM PMW GPROF 2A products."""
    GPM_dict = {  # Using ERA-I as environment ancillary data
        # '2A-GMI-CLIM':  '2A-CLIM.GPM.GMI.*',
        # '2A-TMI-CLIM': '2A-CLIM.TMI.TRMM.*',
        # '2A-SSMI-F11-CLIM': '2A-CLIM.F11.SSMIS.*',
        # '2A-SSMI-F13-CLIM': '2A-CLIM.F13.SSMIS.*',
        # '2A-SSMI-F14-CLIM': '2A-CLIM.F14.SSMIS.*',
        # '2A-SSMI-F15-CLIM': '2A-CLIM.F15.SSMIS.*',
        # '2A-SSMI-F16-CLIM': '2A-CLIM.F16.SSMIS.*',
        # '2A-SSMI-F17-CLIM': '2A-CLIM.F17.SSMIS.*',
        # '2A-SSMI-F18-CLIM': '2A-CLIM.F18.SSMIS.*',
        # '2A-AMSR2-GCOMW1-CLIM': '2A-CLIM.GCOMW1.AMSR2.*',
        # '2A-AMSRE-AQUA-CLIM': '2A-CLIM.AQUA.AMSRE.*',
        # '2A-AMSUB-NOAA15-CLIM': '2A-CLIM.AMSUB.NOAA15.*',
        # '2A-AMSUB-NOAA16-CLIM': '2A-CLIM.AMSUB.NOAA16.*',
        # '2A-AMSUB-NOAA17-CLIM': '2A-CLIM.AMSUB.NOAA17.*',
        # '2A-SAPHIR-MT1-CLIM' : '2A-CLIM.SAPHIR.MT1.*',
        # '2A-MHS-METOPA-CLIM': '2A-CLIM.METOPA.MHS.*',
        # '2A-MHS-METOPB-CLIM': '2A-CLIM.METOPB.MHS.*',
        # '2A-MHS-METOPC-CLIM': '2A-CLIM.METOPC.MHS.*',
        # '2A-MHS-NOAA18-CLIM': '2A-CLIM.NOAA18.MHS.*',
        # '2A-MHS-NOAA19-CLIM': '2A-CLIM.NOAA19.MHS.*',
        # '2A-ATMS-NOAA20-CLIM': '2A-CLIM.NOAA20.ATMS.*',
        # '2A-ATMS-NPP-CLIM': '2A-CLIM.NPP.ATMS.*',
        # Using JMA's GANAL as environment ancillary data
        "2A-GMI": "2A.GPM.GMI.*",
        "2A-TMI": "2A.TMI.TRMM.*",
        "2A-SSMI-F08": "2A.F08.SSMI.*",
        "2A-SSMI-F10": "2A.F10.SSMI.*",
        "2A-SSMI-F11": "2A.F11.SSMI.*",
        "2A-SSMI-F13": "2A.F13.SSMI.*",
        "2A-SSMI-F14": "2A.F14.SSMI.*",
        "2A-SSMI-F15": "2A.F15.SSMI.*",
        "2A-SSMIS-F16": "2A.F16.SSMIS.*",
        "2A-SSMIS-F17": "2A.F17.SSMIS.*",
        "2A-SSMIS-F18": "2A.F18.SSMIS.*",
        "2A-SSMIS-F19": "2A.F18.SSMIS.*",
        "2A-AMSR2-GCOMW1": "2A.GCOMW1.AMSR2.*",
        "2A-AMSRE-AQUA": "2A.AQUA.AMSRE.*",
        "2A-AMSUB-NOAA15": "2A.AMSUB.NOAA15.*",
        "2A-AMSUB-NOAA16": "2A.AMSUB.NOAA16.*",
        "2A-AMSUB-NOAA17": "2A.AMSUB.NOAA17.*",
        "2A-MHS-METOPA": "2A.METOPA.MHS.*",
        "2A-MHS-METOPB": "2A.METOPB.MHS.*",
        "2A-MHS-METOPC": "2A.METOPC.MHS.*",
        "2A-MHS-NOAA18": "2A.NOAA18.MHS.*",
        "2A-MHS-NOAA19": "2A.NOAA19.MHS.*",
        "2A-ATMS-NOAA20": "2A.NOAA20.ATMS.*",
        "2A-ATMS-NPP": "2A.NPP.ATMS.*",
    }
    return GPM_dict


def GPM_PMW_2A_GPROF_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM PMW GPROF 2A products."""
    GPM_dict = {  # Using JMA's GANAL as environment ancillary data
        "2A-GMI": "2A.GPM.GMI.*",
        "2A-SSMI-F16": "2A.F16.SSMIS.*",
        "2A-SSMI-F17": "2A.F17.SSMIS.*",
        "2A-SSMI-F18": "2A.F18.SSMIS.*",
        "2A-ASMR2-GCOMW1": "2A.GCOMW1.ASMR2.*",
        "2A-MHS-METOPB": "2A.METOPB.MHS.*",
        "2A-MHS-METOPC": "2A.METOPC.MHS.*",
        "2A-MHS-NOAA19": "2A.NOAA19.MHS.*",
        "2A-ATMS-NOAA20": "2A.NOAA20.ATMS.*",
        "2A-ATMS-NPP": "2A.NPP.ATMS.*",
    }
    return GPM_dict


def GPM_PMW_2A_PRPS_RS_pattern_dict():
    """Return the filename pattern* associated to GPM PMW PRPS 2A products."""
    GPM_dict = {  # Using ERA-I as environment ancillary data
        "2A-SAPHIR-MT1-CLIM": "2A-CLIM.SAPHIR.MT1.*",
        # Using JMA's GANAL as environment ancillary data
        "2A-SAPHIR-MT1": "2A.SAPHIR.MT1.*",
    }
    return GPM_dict


def GPM_PMW_2A_PRPS_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM PMW PRPS 2A products."""
    GPM_dict = {"2A-SAPHIR-MT1": "2A.SAPHIR.MT1.*"}
    return GPM_dict


def GPM_PMW_RS_pattern_dict():
    """Return the filename pattern* associated to all PMW RS products."""
    GPM_dict = GPM_PMW_1A_RS_pattern_dict()
    GPM_dict.update(GPM_PMW_1B_RS_pattern_dict())
    GPM_dict.update(GPM_PMW_1C_RS_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_GPROF_RS_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_PRPS_RS_pattern_dict())
    return GPM_dict


def GPM_PMW_NRT_pattern_dict():
    """Return the filename pattern* associated to all PMW NRT products."""
    GPM_dict = GPM_PMW_1B_NRT_pattern_dict()
    GPM_dict.update(GPM_PMW_1C_NRT_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_GPROF_NRT_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_PRPS_NRT_pattern_dict())
    return GPM_dict


##----------------------------------------------------------------------------.
### IMERG
def GPM_IMERG_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM IMERG products."""
    GPM_dict = {
        "IMERG-ER": "3B-HHR-E.MS.MRG*",
        "IMERG-LR": "3B-HHR-L.MS.MRG*",
    }
    return GPM_dict


def GPM_IMERG_RS_pattern_dict():
    """Return the filename pattern* associated to GPM IMERG products."""
    GPM_dict = {"IMERG-FR": "3B-HHR.MS.MRG.*"}
    return GPM_dict


def GPM_IMERG_pattern_dict():
    """Return the filename pattern* associated to GPM IMERG products."""
    GPM_dict = GPM_IMERG_NRT_pattern_dict()
    GPM_dict.update(GPM_IMERG_RS_pattern_dict())
    return GPM_dict


##----------------------------------------------------------------------------.
#### GPM Product Levels
def GPM_1B_RS_pattern_dict():
    """Return the filename pattern* associated to GPM 1B RS products."""
    GPM_dict = GPM_RADAR_1B_RS_pattern_dict()
    GPM_dict.update(GPM_PMW_1B_RS_pattern_dict())
    return GPM_dict


def GPM_1B_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM 1B NRT products."""
    GPM_dict = GPM_PMW_1B_NRT_pattern_dict()  # GPM_RADAR_1B_NRT_pattern_dict()
    return GPM_dict


def GPM_2A_RS_pattern_dict():
    """Return the filename pattern* associated to GPM 2A RS products."""
    GPM_dict = GPM_RADAR_2A_RS_pattern_dict()
    GPM_dict.update(GPM_PMW_2A_GPROF_RS_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_PRPS_RS_pattern_dict())
    return GPM_dict


def GPM_2A_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM 2A NRT products."""
    GPM_dict = GPM_RADAR_2A_NRT_pattern_dict()
    GPM_dict.update(GPM_PMW_2A_GPROF_NRT_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_PRPS_NRT_pattern_dict())
    return GPM_dict


def GPM_2B_RS_pattern_dict():
    """Return the filename pattern* associated to GPM CMB NRT products."""
    GPM_dict = GPM_CMB_2B_RS_pattern_dict()
    return GPM_dict


def GPM_2B_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM CMB NRT products."""
    GPM_dict = GPM_CMB_2B_NRT_pattern_dict()
    return GPM_dict


# ----------------------------------------------------------------------------.
### RS vs. NRT
def GPM_RS_products_pattern_dict():
    """Return the filename pattern* associated to all GPM RS products."""
    GPM_dict = GPM_IMERG_RS_pattern_dict()
    GPM_dict.update(GPM_RADAR_RS_pattern_dict())
    GPM_dict.update(GPM_PMW_RS_pattern_dict())
    GPM_dict.update(GPM_CMB_RS_pattern_dict())
    return GPM_dict


def GPM_NRT_products_pattern_dict():
    """Return the filename pattern* associated to all GPM NRT products."""
    GPM_dict = GPM_IMERG_NRT_pattern_dict()
    GPM_dict.update(GPM_RADAR_NRT_pattern_dict())
    GPM_dict.update(GPM_PMW_NRT_pattern_dict())
    GPM_dict.update(GPM_CMB_NRT_pattern_dict())
    return GPM_dict


def GPM_products_pattern_dict():
    """Return the filename pattern* associated to all GPM products."""
    GPM_dict = GPM_NRT_products_pattern_dict()
    GPM_dict.update(GPM_RS_products_pattern_dict())
    return GPM_dict
