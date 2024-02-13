def get_encoding_dict():
    """Get encoding dictionary for 2A-<RADAR> products."""
    # --------------------------------------------------------------------------.
    #### 3D Fields
    # variable = "zFactorMeasured"
    # variable = "zFactorFinal"
    # variable = "precipWater"  # [0-65] 20 000 scale_factor=0.001
    # variable = "precipRate"   # [0-300] scale_factor = 0.01
    # variable = "phase"        # [0,1,2] uint8, fill_value=255
    # variable = "paramDSD"     # [0 - 100] scale_factro = 0.01 (if not above 60, also 0.001)
    # variable = 'DFRforward1'  # [-25?, 20] --> add_offset = -30 --> [0-60] scale_factor = 0.001
    # variable = 'airTemperature' # [150, 333] --> add_offset = 150 --> [0-183] scale_factor = 0.01
    # variable = 'attenuationNP'  # [0, 0.502] --> scale_factor = 0.001 --> [0-502] --> uint16
    # variable = 'epsilon'   # [0.258, 3.5] --> scale_factor = 0.001
    # variable = 'flagEcho' # [0-103] , uint8 , 255 fill
    # variable = 'flagSLV'  # ([-128.,-64.,  0.,  5., 6.,  7., 9., 10., 38.,  39.,   42.]. add_offset=-128, uint8,
    variables = [
        "DFRforward1",
        "airTemperature",
        "attenuationNP",
        "epsilon",
        "flagEcho",
        "flagSLV",
        "paramDSD",
        "phase",
        "precipRate",
        "precipWater",
        "zFactorFinal",
        "zFactorMeasured",
        "height",
    ]
    encoding_dict = {}
    for var in variables:
        encoding_dict[var] = {}

    # zFactorFinal
    encoding_dict["zFactorFinal"]["dtype"] = "uint16"
    encoding_dict["zFactorFinal"]["scale_factor"] = 0.01
    encoding_dict["zFactorFinal"]["add_offset"] = -20
    encoding_dict["zFactorFinal"]["_FillValue"] = 65535
    # zFactorMeasured
    encoding_dict["zFactorMeasured"]["dtype"] = "uint16"
    encoding_dict["zFactorMeasured"]["scale_factor"] = 0.01
    encoding_dict["zFactorMeasured"]["add_offset"] = -20
    encoding_dict["zFactorMeasured"]["_FillValue"] = 65535
    # precipWater
    encoding_dict["precipWater"]["dtype"] = "uint16"
    encoding_dict["precipWater"]["scale_factor"] = 0.001
    encoding_dict["precipWater"]["add_offset"] = 0.0
    encoding_dict["precipWater"]["_FillValue"] = 65535
    # precipRate
    encoding_dict["precipRate"]["dtype"] = "uint16"
    encoding_dict["precipRate"]["scale_factor"] = 0.01
    encoding_dict["precipRate"]["add_offset"] = 0.0
    encoding_dict["precipRate"]["_FillValue"] = 65535
    # phase
    encoding_dict["phase"]["dtype"] = "uint8"
    encoding_dict["phase"]["_FillValue"] = 255
    # paramDSD
    encoding_dict["paramDSD"]["dtype"] = "uint16"
    encoding_dict["paramDSD"]["scale_factor"] = 0.01  # maybe 0.001
    encoding_dict["paramDSD"]["add_offset"] = 0.0
    encoding_dict["paramDSD"]["_FillValue"] = 65535
    # DFRforward1
    encoding_dict["DFRforward1"]["dtype"] = "uint16"
    encoding_dict["DFRforward1"]["scale_factor"] = 0.001
    encoding_dict["DFRforward1"]["add_offset"] = -30  # ?
    encoding_dict["DFRforward1"]["_FillValue"] = 65535
    # airTemperature
    encoding_dict["airTemperature"]["dtype"] = "uint16"
    encoding_dict["airTemperature"]["scale_factor"] = 0.01
    encoding_dict["airTemperature"]["add_offset"] = 0.0
    encoding_dict["airTemperature"]["_FillValue"] = 65535
    # attenuationNP
    encoding_dict["attenuationNP"]["dtype"] = "uint16"
    encoding_dict["attenuationNP"]["scale_factor"] = 0.001
    encoding_dict["attenuationNP"]["add_offset"] = 0.0
    encoding_dict["attenuationNP"]["_FillValue"] = 65535
    # epsilon
    encoding_dict["epsilon"]["dtype"] = "uint16"
    encoding_dict["epsilon"]["scale_factor"] = 0.001
    encoding_dict["epsilon"]["add_offset"] = 0.0
    encoding_dict["epsilon"]["_FillValue"] = 65535
    # flagEcho
    encoding_dict["flagEcho"]["dtype"] = "uint8"
    encoding_dict["flagEcho"]["_FillValue"] = 255
    # flagSLV
    encoding_dict["epsilon"]["dtype"] = "uint8"
    encoding_dict["epsilon"]["scale_factor"] = 1.0
    encoding_dict["epsilon"]["add_offset"] = -128.0
    encoding_dict["epsilon"]["_FillValue"] = 255
    # height [-62? - 22'000]
    encoding_dict["height"]["dtype"] = "uint16"
    encoding_dict["height"]["scale_factor"] = 1.0
    encoding_dict["height"]["add_offset"] = -200
    encoding_dict["height"]["_FillValue"] = 65535

    # --------------------------------------------------------------------------.
    #### Flag variables
    variables = [
        "flagAnvil",  # [0, 1, 2]
        "flagBB",  # [0, 1, 2, 3]
        "flagGraupelHail",  # [0, 1]
        "flagHail",  # [0, 1]
        "flagHeavyIcePrecip",  #  [4, 8, 12, 16, 24, 32, 40]
        "flagPrecip",  # [0, 1, 10, 11]
        # 'flagInversion',
        # 'flagMLquality',
        # 'flagShallowRain',
        # 'flagSigmaZeroSaturation',
        "flagSurfaceSnowfall",  #  [0, 1]
    ]
    for var in variables:
        encoding_dict[var] = {}
        encoding_dict[var]["dtype"] = "uint8"
        encoding_dict[var]["_FillValue"] = 255

    return encoding_dict
