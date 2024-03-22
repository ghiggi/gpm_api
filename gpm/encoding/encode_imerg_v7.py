# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""This module contains encodings for the IMERG products."""


def get_encoding_dict():
    """Get encoding dictionary for IMERG-FR V7 products."""
    variables = [
        "precipitation",
        "randomError",
        "probabilityLiquidPrecipitation",
        "precipitationQualityIndex",
        "MWprecipitation",
        "MWprecipSource",
        "MWobservationTime",
        "IRprecipitation",
        "IRinfluence",
        "precipitationUncal",
    ]
    encoding_dict = {}
    for var in variables:
        encoding_dict[var] = {}
    # precipitation
    encoding_dict["precipitation"]["dtype"] = "uint16"
    encoding_dict["precipitation"]["scale_factor"] = 0.01
    encoding_dict["precipitation"]["add_offset"] = 0.0
    encoding_dict["precipitation"]["_FillValue"] = 65535
    # precipitationUncal
    encoding_dict["precipitationUncal"]["dtype"] = "uint16"
    encoding_dict["precipitationUncal"]["scale_factor"] = 0.01
    encoding_dict["precipitationUncal"]["add_offset"] = 0.0
    encoding_dict["precipitationUncal"]["_FillValue"] = 65535
    # IRprecipitation
    encoding_dict["IRprecipitation"]["dtype"] = "uint16"
    encoding_dict["IRprecipitation"]["scale_factor"] = 0.01
    encoding_dict["IRprecipitation"]["add_offset"] = 0.0
    encoding_dict["IRprecipitation"]["_FillValue"] = 65535
    # MWprecipitation
    encoding_dict["MWprecipitation"]["dtype"] = "uint16"
    encoding_dict["MWprecipitation"]["scale_factor"] = 0.01
    encoding_dict["MWprecipitation"]["add_offset"] = 0.0
    encoding_dict["MWprecipitation"]["_FillValue"] = 65535
    # precipitationQualityIndex
    # --> Should first round to 3 decimals ...
    encoding_dict["precipitationQualityIndex"]["dtype"] = "uint16"
    encoding_dict["precipitationQualityIndex"]["scale_factor"] = 0.0001
    encoding_dict["precipitationQualityIndex"]["add_offset"] = 0.0
    encoding_dict["precipitationQualityIndex"]["_FillValue"] = 65535
    # randomError
    encoding_dict["randomError"]["dtype"] = "uint16"
    encoding_dict["randomError"]["scale_factor"] = 0.01
    encoding_dict["randomError"]["add_offset"] = 0.0
    encoding_dict["randomError"]["_FillValue"] = 65535
    # probabilityLiquidPrecipitation
    encoding_dict["probabilityLiquidPrecipitation"]["dtype"] = "uint8"
    encoding_dict["probabilityLiquidPrecipitation"]["scale_factor"] = 1.0
    encoding_dict["probabilityLiquidPrecipitation"]["add_offset"] = 0.0
    encoding_dict["probabilityLiquidPrecipitation"]["_FillValue"] = 255
    # MWobservationTime
    encoding_dict["MWobservationTime"]["dtype"] = "uint8"
    encoding_dict["MWobservationTime"]["scale_factor"] = 1.0
    encoding_dict["MWobservationTime"]["add_offset"] = 0.0
    encoding_dict["MWobservationTime"]["_FillValue"] = 255
    # IRinfluence
    encoding_dict["MWobservationTime"]["dtype"] = "uint8"
    encoding_dict["MWobservationTime"]["scale_factor"] = 1.0
    encoding_dict["MWobservationTime"]["add_offset"] = 0.0
    encoding_dict["MWobservationTime"]["_FillValue"] = 255
    # MWprecipSource
    encoding_dict["MWprecipSource"]["dtype"] = "uint8"
    encoding_dict["MWprecipSource"]["scale_factor"] = 1.0
    encoding_dict["MWprecipSource"]["add_offset"] = 0.0
    encoding_dict["MWprecipSource"]["_FillValue"] = 255
    return encoding_dict
