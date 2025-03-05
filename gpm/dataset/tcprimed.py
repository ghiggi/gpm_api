import re

import pyproj
import xarray as xr

from gpm.utils.pmw import PMWFrequency


def _get_freq_string(pmw_freq):
    freq = pmw_freq.replace("TB_A", "").replace("TB_B", "").replace("TB_", "")
    match = re.search(r"([A-Za-z]+)$", freq)
    polarization = match.group(1)
    freq = freq.replace(polarization, "")
    # Identify center frequency and offset
    split_list = freq.split("_")
    center_frequency = split_list[0]
    # Identify offset
    offset = split_list[1] if len(split_list) == 2 else None
    return PMWFrequency(center_frequency=center_frequency, polarization=polarization, offset=offset).to_string()


def convert_passive_microwave(ds):
    variables = [var for var in ds.data_vars if var.startswith("TB")]
    da_tb = ds[variables].to_array(dim="pmw_frequency")
    pmw_frequencies = da_tb["pmw_frequency"].data
    pmw_frequencies = [_get_freq_string(freq) for freq in pmw_frequencies]
    da_tb = da_tb.assign_coords({"pmw_frequency": pmw_frequencies})
    ds["Tc"] = da_tb
    platform = ds.attrs["platform"]
    instrument = ds.attrs["instrument"]
    gpm_api_product = f"1C-{instrument}" if instrument in ["GMI", "TMI"] else f"1C-{instrument}-{platform}"
    ds.attrs["gpm_api_product"] = gpm_api_product
    return ds


def ensure_standard_longitude_values(da_lon):
    da_lon = ((da_lon + 180) % 360) - 180
    da_lon.attrs["valid_range"] = [-180, 180]
    return da_lon


def open_granule_tcprimed(filepath, chunks={}, **kwargs):
    from gpm.dataset.crs import set_dataset_crs

    # Open datatree
    dt = xr.open_datatree(filepath, chunks=chunks, **kwargs)

    # Ensure standard longitude
    dt["overpass_storm_metadata"]["storm_longitude"] = ensure_standard_longitude_values(
        dt["overpass_storm_metadata"]["storm_longitude"],
    )

    # Reformat "passive_microwave" node
    if "passive_microwave" in dt:
        scan_modes = list(dt["passive_microwave"])
        for scan_mode in scan_modes:
            # Retrieve dataset
            ds = dt["passive_microwave"][scan_mode].to_dataset()
            # Rename dimensions
            ds = ds.rename_dims({"scan": "along_track", "pixel": "cross_track"})
            # Add GPM-API Tc variable
            ds = convert_passive_microwave(ds)
            # Rename x and y coordinates
            ds = ds.rename_vars({"x": "x_c", "y": "y_c"})
            # Ensure longitude between -180 and 180
            ds["longitude"] = ensure_standard_longitude_values(ds["longitude"])
            # Ensure (cross_track, along_track) dimension order
            ds = ds.transpose("cross_track", "along_track", ...)
            # Add GPM-API scan 'time' coordinate
            ds = ds.drop_vars("time").set_coords("ScanTime").rename_vars({"ScanTime": "time"})
            # Add CRS information
            crs = pyproj.CRS(proj="longlat", ellps="WGS84")
            ds = set_dataset_crs(ds, crs=crs, grid_mapping_name="crsWGS84", inplace=False)
            # Update datatree
            dt["passive_microwave"][scan_mode] = ds

    # -------------------------------------------------
    # Reformat "GPROF" node
    if "GPROF" in dt:
        # Retrieve dataset
        ds = dt["GPROF"]["S1"].to_dataset()
        # Rename dimensions
        ds = ds.rename_dims({"scan": "along_track", "pixel": "cross_track", "hgtTopLayer": "height"})
        # Rename x and y coordinates
        ds = ds.rename_vars({"x": "x_c", "y": "y_c"})
        # Ensure longitude between -180 and 180
        ds["longitude"] = ensure_standard_longitude_values(ds["longitude"])
        # Ensure (cross_track, along_track) dimension order
        ds = ds.transpose("cross_track", "along_track", ...)
        # Add CRS information
        crs = pyproj.CRS(proj="longlat", ellps="WGS84")
        ds = set_dataset_crs(ds, crs=crs, grid_mapping_name="crsWGS84", inplace=False)
        # Add GPM-API product attribute
        platform = ds.attrs["platform"]
        instrument = ds.attrs["instrument"]
        gpm_api_product = f"2A-{instrument}" if instrument in ["GMI", "TMI"] else f"2A-{instrument}-{platform}"
        ds.attrs["gpm_api_product"] = gpm_api_product
        # Update datatree
        dt["GPROF"]["S1"] = ds

    # -------------------------------------------------
    # Reformat "radar_radiometer" node
    if "radar_radiometer" in dt:
        scan_modes = [
            scan_mode for scan_mode in list(dt["radar_radiometer"]) if scan_mode in ["KuGMI", "KuKaGMI", "KuTMI"]
        ]
        for scan_mode in scan_modes:
            # Retrieve dataset
            ds = dt["radar_radiometer"][scan_mode].to_dataset()
            # Rename dimensions
            ds = ds.rename_dims({"scan": "along_track", "beam": "cross_track"}).rename_vars({"binHeight": "height"})
            # Rename x and y coordinates
            ds = ds.rename_vars({"x": "x_c", "y": "y_c"})
            # Ensure longitude between -180 and 180
            ds["longitude"] = ensure_standard_longitude_values(ds["longitude"])
            # Ensure (cross_track, along_track) dimension order
            ds = ds.transpose("cross_track", "along_track", ...)
            # Add CRS information
            crs = pyproj.CRS(proj="longlat", ellps="WGS84")
            ds = set_dataset_crs(ds, crs=crs, grid_mapping_name="crsWGS84", inplace=False)
            # Update datatree
            dt["radar_radiometer"][scan_mode] = ds

    # -------------------------------------------------
    # Reformat "infrared" node
    if "infrared" in dt:
        # Retrieve dataset
        ds = dt["infrared"].to_dataset()
        # Set lon and lat as 1D coordinates
        ds = ds.assign_coords({"longitude_1d": ds["longitude"].isel(ny=0), "latitude_1d": ds["latitude"].isel(nx=0)})
        ds = ds.drop_vars(["longitude", "latitude"])
        # Rename x and y coordinates
        ds = ds.rename_vars({"x": "x_c", "y": "y_c"})
        # Rename dimensions
        ds = ds.rename_dims({"nx": "longitude", "ny": "latitude"})
        ds = ds.rename_vars({"longitude_1d": "longitude", "latitude_1d": "latitude"})
        # Ensure longitude between -180 and 180
        ds["longitude"] = ensure_standard_longitude_values(ds["longitude"])
        # Set CRS
        crs = pyproj.CRS(proj="longlat", ellps="WGS84")
        ds = set_dataset_crs(ds, crs=crs, grid_mapping_name="crsWGS84", inplace=False)
        dt["infrared"] = ds

    return dt
