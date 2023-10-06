import datetime
import h5py
import os

from dateutil.relativedelta import relativedelta

from gpm_api.configs import get_gpm_username, get_gpm_password
from gpm_api.dataset.granule import open_granule
from gpm_api.io import download, products
from gpm_api.io.pps import find_pps_filepaths


# Create asset directories #####################################################


assets_dir_path = "assets"
raw_assets_dir_path = os.path.join(assets_dir_path, "raw")
cut_assets_dir_path = os.path.join(assets_dir_path, "cut")
processed_assets_dir_path = os.path.join(assets_dir_path, "processed")

# Change current working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create the assets directories
for path in [assets_dir_path, raw_assets_dir_path, cut_assets_dir_path, processed_assets_dir_path]:
    os.makedirs(path, exist_ok=True)


# Download raw assets ##########################################################


def download_raw_assets(products: dict) -> list[str]:
    pps_filepaths = list_files_to_download(products)
    filenames = [pps_filepath.split("/")[-1] for pps_filepath in pps_filepaths]
    local_filepaths = [os.path.join(raw_assets_dir_path, filename) for filename in filenames]

    print("Downloading raw assets...")

    download._download_files(
        pps_filepaths,
        local_filepaths,
        get_gpm_username(),
        get_gpm_password(),
        verbose=True,
    )


def list_files_to_download(products: dict) -> list[str]:
    pps_filepaths = list_pps_filepaths(products)
    missing_pps_filepaths = []

    # Filter out files that have already been downloaded
    for pps_filepath in pps_filepaths:
        filename = pps_filepath.split("/")[-1]
        if not os.path.exists(os.path.join(raw_assets_dir_path, filename)):
            missing_pps_filepaths.append(pps_filepath)

    return missing_pps_filepaths


def list_pps_filepaths(products: dict) -> list[str]:
    pps_filepaths = []

    print("TODO: add start_date and end_date to all products")
    for product, product_info in products.items():
        if "start_date" not in product_info:  # TODO: add start_date and end_date to all products
            continue

        for product_type in product_info["product_types"]:
            pps_filepath = find_first_pps_filepath(
                product, product_type, product_info["start_date"]
            )
            if pps_filepath is not None:
                pps_filepaths.append(pps_filepath)

    return pps_filepaths


def find_first_pps_filepath(
    product: str, product_type: str, start_date: datetime.date
) -> str | None:
    start_time = datetime.datetime(start_date.year, start_date.month, start_date.day)
    end_time = start_time + relativedelta(days=1)

    pps_filepaths = find_pps_filepaths(
        product,
        start_time
        + relativedelta(days=1),  # gets extended to (start_time - 1 day) in find_pps_filepaths
        end_time,
        product_type=product_type,
    )

    if len(pps_filepaths) == 0:
        print(f"No PPS files found for {product}")
        return None

    return pps_filepaths[0]


products = products.get_info_dict()
filenames = download_raw_assets(products)


# Cut raw assets ##############################################################


def _get_fixed_dimensions():
    """Dimensions over which to not subset the GPM HDF5 files."""
    fixed_dims = [
        # Elevations / Range
        "nBnPSD",
        "nBnPSDhi",
        "nBnEnv",
        "nbinMS",
        "nbinHS",
        "nbinFS",
        "nbin",
        # Radar frequency
        "nKuKa",
        "nfreq",
        # PMW frequency
        "nemiss",
        "nchan1",
        "nchan2",
        "nchannel1",
        "nchannel2",
        "nchannel3",
        "nchannel4",
        "nchannel5",
        "nchannel6",
    ]
    return fixed_dims


def _get_subset_shape_chunks(h5_obj, subset_size=5):
    """Return the shape and chunks of the subsetted HDF5 file."""
    dimnames = h5_obj.attrs.get("DimensionNames", None)
    fixed_dims = _get_fixed_dimensions()
    chunks = h5_obj.chunks
    if dimnames is not None:
        # Get dimension names list
        dimnames = dimnames.decode().split(",")
        # Get dimension shape
        shape = h5_obj.shape
        # Create dimension dictionary
        dict_dims = dict(zip(dimnames, shape))
        # Create chunks dictionary
        dict_chunks = dict(zip(dimnames, chunks))
        # Define subset shape and chunks
        subset_shape = []
        subset_chunks = []
        for dim, src_size in dict_dims.items():
            chunk = dict_chunks[dim]
            if dim in fixed_dims:
                subset_shape.append(src_size)
                subset_chunks.append(chunk)
            else:
                subset_size = min(subset_size, src_size)
                subset_chunk = min(chunk, subset_size)
                subset_shape.append(subset_size)
                subset_chunks.append(subset_chunk)

        # Determine subset shape
        subset_shape = tuple(subset_shape)
        subset_chunks = tuple(subset_chunks)
    else:
        subset_shape = h5_obj.shape
        subset_chunks = h5_obj.chunks
    return subset_shape, subset_chunks


def _copy_attrs(src_h5_obj, dst_h5_obj):
    """Copy attributes from the source file to the destination file."""
    for key, value in src_h5_obj.attrs.items():
        dst_h5_obj.attrs[key] = value


def _copy_datasets(src_group, dst_group, subset_size=5):
    for name, h5_obj in src_group.items():
        if isinstance(h5_obj, h5py.Dataset):
            # Determine the subset shape (2 indices per dimension)
            subset_shape, subset_chunks = _get_subset_shape_chunks(h5_obj, subset_size=subset_size)

            # Create a new dataset in the subset group with the subset shape
            subset_dataset = dst_group.create_dataset(
                name, subset_shape, dtype=h5_obj.dtype, chunks=subset_chunks
            )

            # Copy data from the src_h5_obj dataset to the subset dataset
            subset_dataset[:] = h5_obj[tuple(slice(0, size) for size in subset_shape)]

            # Copy attributes from the src_h5_obj dataset to the subset dataset
            _copy_attrs(h5_obj, subset_dataset)

            # Copy encoding information
            if h5_obj.compression is not None and "compression" in h5_obj.compression:
                subset_dataset.compression = h5_obj.compression
                subset_dataset.compression_opts = h5_obj.compression_opts

        elif isinstance(h5_obj, h5py.Group):
            # If the h5_obj is a group, create a corresponding group in the subset file and copy its datasets recursively
            subgroup = dst_group.create_group(name)
            # Copy group attributes
            _copy_attrs(h5_obj, subgroup)
            _copy_datasets(h5_obj, subgroup, subset_size=subset_size)


def create_test_hdf5(src_fpath, dst_fpath):
    # Open source HDF5 file
    src_file = h5py.File(src_fpath, "r")

    # Create empty HDF5 file
    dst_file = h5py.File(dst_fpath, "w")

    # Write a subset of the source HDF5 groups and leafs into the new HDF5 file
    _copy_datasets(src_file, dst_file, subset_size=10)

    # Write attributes from the source HDF5 root group to the new HDF5 file root group
    _copy_attrs(src_file, dst_file)

    # Close connection
    src_file.close()
    dst_file.close()


def cut_raw_assets():
    filenames = os.listdir(raw_assets_dir_path)

    for filename in filenames:
        print(f"Cutting {filename}")
        raw_asset_filepath = os.path.join(raw_assets_dir_path, filename)
        cut_asset_filepath = os.path.join(cut_assets_dir_path, filename)
        try:
            create_test_hdf5(raw_asset_filepath, cut_asset_filepath)
        except Exception as e:
            print(f"Failed to cut {filename}: {e}")


cut_raw_assets()


# Open assets with gpm_api and save as netCDF ##################################


def open_and_save_processed_assets(products: dict):
    filenames = os.listdir(cut_assets_dir_path)

    for product, product_info in products.items():
        filename = find_filename_from_pattern(filenames, product_info["pattern"])
        if filename is None:
            print(f"Could not find {product} file")
            continue

        process_asset(filename, product_info["scan_modes_v7"])


def find_filename_from_pattern(filenames: list[str], pattern: str) -> str | None:
    for filename in filenames:
        if pattern.rstrip("*").rstrip("\\d-") in filename:  # TODO: clarify specs of pattern
            return filename

    return None


def process_asset(filename: str, scan_modes: list[str]):
    asset_filepath = os.path.join(cut_assets_dir_path, filename)
    processed_dir_path = os.path.join(processed_assets_dir_path, os.path.splitext(filename)[0])
    os.makedirs(processed_dir_path, exist_ok=True)

    for scan_mode in scan_modes:
        print(f"Processing {filename} with scan mode {scan_mode}")
        processed_asset_filepath = os.path.join(processed_dir_path, f"{scan_mode}.nc")
        try:
            ds = open_granule(asset_filepath, scan_mode)
            ds.to_netcdf(processed_asset_filepath)
        except Exception as e:
            print(f"Failed to process {filename} with scan mode {scan_mode}: {e}")


open_and_save_processed_assets(products)
