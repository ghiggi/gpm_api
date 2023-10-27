import datetime
import h5py
import os

from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from gpm_api.dataset.granule import open_granule
from gpm_api.io import download, products as gpm_products
from gpm_api.io.find import find_filepaths


RAW_DIRNAME = "raw"
CUT_DIRNAME = "cut"
PROCESSED_DIRNAME = "processed"
KEPT_PRODUCT_TYPES = ["RS"]


# Create granule directories ###################################################


# Create the granules directory
granules_dir_path = "test_granule_data"
os.makedirs(granules_dir_path, exist_ok=True)

# Change current working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Check available versions and scan_modes ######################################


def check_scan_mode_versions(products: dict):
    for product, info in products.items():
        version = info["available_versions"][-1]
        if f"V{version}" not in info["scan_modes"]:
            print(
                f"WARNING: {product} does not have scan modes listed for latest version {version}"
            )


products = gpm_products.get_info_dict()
check_scan_mode_versions(products)


# Download raw granules ########################################################


def download_raw_granules(products: dict) -> None:
    print("Listing files to download...")
    print('Please ignore the "No data found" warnings')

    pps_filepaths = list_files_to_download(products)
    filenames = [pps_filepath.split("/")[-1] for pps_filepath in pps_filepaths]
    product_basenames = [os.path.splitext(filename)[0] for filename in filenames]
    local_filepaths = [
        os.path.join(granules_dir_path, product_basename, RAW_DIRNAME, product_basename + ".HDF5")
        for product_basename in product_basenames
    ]

    print("Downloading raw granules...")

    download._download_files(
        pps_filepaths,
        local_filepaths,
        storage="pps",
        transfer_tool="wget",
        verbose=True,
    )


def list_files_to_download(products: dict) -> list[str]:
    pps_filepaths = list_pps_filepaths(products)
    missing_pps_filepaths = []

    # Filter out files that have already been downloaded
    for pps_filepath in pps_filepaths:
        filename = pps_filepath.split("/")[-1]
        product_basename = os.path.splitext(filename)[0]
        if not os.path.exists(os.path.join(granules_dir_path, product_basename, RAW_DIRNAME)):
            missing_pps_filepaths.append(pps_filepath)

    return missing_pps_filepaths


def list_pps_filepaths(products: dict) -> list[str]:
    pps_filepaths = []

    for product, product_info in tqdm(products.items()):
        if "start_time" not in product_info:
            print(f"Skipping {product}: no start_time was provided")
            continue

        for product_type in product_info["product_types"]:
            if product_type not in KEPT_PRODUCT_TYPES:
                continue

            pps_filepath = find_first_pps_filepath(
                product, product_type, product_info["start_time"]
            )
            if pps_filepath is not None:
                pps_filepaths.append(pps_filepath)

    return pps_filepaths


def find_first_pps_filepath(
    product: str, product_type: str, start_time: datetime.datetime
) -> str | None:
    end_time = start_time + relativedelta(days=1)

    pps_filepaths = find_filepaths(
        storage="pps",
        product=product,
        start_time=start_time,
        # start_time gets extended to (start_time - 1 day) in find_filepaths.
        # May produce "No data found" warning
        end_time=end_time,
        product_type=product_type,
    )

    if len(pps_filepaths) == 0:
        print(f"WARNING: No PPS files found for {product}")
        return None

    return pps_filepaths[0]


download_raw_granules(products)


# Cut raw granules #############################################################


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


def cut_raw_granules():
    product_basenames = os.listdir(granules_dir_path)

    for product_basename in product_basenames:
        print(f"Cutting {product_basename}")
        raw_filepath = os.path.join(
            granules_dir_path, product_basename, RAW_DIRNAME, product_basename + ".HDF5"
        )
        cut_dir_path = os.path.join(granules_dir_path, product_basename, CUT_DIRNAME)
        cut_filepath = os.path.join(cut_dir_path, product_basename + ".HDF5")
        os.makedirs(cut_dir_path, exist_ok=True)
        try:
            create_test_hdf5(raw_filepath, cut_filepath)
        except Exception as e:
            print(f"Failed to cut {product_basename}: {e}")


cut_raw_granules()


# Open granules with gpm_api and save as netCDF ################################


def open_and_save_processed_granules(products: dict):
    product_basenames = os.listdir(granules_dir_path)

    for product, product_info in products.items():
        if "start_time" not in product_info:
            continue

        product_basename = find_product_basename_from_pattern(
            product_basenames, product_info["pattern"]
        )
        if product_basename is None:
            print(f"Could not find {product} file")
            continue

        version = gpm_products.get_last_product_version(product)
        scan_modes = product_info["scan_modes"][f"V{version}"]
        process_granule(product_basename, scan_modes)


def find_product_basename_from_pattern(product_basenames: list[str], pattern: str) -> str | None:
    for product_basename in product_basenames:
        if pattern.rstrip("*").rstrip("\\d-") in product_basename:  # TODO: clarify specs of pattern
            return product_basename

    return None


def process_granule(product_basename: str, scan_modes: list[str]):
    granule_path = os.path.join(
        granules_dir_path, product_basename, CUT_DIRNAME, product_basename + ".HDF5"
    )
    processed_dir_path = os.path.join(granules_dir_path, product_basename, PROCESSED_DIRNAME)
    os.makedirs(processed_dir_path, exist_ok=True)

    for scan_mode in scan_modes:
        print(f"Processing {product_basename} with scan mode {scan_mode}")
        processed_granule_filepath = os.path.join(processed_dir_path, f"{scan_mode}.nc")
        try:
            ds = open_granule(granule_path, scan_mode)
            ds.to_netcdf(processed_granule_filepath)
        except Exception as e:
            print(f"Failed to process {product_basename} with scan mode {scan_mode}: {e}")


open_and_save_processed_granules(products)
