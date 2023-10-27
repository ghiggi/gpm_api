import pytest
import os
import datetime
import ftplib
from typing import Any, List, Dict
from pytest_mock.plugin import MockerFixture
from gpm_api.io import find
from gpm_api.io import download as dl
from gpm_api.io.products import available_products
from gpm_api.utils.warnings import GPMDownloadWarning
from gpm_api import configs


def test_construct_curl_pps_cmd(
    server_paths: Dict[str, Dict[str, Any]],
    tmpdir: str,
) -> None:
    """Test that the curl command constructor works as expected

    `disk_path` relates to a file on the disk
    """

    # Use datetime as path as to be unique to every test
    disk_path = os.path.join(
        tmpdir,
        datetime.datetime.utcnow().isoformat().replace(":", "-"),
        "curl",
        "output_file.hdf5",
    )
    assert not os.path.exists(
        os.path.dirname(disk_path)
    ), f"Folder {os.path.dirname(disk_path)} already exists"

    curl_truth = (
        "curl --verbose --ipv4 --insecure "
        "--user {username}:{password} --ftp-ssl "
        "--header 'Connection: close' --connect-timeout 20 "
        "--retry 5 --retry-delay 10 -n {server_path} -o {local_path}"
    )

    gpm_username, gpm_password, gpm_base_dir = configs.read_gpm_api_configs().values()

    for server_path in server_paths:
        path = dl.curl_pps_cmd(
            server_path=server_path,
            disk_path=disk_path,
            username=gpm_username,
            password=gpm_password,
        )

        # Test against ftps -> ftp casting
        ftp_designation = server_path.split(":")[0]  # Split out URI scheme
        if ftp_designation == "ftps":
            server_path = server_path.replace("ftps://", "ftp://", 1)

        assert path == curl_truth.format(
            username=gpm_username,
            password=gpm_password,
            server_path=server_path,
            local_path=disk_path,
        )

        # Check that the folder is created
        assert os.path.exists(
            os.path.dirname(disk_path)
        ), f"Folder {os.path.dirname(disk_path)} was not created"


def test_construct_wget_pps_cmd(
    # username: str,
    # password: str,
    server_paths: Dict[str, Dict[str, Any]],
    tmpdir: str,
) -> None:
    """Test that the wget command constructor works as expected

    `disk_path` relates to a file on the disk
    """

    # Use datetime as path as to be unique to every test
    disk_path = os.path.join(
        tmpdir,
        datetime.datetime.utcnow().isoformat().replace(":", "-"),
        "wget",
        "output_file.hdf5",
    )
    assert not os.path.exists(
        os.path.dirname(disk_path)
    ), f"Folder {os.path.dirname(disk_path)} already exists"

    wget_truth = (
        "wget -4 --ftp-user={username} --ftp-password={password} "
        "-e robots=off -np -R .html,.tmp -nH -c --read-timeout=10 "
        "--tries=5 -O {local_path} {server_path}"
    )

    gpm_username, gpm_password, gpm_base_dir = configs.read_gpm_api_configs().values()

    for server_path in server_paths:
        path = dl.wget_pps_cmd(
            server_path=server_path,
            disk_path=disk_path,
            username=gpm_username,
            password=gpm_password,
        )

        assert path == wget_truth.format(
            username=gpm_username,
            password=gpm_password,
            server_path=server_path,
            local_path=disk_path,
        )

        # Check that the folder is created
        assert os.path.exists(
            os.path.dirname(disk_path)
        ), f"Folder {os.path.dirname(disk_path)} was not created"


def test_download_file_private(
    server_paths: Dict[str, Dict[str, Any]],
    tmpdir: str,
    mocker: MockerFixture,
) -> None:
    """Build curl/wget calls for download, but don't actually download anything

    Uses tmpdir to create a unique path for each test and mocker to mock the
    download function
    """
    storage = "pps"

    # Don't actually download anything, so mock the run function
    mocker.patch.object(dl, "run", autospec=True, return_value=None)

    # Use server paths in fixture and try curl and wget
    for server_path in server_paths:
        disk_path = os.path.join(
            tmpdir,
            "test_download_file_private",
            os.path.basename(server_path),
        )
        dl._download_files(
            src_fpaths=[server_path],
            dst_fpaths=[disk_path],
            storage=storage,
            transfer_tool="curl",
        )
        dl._download_files(
            src_fpaths=[server_path],
            dst_fpaths=[disk_path],
            storage=storage,
            transfer_tool="wget",
        )

        # Use non-existent transfer tool
        with pytest.raises(NotImplementedError):
            dl._download_files(
                src_fpaths=[server_path],
                dst_fpaths=[disk_path],
                storage=storage,
                transfer_tool="fake",
            )


def test_download_data(
    products: List[str],
    product_types: List[str],
    server_paths: Dict[str, Dict[str, Any]],
    mocker: MockerFixture,
    versions: List[str],
):
    """Test download_data function

    This test is somewhat redundant considering it is testing methods
    bundled in another functions which need to be turned off in order to
    test this function. However, it is useful to have a test that checks
    the entire download process.

    It may be useful as boilerplate to increase the number of tests here in the
    future.
    """

    mocker.patch.object(dl, "_download_files", autospec=True, return_value=[])
    mocker.patch.object(dl, "_download_daily_data", autospec=True, return_value=([], versions))
    mocker.patch.object(dl, "run", autospec=True, return_value=None)
    from gpm_api.io import info, pps

    mocker.patch.object(
        info,
        "_get_info_from_filename",
        autospec=True,
        return_value={
            "product": "2A-CLIM",
            "product_type": "CLIM",
            "start_time": datetime.datetime(2022, 9, 7, 12, 0, 0),
            "end_time": datetime.datetime(2022, 9, 7, 13, 0, 0),
            "version": "V07A",
            "satellite": "GPM",
            "granule_id": "2A-CLIM.GPM.GMI.GPROF2021v1.20150301-S121433-E134706.005708.V07A.HDF5",
        },
    )
    mocker.patch.object(
        find,
        "find_daily_filepaths",
        autospec=True,
        return_value=(server_paths.keys(), versions),
    )

    # Assume files pass file integrity check by mocking return as empty
    for product in products:
        for product_type in product_types:
            if product in available_products(product_type=product_type):
                res = dl.download_archive(
                    product=product,
                    start_time=datetime.datetime(2022, 9, 7, 12, 0, 0),
                    end_time=datetime.datetime(2022, 9, 8, 12, 0, 0),
                    product_type=product_type,
                )

        assert res is None  # Assume data is downloaded


def test_download_daily_data_private(
    tmpdir: str,
    versions: List[str],
    products: List[str],
    product_types: List[str],
    mocker: MockerFixture,
) -> None:
    """Test download_daily_data function

    Tests only the ability for the function to run without errors. Does not
    test the actual download process as communication with the server is
    mocked.
    """

    # Patch download functions as to not actually download anything
    mocker.patch.object(dl, "_download_files", autospec=True, return_value=[])
    mocker.patch.object(dl, "run", autospec=True, return_value=None)
    mocker.patch.object(dl, "find_daily_filepaths", autospec=True, return_value=([], versions))

    # Mocking empty responses will cause a DownloadWarning. Test that it is raised
    with pytest.warns(GPMDownloadWarning):
        for version in versions:
            for product_type in product_types:
                for product in available_products(product_type=product_type):
                    dl._download_daily_data(
                        storage="pps",
                        date=datetime.datetime(2022, 9, 7, 12, 0, 0),
                        version=version,
                        product=product,
                        product_type=product_type,
                        start_time=None,
                        end_time=None,
                        n_threads=4,
                        transfer_tool="curl",
                        progress_bar=True,
                        force_download=False,
                        verbose=True,
                        warn_missing_files=True,
                    )


def test_download_files(
    server_paths: Dict[str, Dict[str, Any]],
    versions: List[str],
    mocker: MockerFixture,
) -> None:
    """Test download_files function"""

    # Mock called functions as to not download any data
    mocker.patch.object(dl, "_download_files", autospec=True, return_value=[])
    mocker.patch.object(dl, "_download_daily_data", autospec=True, return_value=([], versions))
    mocker.patch.object(dl, "run", autospec=True, return_value=None)
    mocker.patch.object(dl, "check_filepaths_integrity", autospec=True, return_value=[])

    # Download a simple list of files
    assert dl.download_files(filepaths=list(server_paths.keys())) == []

    # Test that None filepaths returns None
    assert dl.download_files(filepaths=None) is None

    # Test that an empty list of filepaths returns None
    assert dl.download_files(filepaths=[]) is None

    # Test that a single filepath as a string also accepted as input
    assert dl.download_files(filepaths=list(server_paths.keys())[0]) == []

    # Test that filetypes other than a list are not accepted
    with pytest.raises(TypeError):
        for obj in [(), {}, 1, 1.0, "str", True]:
            dl.download_files(filepaths=obj)


def test_check_download_status(
    products: List[str],
) -> None:
    """Test check_download_status function"""

    for product in products:
        assert dl._check_download_status([-1, -1, -1], product, True) is True  # All already on disk
        assert dl._check_download_status([0, 0, 0], product, True) is None  # All failed download
        assert dl._check_download_status([1, 1, 1], product, True) is True  # All success download
        assert dl._check_download_status([], product, True) is None  # No data available
        assert dl._check_download_status([1, 0, 1], product, True) is True  # Some failed download


def test_get_fpaths_from_fnames(
    server_paths: Dict[str, Dict[str, Any]],
    versions: List[str],
    products: List[str],
    product_types: List[str],
    tmpdir: str,
) -> None:
    """Test convert_pps_to_disk_filepaths function

    Parameters
    """
    # TODO: WRONG REDO !
    assert dl.get_fpaths_from_fnames(
        filepaths=[
            "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5"
        ],
        storage="local",
        product_type="RS",
    ) == [
        os.path.join(
            tmpdir,
            "GPM",
            "RS",
            "V07",
            "RADAR",
            "2A-DPR",
            "2020",
            "07",
            "05",
            "2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5",
        )
    ]
