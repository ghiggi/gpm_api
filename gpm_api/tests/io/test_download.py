import pytest
import os
import datetime
import ftplib
from typing import Any, List, Dict
from pytest_mock.plugin import MockerFixture
from gpm_api.io import download as dl
from gpm_api.io.products import available_products


def test_construct_curl_cmd(
    username: str,
    password: str,
    server_paths: List[str],
    tmpdir: str,
) -> None:
    """Test that the curl command constructor works as expected

    `disk_path` relates to a file on the disk
    """

    # Use datetime as path as to be unique to every test
    disk_path = os.path.join(
        tmpdir,
        datetime.datetime.utcnow().isoformat(),
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

    for server_path in server_paths:
        path = dl.curl_cmd(
            server_path=server_path,
            disk_path=disk_path,
            username=username,
            password=password,
        )

        # Test against ftps -> ftp casting
        ftp_designation = server_path.split(":")[0]  # Split out URI scheme
        if ftp_designation == "ftps":
            server_path = server_path.replace("ftps://", "ftp://", 1)

        assert path == curl_truth.format(
            username=username,
            password=password,
            server_path=server_path,
            local_path=disk_path,
        )

        # Check that the folder is created
        assert os.path.exists(
            os.path.dirname(disk_path)
        ), f"Folder {os.path.dirname(disk_path)} was not created"


def test_construct_wget_cmd(
    username: str,
    password: str,
    server_paths: List[str],
    tmpdir: str,
) -> None:
    """Test that the wget command constructor works as expected

    `disk_path` relates to a file on the disk
    """

    # Use datetime as path as to be unique to every test
    disk_path = os.path.join(
        tmpdir,
        datetime.datetime.utcnow().isoformat(),
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

    for server_path in server_paths:
        path = dl.wget_cmd(
            server_path=server_path,
            disk_path=disk_path,
            username=username,
            password=password,
        )
        print(path)
        assert path == wget_truth.format(
            username=username,
            password=password,
            server_path=server_path,
            local_path=disk_path,
        )

        # Check that the folder is created
        assert os.path.exists(
            os.path.dirname(disk_path)
        ), f"Folder {os.path.dirname(disk_path)} was not created"


def test_download_with_ftplib(
    mocker: MockerFixture,
    server_paths: Dict[str, Dict[str, Any]],
    username: str,
    password: str,
    tmpdir: str,
) -> None:
    ftp_mock = mocker.patch(
        "ftplib.FTP_TLS",
        autospec=True,
    )
    file_open_mock = mocker.patch("builtins.open", autospec=True)

    # ftp_mock_instance = mocker.Mock(spec=FTP_TLS)
    ftp_mock.login.return_value = "230 Login successful."
    # Set the return value for the retrbinary method to indicate a successful download
    ftp_mock.retrbinary.return_value = "226 Transfer complete."
    ftp_mock.retrbinary.side_effect = [
        "226 Transfer complete.",
    ]

    disk_paths = []
    for server_path in server_paths:
        disk_paths.append(
            os.path.join(
                tmpdir,
                "test_download_with_ftplib",
                os.path.basename(server_path),
            )
        )

    dl.ftplib_download(
        server_paths=server_paths.keys(),
        disk_paths=server_paths,
        username=username,
        password=password,
    )

    assert ftp_mock.called

    # TODO: Assert username/password are passed to ftp_mock.login (assert_called_with not working?)


def test_download_file_private(
    server_paths: Dict[str, Dict[str, Any]],
    tmpdir: str,
    mocker: MockerFixture,
) -> None:
    """Build curl/wget calls for download, but don't actually download anything

    Uses tmpdir to create a unique path for each test and mocker to mock the
    download function
    """
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
            username="test",
            password="test",
            transfer_tool="curl",
        )
        dl._download_files(
            src_fpaths=[server_path],
            dst_fpaths=[disk_path],
            username="test",
            password="test",
            transfer_tool="wget",
        )

        # Use non-existent transfer tool
        with pytest.raises(NotImplementedError):
            dl._download_files(
                src_fpaths=[server_path],
                dst_fpaths=[disk_path],
                username="test",
                password="test",
                transfer_tool="fake",
            )


def test_check_file_completeness(
    server_paths: Dict[str, Dict[str, Any]],
    mocker: MockerFixture,
    tmpdir: str,
) -> None:
    # Assume files pass file integrity check by mocking return as empty
    file_integrity_checker = mocker.patch.object(
        dl, "check_file_integrity", autospec=True, return_value=[]
    )

    # Patch redownload function, assume all redownloaded files pass file integrity check
    redownload_func = mocker.patch.object(dl, "redownload_from_filepaths", return_value=[])

    for server_path, props in server_paths.items():
        disk_path = os.path.join(
            tmpdir,
            "test_download_file_private",
            os.path.basename(server_path),
        )
        res = dl._check_file_completness(
            base_dir=None,
            product=None,
            start_time=None,
            end_time=None,
            version=None,
            product_type=None,
            remove_corrupted=None,
            verbose=None,
            username=None,
            transfer_tool=None,
            retry=0,
            n_threads=1,
            progress_bar=True,
        )

        assert res == []
        assert file_integrity_checker.called
        assert not redownload_func.called

    # Attempt with corrupted files

    for server_path, props in server_paths.items():
        file_integrity_checker.return_value = [server_path]
        redownload_func.return_value = []
        disk_path = os.path.join(
            tmpdir,
            "test_download_file_private",
            os.path.basename(server_path),
        )
        res = dl._check_file_completness(
            base_dir=None,
            product=None,
            start_time=None,
            end_time=None,
            version=None,
            product_type=None,
            remove_corrupted=True,
            verbose=True,
            username=None,
            transfer_tool=None,
            retry=1,
            n_threads=1,
            progress_bar=True,
        )
        assert res == []
        assert redownload_func.called

        # Test that redownload_from_filepaths did not redownload the file
        file_integrity_checker.return_value = [server_path]
        redownload_func.return_value = [server_path]
        res = dl._check_file_completness(
            base_dir=None,
            product=None,
            start_time=None,
            end_time=None,
            version=None,
            product_type=None,
            remove_corrupted=True,
            verbose=True,
            username=None,
            transfer_tool=None,
            retry=1,
            n_threads=1,
            progress_bar=True,
        )
        assert len(res) == 1


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
    mocker.patch.object(dl, "_check_file_completness", autospec=True, return_value=[])
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
            "end_time": datetime.datetime(2022, 9, 7, 12, 0, 0),
            "version": "V06A",
            "satellite": "GPM",
            "granule_id": "2A-CLIM.GPM.GMI.XCAL2016-C.20220907-S120000-E120000.V06A.HDF5",
        },
    )
    mocker.patch.object(
        pps,
        "_find_pps_daily_filepaths",
        autospec=True,
        return_value=(server_paths.keys(), versions),
    )
    # Assume files pass file integrity check by mocking return as empty
    file_integrity_checker = mocker.patch.object(
        dl, "check_file_integrity", autospec=True, return_value=[]
    )

    # Patch redownload function, assume all redownloaded files pass file integrity check
    redownload_func = mocker.patch.object(dl, "redownload_from_filepaths", return_value=[])

    for product in products:
        for product_type in product_types:
            if product in available_products(product_type=product_type):
                res = dl.download_data(
                    product=product,
                    start_time=datetime.datetime(2022, 9, 7, 12, 0, 0),
                    end_time=datetime.datetime(2022, 9, 8, 12, 0, 0),
                    product_type=product_type,
                )

        assert res is None  # Assume data is downloaded


#     def download_daily_data(
#     product,
#     year,
#     month,
#     day,
#     product_type="RS",
#     version=GPM_VERSION,
#     n_threads=10,
#     transfer_tool="curl",
#     progress_bar=False,
#     force_download=False,
#     check_integrity=True,
#     remove_corrupted=True,
#     verbose=True,
#     retry=1,
#     base_dir=None,
#     username=None,
#     password=None,
# ):
#     from gpm_api.io.download import download_data

#     start_time = datetime.date(year, month, day)
#     end_time = start_time + relativedelta(days=1)

#     l_corrupted = download_data(
#         product=product,
#         start_time=start_time,
#         end_time=end_time,
#         product_type=product_type,
#         version=version,
#         n_threads=n_threads,
#         transfer_tool=transfer_tool,
#         progress_bar=progress_bar,
#         force_download=force_download,
#         check_integrity=check_integrity,
#         remove_corrupted=remove_corrupted,
#         verbose=verbose,
#         retry=retry,
#         base_dir=base_dir,
#         username=username,
#         password=password,
#     )
