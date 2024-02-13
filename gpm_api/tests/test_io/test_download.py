import pytest
import os
import datetime
import platform
from typing import Any, List, Dict
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from pytest_mock.plugin import MockerFixture
from gpm_api.io import find
from gpm_api.io import download as dl
from gpm_api.io.products import available_products, get_product_start_time
from gpm_api.utils.warnings import GPMDownloadWarning


def test_construct_curl_pps_cmd(
    mock_configuration: Dict[str, str],
    remote_filepaths: Dict[str, Dict[str, Any]],
    tmpdir: str,
) -> None:
    """Test that the curl command constructor works as expected

    `local_filepath` relates to a file on the disk
    """

    # Use datetime as path as to be unique to every test
    local_filepath = os.path.join(
        tmpdir,
        datetime.datetime.utcnow().isoformat().replace(":", "-"),
        "curl",
        "output_file.hdf5",
    )
    assert not os.path.exists(
        os.path.dirname(local_filepath)
    ), f"Folder {os.path.dirname(local_filepath)} already exists"

    curl_truth = (
        "curl --ipv4 --insecure -n "
        "--user {username}:{password} --ftp-ssl "
        "--header 'Connection: close' --connect-timeout 20 "
        "--retry 5 --retry-delay 10 --url {remote_filepath} -o {local_filepath}"
    )

    username_pps = mock_configuration["username_pps"]
    password_pps = mock_configuration["password_pps"]

    for remote_filepath in remote_filepaths:
        path = dl.curl_pps_cmd(
            remote_filepath=remote_filepath,
            local_filepath=local_filepath,
            username=username_pps,
            password=password_pps,
        )

        # Test against ftps -> ftp casting
        ftp_designation = remote_filepath.split(":")[0]  # Split out URI scheme
        if ftp_designation == "ftps":
            remote_filepath = remote_filepath.replace("ftps://", "ftp://", 1)

        assert path == curl_truth.format(
            username=username_pps,
            password=password_pps,
            remote_filepath=remote_filepath,
            local_filepath=local_filepath,
        )

        # Check that the folder is created
        assert os.path.exists(
            os.path.dirname(local_filepath)
        ), f"Folder {os.path.dirname(local_filepath)} was not created"


def test_construct_wget_pps_cmd(
    mock_configuration: Dict[str, str],
    remote_filepaths: Dict[str, Dict[str, Any]],
    tmpdir: str,
) -> None:
    """Test that the wget command constructor works as expected

    `local_filepath` relates to a file on the disk
    """

    # Use datetime as path as to be unique to every test
    local_filepath = os.path.join(
        tmpdir,
        datetime.datetime.utcnow().isoformat().replace(":", "-"),
        "wget",
        "output_file.hdf5",
    )
    assert not os.path.exists(
        os.path.dirname(local_filepath)
    ), f"Folder {os.path.dirname(local_filepath)} already exists"

    wget_truth = (
        "wget -4 --ftp-user={username} --ftp-password={password} "
        "-e robots=off -np -R .html,.tmp -nH -c --read-timeout=10 "
        "--tries=5 -O {local_filepath} {remote_filepath}"
    )

    username_pps = mock_configuration["username_pps"]
    password_pps = mock_configuration["password_pps"]

    for remote_filepath in remote_filepaths:
        path = dl.wget_pps_cmd(
            remote_filepath=remote_filepath,
            local_filepath=local_filepath,
            username=username_pps,
            password=password_pps,
        )

        assert path == wget_truth.format(
            username=username_pps,
            password=password_pps,
            remote_filepath=remote_filepath,
            local_filepath=local_filepath,
        )

        # Check that the folder is created
        assert os.path.exists(
            os.path.dirname(local_filepath)
        ), f"Folder {os.path.dirname(local_filepath)} was not created"


class TestDownloadUtility:
    def test_get_commands_futures(self):
        """Test _get_commands_futures."""
        commands = ["echo 'Hello, World!'", "exit 1"]

        with ThreadPoolExecutor() as executor:
            with patch("subprocess.check_call", return_value=0) as mock_check_call:
                futures = dl._get_commands_futures(executor, commands)
                assert len(futures) == len(
                    commands
                ), "Number of futures should match number of commands"
                # Ensure each future is for a subprocess.check_call call
                for future in futures:
                    assert (
                        futures[future][1] in commands
                    ), "Future command should be in commands list"
                mock_check_call.assert_called()

    def test_get_list_failing_commands(self):
        """Test _get_list_failing_commands."""
        commands = ["echo 'Hello, World!'", "exit 1"]  # Second command simulates a failure

        with ThreadPoolExecutor() as executor:
            futures = dl._get_commands_futures(executor, commands)
            failing_commands = dl._get_list_failing_commands(futures)
            # Check the failing command is in the failing_commands list
            assert (
                commands[1] in failing_commands
            ), "Failing command should be in the list of failed commands"
            # Check the valid command is not in the failing_commands list
            assert (
                commands[0] not in failing_commands
            ), "Successful command should not be in the list of failed commands"

    def test_get_list_status_commands(self):
        """Test _get_list_status_commands."""
        commands = ["echo 'Hello, World!'", "exit 1"]  # Second command simulates a failure

        with ThreadPoolExecutor() as executor:
            futures = dl._get_commands_futures(executor, commands)
            status_list = dl._get_list_status_commands(futures)
            assert len(status_list) == len(
                commands
            ), "Status list length should match number of commands"
            # Assuming the first command does not fails, its status should be 1
            assert status_list[0] == 1, "Status for valid command should be 1"
            # Assuming the second command fails, its status should be 0
            assert status_list[1] == 0, "Status for failing command should be 0"

    @pytest.mark.parametrize("verbose", [True, False])
    @pytest.mark.parametrize("progress_bar", [True, False])
    @pytest.mark.parametrize("n_threads", [0, 1, 20])  # Test for n_threads > n_commands
    def test_run(self, mocker: MockerFixture, verbose, progress_bar, n_threads) -> None:
        """Test run function."""

        commands = [
            "echo 'Hello, World!'",
            "exit 1",
        ]  # subprocess.run("exit 1", shell=True).returncode --> 1
        expected_status = [0, 1]

        mock_get_cmds = mocker.patch.object(
            dl, "_get_commands_futures", autospec=True, return_value=[]
        )
        mock_status_cmds = mocker.patch.object(
            dl, "_get_list_status_commands", autospec=True, return_value=expected_status
        )

        if progress_bar:
            mock_tqdm = mocker.patch("tqdm.tqdm")

        status = dl.run(commands, progress_bar=progress_bar)
        assert status == expected_status

        # Assert called
        if n_threads >= 1 and progress_bar:
            mock_get_cmds.assert_called()
            mock_status_cmds.assert_called()

        # Assert tqdm is called
        if progress_bar:
            mock_tqdm.assert_called_once_with(total=len(commands))


class TestGetFpathsFromfilenames:
    """Test get_filepaths_from_filenames function"""

    filename = "2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5"

    def test_local(
        self,
        mock_configuration: Dict[str, str],
    ) -> None:
        assert dl.get_filepaths_from_filenames(
            filepaths=[self.filename],
            storage="local",
            product_type="RS",
        ) == [
            os.path.join(
                mock_configuration["gpm_base_dir"],
                "GPM",
                "RS",
                "V07",
                "RADAR",
                "2A-DPR",
                "2020",
                "07",
                "05",
                self.filename,
            )
        ]

    def test_pps(self) -> None:
        assert dl.get_filepaths_from_filenames(
            filepaths=[self.filename],
            storage="pps",
            product_type="RS",
        ) == [f"ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/{self.filename}"]

    def test_ges_disc(self) -> None:
        assert dl.get_filepaths_from_filenames(
            filepaths=[self.filename],
            storage="ges_disc",
            product_type="RS",
        ) == [
            f"https://gpm2.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.07/2020/187/{self.filename}"
        ]

    def test_invalid_filename(self) -> None:
        with pytest.raises(ValueError):
            dl.get_filepaths_from_filenames(
                filepaths=["invalid_filename"],
                storage="local",
                product_type="RS",
            )


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


@pytest.mark.parametrize("transfer_tool", ["wget", "curl"])
@pytest.mark.parametrize("storage", ["pps", "ges_disc"])
def test__download_files(
    remote_filepaths: Dict[str, Dict[str, Any]],
    tmpdir: str,
    mocker: MockerFixture,
    transfer_tool: str,
    storage: str,
) -> None:
    """Build curl/wget calls for download, but don't actually download anything

    Uses tmpdir to create a unique path for each test and mocker to mock the
    download function
    """
    if platform.system() == "Windows" and transfer_tool == "wget":
        return None

    # Don't actually download anything, so mock the run function
    mocker.patch.object(dl, "run", autospec=True, return_value=None)

    # Use server paths in fixture and try curl and wget
    for remote_filepath in remote_filepaths:
        local_filepath = os.path.join(
            tmpdir,
            "test_download_file_private",
            os.path.basename(remote_filepath),
        )
        dl._download_files(
            remote_filepaths=[remote_filepath],
            local_filepaths=[local_filepath],
            storage=storage,
            transfer_tool=transfer_tool,
        )

    # Test non-existent transfer tool
    with pytest.raises(ValueError):
        dl._download_files(
            remote_filepaths=[remote_filepath],
            local_filepaths=[local_filepath],
            storage=storage,
            transfer_tool="fake",
        )


def test_download_files(
    remote_filepaths: Dict[str, Dict[str, Any]],
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
    assert dl.download_files(filepaths=list(remote_filepaths.keys())) == []

    # Test that None filepaths returns None
    assert dl.download_files(filepaths=None) is None

    # Test that an empty list of filepaths returns None
    assert dl.download_files(filepaths=[]) is None

    # Test that a single filepath as a string also accepted as input
    assert dl.download_files(filepaths=list(remote_filepaths.keys())[0]) == []

    # Test that filetypes other than a list are not accepted
    with pytest.raises(TypeError):
        for obj in [(), {}, 1, 1.0, "str", True]:
            dl.download_files(filepaths=obj)

    # Test data already on disk (force_download=False)
    mocker.patch.object(dl, "filter_download_list", autospec=True, return_value=([], []))
    assert dl.download_files(filepaths=list(remote_filepaths.keys())) == None


@pytest.mark.parametrize("storage", ["pps", "ges_disc"])
def test__download_daily_data(
    tmpdir: str,
    versions: List[str],
    products: List[str],
    product_types: List[str],
    mocker: MockerFixture,
    storage: str,
) -> None:
    """Test download_daily_data function

    Tests only the ability for the function to run without errors. Does not
    test the actual download process as communication with the server is
    mocked.
    """
    # TODO: this currently test only behaviour when no files available !!!
    # --> mock find_daily_filepaths and get_filepaths_from_filenames to return valid paths !
    # --> mock filter_download_list (force True or False) to return something or nothing !

    # Patch download functions as to not actually download anything
    mocker.patch.object(dl, "_download_files", autospec=True, return_value=[])
    mocker.patch.object(dl, "run", autospec=True, return_value=None)
    mocker.patch.object(dl, "find_daily_filepaths", autospec=True, return_value=([], versions))

    # Mocking empty responses will cause a DownloadWarning. Test that it is raised
    with pytest.warns(GPMDownloadWarning):
        for version in versions:
            for product_type in product_types:
                for product in available_products(product_types=product_type):
                    dl._download_daily_data(
                        storage=storage,
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


class TestDownloadArchive:
    @pytest.fixture(autouse=True)
    def mock_download(
        self,
        mocker: MockerFixture,
        remote_filepaths: Dict[str, Dict[str, Any]],
        versions: List[str],
    ) -> None:
        from gpm_api.io import info

        mocker.patch.object(dl, "_download_files", autospec=True, return_value=[])
        mocker.patch.object(dl, "_download_daily_data", autospec=True, return_value=([], versions))
        mocker.patch.object(dl, "run", autospec=True, return_value=None)
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
            return_value=(remote_filepaths.keys(), versions),
        )

    @pytest.mark.parametrize("check_integrity", [True, False])
    @pytest.mark.parametrize("remove_corrupted", [True, False])
    def test_download_data_with_no_data_available(
        self,
        check,  # For non-failing asserts
        products: List[str],
        product_types: List[str],
        check_integrity,
        remove_corrupted,
    ):
        """Test download_data function

        This test is somewhat redundant considering it is testing methods
        bundled in another functions which need to be turned off in order to
        test this function. However, it is useful to have a test that checks
        the entire download process.

        It may be useful as boilerplate to increase the number of tests here in the
        future.
        """
        for product_type in product_types:
            for product in available_products(product_types=product_type):
                start_time = get_product_start_time(product)
                if start_time is None:
                    continue
                res = dl.download_archive(
                    product=product,
                    start_time=start_time,
                    end_time=start_time + datetime.timedelta(hours=1),
                    product_type=product_type,
                    check_integrity=check_integrity,
                    remove_corrupted=remove_corrupted,
                )
            with check:
                assert res is None  # Assume data is downloaded

    def test_corrupted_archive(
        self,
        mocker: MockerFixture,
    ) -> None:
        product = "1A-GMI"
        start_time = get_product_start_time(product)
        end_time = start_time + datetime.timedelta(hours=1)

        # Mock download status as failed
        mocker.patch.object(dl, "_check_download_status", autospec=True, return_value=False)

        # Mock file integrity check as passed
        mocker.patch.object(dl, "check_archive_integrity", autospec=True, return_value=[])

        dl.download_archive(
            product=product,
            start_time=start_time,
            end_time=end_time,
        )


def test_download_daily_data(mocker: MockerFixture):
    """Test download_daily_data."""
    # Patch download functions as to not actually download anything
    mocker.patch.object(dl, "download_archive", autospec=True, return_value=[])
    l_corrupted = dl.download_daily_data(product="dummy", year=2012, month=2, day=1)
    assert l_corrupted == []


def test_download_monthly_data(mocker: MockerFixture):
    """Test download_monhtly_data."""
    # Patch download functions as to not actually download anything
    mocker.patch.object(dl, "download_archive", autospec=True, return_value=[])
    l_corrupted = dl.download_monthly_data(
        product="dummy",
        year=2012,
        month=2,
    )
    assert l_corrupted == []


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("remove_corrupted", [True, False])
@pytest.mark.parametrize("retry", [0, 1])
@pytest.mark.parametrize("filepaths", [["some_corrupted_filepaths"], []])
def test_ensure_files_completness(
    mocker: MockerFixture, verbose, remove_corrupted, retry, filepaths
):
    """Test _ensure_files_completness."""
    # Patch download functions as to not actually download anything
    mocker.patch.object(dl, "download_files", autospec=True, return_value=filepaths)
    mocker.patch.object(dl, "check_filepaths_integrity", autospec=True, return_value=filepaths)

    # Ensure completeness
    l_corrupted = dl._ensure_files_completness(
        filepaths=filepaths,
        retry=retry,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
        # Dummy
        product_type="RS",
        transfer_tool="wget",
        n_threads=1,
        progress_bar=True,
    )
    assert l_corrupted == filepaths


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("remove_corrupted", [True, False])
@pytest.mark.parametrize("retry", [0, 1])
@pytest.mark.parametrize("filepaths", [["some_corrupted_filepaths"], []])
def test_ensure_archive_completness(
    mocker: MockerFixture, verbose, remove_corrupted, retry, filepaths
):
    """Test _ensure_archive_completness."""
    # Patch download functions as to not actually download anything
    mocker.patch.object(dl, "download_files", autospec=True, return_value=filepaths)
    mocker.patch.object(dl, "check_archive_integrity", autospec=True, return_value=filepaths)

    # Ensure completeness
    l_corrupted = dl._ensure_archive_completness(
        retry=retry,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
        # Dummy
        product="dummy",
        start_time="dummy",
        end_time="dummy",
        version=7,
        product_type="RS",
        transfer_tool="wget",
        n_threads=1,
        progress_bar=True,
    )
    assert l_corrupted == filepaths
