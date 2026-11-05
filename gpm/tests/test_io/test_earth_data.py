"""Test Earthdata discovery and downloads without accessing NASA services."""

import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
import xarray as xr
from earthaccess.exceptions import DownloadFailure
from requests.exceptions import ConnectionError, Timeout

import gpm
from gpm.io import download as dl
from gpm.io import earthdata as ed
from gpm.io import find
from gpm.io.checks import check_remote_storage, check_storage
from gpm.utils.warnings import GPM_Warning

FILENAME = "2A.GPM.DPR.V9-20211125.20200705-S231057-E004329.036096.V08A.HDF5"
SECOND_FILENAME = "2A.GPM.DPR.V9-20211125.20200706-S004330-E021602.036097.V08A.HDF5"
START = datetime.datetime(2020, 7, 6)
END = datetime.datetime(2020, 7, 6, 1)


def granule(filename):
    return Mock(data_links=Mock(return_value=[f"https://data.example.org/{filename}"]))


@pytest.fixture
def cloud(mocker):
    """Mock only earthaccess, leaving GPM filtering, paths and integrity checks real."""
    token = mocker.patch.object(ed.earthaccess, "get_edl_token", return_value=None)

    def authenticate():
        token.return_value = "test-token"

    login = mocker.patch.object(ed.earthaccess, "login", side_effect=authenticate)
    search = mocker.patch.object(ed.earthaccess, "search_data", return_value=[granule(FILENAME)])

    def download(granules, local_path, **kwargs):
        paths = []
        for result in granules:
            path = Path(local_path) / Path(result).name
            xr.Dataset(attrs={"downloaded": "yes"}).to_netcdf(path)
            paths.append(path)
        return paths[::-1]

    download = mocker.patch.object(ed.earthaccess, "download", side_effect=download)
    return login, search, download


@pytest.mark.parametrize("storage", ["EARTHDATA", "earthdata"])
def test_storage(storage):
    assert check_storage(storage) == "EARTHDATA"
    assert check_remote_storage(storage) == "EARTHDATA"


@pytest.mark.parametrize(
    ("product", "product_type", "date", "directory"),
    [
        ("2A-DPR", "RS", datetime.date(2018, 12, 31), "GPM_L2/GPM_2ADPR.08/2018/365"),
        ("2A-DPR", "RS", datetime.date(2020, 12, 31), "GPM_L2/GPM_2ADPR.08/2020/366"),
        ("1A-GMI", "RS", datetime.date(2020, 1, 1), "GPM_L1A/GPM_1AGMI.08/2020/001"),
    ],
)
def test_define_earthdata_filepath(product, product_type, date, directory):
    filename = "example.nc"
    assert ed.define_earthdata_filepath(product, product_type, date, 8, filename) == (
        f"https://data.gesdisc.earthdata.nasa.gov/data/{directory}/{filename}"
    )


def test_filename_path_helpers_do_not_search_earthdata(cloud):
    filename = "2A.GPM.DPR.V10-20260310.20181231-S224214-E001446.027509.V08A.nc"
    expected = f"https://data.gesdisc.earthdata.nasa.gov/data/GPM_L2/GPM_2ADPR.08/2018/365/{filename}"
    assert dl._get_func_filepath_definition("EARTHDATA") is ed.define_earthdata_filepath
    assert dl.get_filepath_from_filename(filename, "EARTHDATA", "RS") == expected
    assert dl.get_filepaths_from_filenames([filename], "EARTHDATA", "RS") == [expected]
    cloud[0].assert_not_called()
    cloud[1].assert_not_called()


@pytest.mark.parametrize("version", [7, 8])
def test_search_filters_and_formats_version(cloud, version):
    login, search, _ = cloud
    filename = FILENAME.replace("V08A", f"V{version:02d}A")
    valid = granule(filename)
    search.return_value = [
        valid,
        granule(filename),  # Duplicate CMR record.
        granule(filename.replace("S231057-E004329", "S170044-E183317")),
        granule(filename.replace(f"V{version:02d}A", "V05A")),
        granule("metadata.xml"),
        Mock(data_links=Mock(return_value=[])),
    ]
    bbox = (-10, 40, 10, 50)
    assert ed.search_earthdata("2A-DPR", version, START, END, bounding_box=bbox) == [valid]
    login.assert_called_once_with()
    search.assert_called_once_with(
        short_name="GPM_2ADPR",
        version=f"{version:02d}",
        temporal=(START, END),
        cloud_hosted=True,
        bounding_box=bbox,
    )


@pytest.mark.parametrize("operation", ["search", "download"])
@pytest.mark.parametrize("error_type", [ConnectionError, Timeout])
def test_login_connection_failure_reports_unavailability(cloud, tmp_path, operation, error_type):
    original_error = error_type("Could not connect to urs.earthdata.nasa.gov")
    cloud[0].side_effect = original_error
    if operation == "search":
        function, args = ed.search_earthdata, ("2A-DPR", 8, START, END)
    else:
        function, args = ed.download_earthdata, ([f"https://data.example.org/{FILENAME}"], [str(tmp_path / FILENAME)])
    with pytest.raises(RuntimeError, match="EarthData Cloud is not available right now") as exc_info:
        function(*args)
    assert f"Error is: {error_type.__name__}: {original_error}" in str(exc_info.value)
    assert exc_info.value.__cause__ is original_error
    cloud[1].assert_not_called()
    cloud[2].assert_not_called()


def test_login_is_reused_across_find_and_download_calls(cloud, tmp_path):
    with gpm.config.set({"base_dir": str(tmp_path / "GPM")}):
        for _ in range(2):
            gpm.find_files("EARTHDATA", "2A-DPR", START, END, version=8)
            dl.download_files(FILENAME, storage="EARTHDATA", force_download=True, verbose=False)
    cloud[0].assert_called_once_with()
    assert cloud[1].call_count == 4
    assert cloud[2].call_count == 2


def test_existing_earthaccess_login_is_reused(cloud, tmp_path):
    ed.earthaccess.get_edl_token.return_value = "existing-token"
    ed.search_earthdata("2A-DPR", 8, START, END)
    ed.download_earthdata([f"https://data.example.org/{FILENAME}"], [str(tmp_path / FILENAME)])
    cloud[0].assert_not_called()


def test_failed_login_can_be_retried(cloud):
    cloud[0].side_effect = [ConnectionError("Temporary failure in name resolution"), None]
    with pytest.raises(RuntimeError, match="Check your internet connection"):
        ed._login_earthdata()
    ed._login_earthdata()
    assert cloud[0].call_count == 2


@pytest.mark.parametrize("operation", ["find", "download"])
@pytest.mark.parametrize("error_type", [ConnectionError, Timeout])
def test_lost_connection_after_login_has_informative_error(cloud, tmp_path, operation, error_type):
    ed._login_earthdata()
    original_error = error_type("Temporary failure in name resolution")
    if operation == "find":
        cloud[1].side_effect = original_error
        function, args = gpm.find_files, ("EARTHDATA", "2A-DPR", START, END)
        kwargs = {"version": 8}
    else:
        cloud[2].side_effect = original_error
        function, args = dl.download_files, (FILENAME,)
        kwargs = {"storage": "EARTHDATA", "verbose": False}
    with (
        gpm.config.set({"base_dir": str(tmp_path / "GPM")}),
        pytest.raises(RuntimeError, match="Check your internet connection") as exc_info,
    ):
        function(*args, **kwargs)
    assert f"Error is: {error_type.__name__}: {original_error}" in str(exc_info.value)
    assert exc_info.value.__cause__ is original_error
    cloud[0].assert_called_once_with()
    if operation == "download":
        assert not Path(cloud[2].call_args.kwargs["local_path"]).exists()


def test_granule_filename():
    result = Mock(data_links=Mock(return_value=["https://example.org/data%20file.HDF5?token=abc"]))
    assert ed.get_granule_filename(result) == "data file.HDF5"
    with pytest.raises(ValueError, match="no data links"):
        ed.get_granule_filename(Mock(data_links=Mock(return_value=[])))


def test_granule_data_link():
    result = Mock(data_links=Mock(return_value=["https://example.org/data.HDF5", "https://example.org/other.HDF5"]))
    assert ed.get_granule_url(result) == "https://example.org/data.HDF5"
    with pytest.raises(ValueError, match="no data links"):
        ed.get_granule_url(Mock(data_links=Mock(return_value=[])))


def test_granule_access_is_isolated_from_earthaccess_results(cloud, mocker):
    # A future earthaccess result need not expose data_links if the adapter changes.
    result = object()
    url = f"https://data.example.org/{FILENAME}"
    accessor = mocker.patch.object(ed, "get_granule_url", return_value=url)
    cloud[1].return_value = [result]
    assert ed.get_granule_filename(result) == FILENAME
    assert ed.search_earthdata("2A-DPR", 8, START, END) == [result]
    assert ed.get_earthdata_daily_filepaths("2A-DPR", "RS", START.date() - datetime.timedelta(days=1), 8) == [url]
    accessor.assert_called_with(result)


def test_search_missing_collection(cloud):
    with pytest.raises(ValueError, match="No NASA collection short name"):
        ed.search_earthdata("2B-GPM-CSAT", 7, START, END)
    cloud[0].assert_not_called()


def test_search_unsupported_nrt(cloud):
    with pytest.raises(ValueError, match="NRT products only for IMERG"):
        ed.search_earthdata("2A-DPR", 8, START, END, product_type="NRT")
    cloud[0].assert_not_called()


def test_search_nrt_imerg(cloud):
    cloud[1].return_value = []
    assert ed.search_earthdata("IMERG-ER", 7, START, END, product_type="NRT") == []
    assert cloud[1].call_args.kwargs["short_name"] == "GPM_3IMERGHHE"


def test_daily_discovery_avoids_duplicate_cross_day_granules(cloud):
    cloud[1].return_value = [granule(FILENAME), granule(SECOND_FILENAME)]
    paths, versions = find.find_daily_filepaths(
        storage="EARTHDATA",
        date=START.date(),
        product="2A-DPR",
        product_type="RS",
        version=8,
    )
    assert paths == [f"https://data.example.org/{SECOND_FILENAME}"]
    assert versions == [8]


@pytest.mark.parametrize("parallel", [True, False])
def test_period_discovery_queries_daily_and_includes_cross_midnight(cloud, parallel, mocker):
    compute = mocker.patch.object(
        find.dask,
        "compute",
        side_effect=AssertionError("Earthdata searches must be sequential"),
    )
    cloud[1].return_value = [granule(SECOND_FILENAME), granule(FILENAME)]
    paths = gpm.find_files("EARTHDATA", "2A-DPR", START, END, version=8, groups="product", parallel=parallel)
    assert paths == {"2A-DPR": [f"https://data.example.org/{name}" for name in [FILENAME, SECOND_FILENAME]]}
    assert cloud[1].call_count == 2
    periods = sorted(call.kwargs["temporal"] for call in cloud[1].call_args_list)
    assert periods == [
        (START - datetime.timedelta(days=1), START),
        (START, START + datetime.timedelta(days=1)),
    ]
    compute.assert_not_called()


def test_yearly_discovery_limits_searches_to_one_day(cloud):
    cloud[1].return_value = []
    start = datetime.datetime(2021, 1, 1)
    end = datetime.datetime(2022, 1, 1)
    assert gpm.find_files("EARTHDATA", "2A-DPR", start, end, version=8, parallel=False) == []
    periods = [call.kwargs["temporal"] for call in cloud[1].call_args_list]
    assert len(periods) == 367  # Includes the previous day for cross-midnight granules.
    assert all(stop - begin == datetime.timedelta(days=1) for begin, stop in periods)


def test_yearly_archive_limits_searches_to_one_day_and_forwards_bbox(cloud):
    cloud[1].return_value = []
    bbox = (-10, 40, 10, 50)
    assert (
        dl.download_archive(
            "2A-DPR",
            "2021-01-01",
            "2022-01-01",
            version=8,
            storage="EARTHDATA",
            bounding_box=bbox,
            verbose=False,
        )
        is None
    )
    assert cloud[1].call_count == 367
    for call in cloud[1].call_args_list:
        start, end = call.kwargs["temporal"]
        assert end - start == datetime.timedelta(days=1)
        assert call.kwargs["bounding_box"] == bbox
    cloud[2].assert_not_called()


def test_daily_batches_download_before_searching_the_next_day(cloud, tmp_path):
    real_download = cloud[2].side_effect
    events = []

    def search(**kwargs):
        events.append(("search", kwargs["temporal"][0].date()))
        return [granule(FILENAME), granule(SECOND_FILENAME)]

    def download(granules, local_path, **kwargs):
        events.append(("download", [Path(result).name for result in granules]))
        return real_download(granules, local_path, **kwargs)

    cloud[1].side_effect = search
    cloud[2].side_effect = download
    with gpm.config.set({"base_dir": str(tmp_path / "GPM")}):
        assert (
            dl.download_archive(
                "2A-DPR",
                "2020-07-05 22:00:00",
                "2020-07-06 02:00:00",
                version=8,
                storage="EARTHDATA",
                force_download=True,
                verbose=False,
            )
            == []
        )
    assert events == [
        ("search", datetime.date(2020, 7, 4)),
        ("search", datetime.date(2020, 7, 5)),
        ("download", [FILENAME]),
        ("search", datetime.date(2020, 7, 6)),
        ("download", [SECOND_FILENAME]),
    ]


def test_temporary_download_and_name_matching(cloud, tmp_path):
    local = [str(tmp_path / "archive" / name) for name in [FILENAME, SECOND_FILENAME]]
    status = ed.download_earthdata(
        [f"https://data.example.org/{name}" for name in [FILENAME, SECOND_FILENAME]],
        local,
        n_threads=3,
        progress_bar=False,
    )
    assert status == [1, 1]
    assert all(Path(path).is_file() for path in local)
    kwargs = cloud[2].call_args.kwargs
    assert kwargs["threads"] == 3
    assert kwargs["show_progress"] is False
    assert kwargs["force"] is True
    assert kwargs["provider"] == "GES_DISC"
    cloud[0].assert_called_once_with()
    cloud[1].assert_not_called()
    assert not Path(kwargs["local_path"]).exists()


def test_download_failure_preserves_archive_and_cleans_temporary_files(cloud, tmp_path):
    target = tmp_path / FILENAME
    target.write_bytes(b"existing data")
    staging_paths = []

    def fail(granules, local_path, **kwargs):
        staging_paths.append(Path(local_path))
        (Path(local_path) / FILENAME).write_bytes(b"partial data")
        raise DownloadFailure("interrupted")

    cloud[2].side_effect = fail
    with pytest.raises(DownloadFailure, match="interrupted"):
        ed.download_earthdata([f"https://data.example.org/{FILENAME}"], [str(target)])
    assert target.read_bytes() == b"existing data"
    assert not staging_paths[0].exists()


def test_missing_download_is_not_archived(cloud, tmp_path):
    cloud[2].return_value = []
    cloud[2].side_effect = None
    target = tmp_path / FILENAME
    with pytest.warns(GPM_Warning, match="could not download"):
        assert ed.download_earthdata([f"https://data.example.org/{FILENAME}"], [str(target)]) == [0]
    assert not target.exists()
    assert not Path(cloud[2].call_args.kwargs["local_path"]).exists()


def test_archive_download_skip_and_force(cloud, tmp_path, mocker):
    transfer_check = mocker.patch.object(dl, "check_transfer_tool", side_effect=AssertionError("No curl needed"))
    bbox = (-10, 40, 10, 50)
    with gpm.config.set({"base_dir": str(tmp_path / "GPM")}):
        kwargs = {
            "product": "2A-DPR",
            "start_time": START,
            "end_time": END,
            "version": 8,
            "storage": "earthdata",
            "n_threads": 2,
            "progress_bar": False,
            "bounding_box": bbox,
            "verbose": False,
        }
        assert dl.download_archive(**kwargs) == []
        target = Path(dl.get_filepath_from_filename(FILENAME, "LOCAL", "RS"))
        assert target.is_file()
        with xr.open_dataset(target) as ds:
            assert ds.attrs["downloaded"] == "yes"
        assert dl.download_archive(**kwargs) == []
        assert cloud[2].call_count == 1
        dl.download_archive(**kwargs, force_download=True)
        assert cloud[2].call_count == 2
    transfer_check.assert_not_called()
    assert cloud[1].call_args.kwargs["bounding_box"] == bbox


def test_bounding_box_download_checks_full_local_time_range(cloud, tmp_path):
    with gpm.config.set({"base_dir": str(tmp_path / "GPM")}):
        existing = Path(dl.get_filepath_from_filename(SECOND_FILENAME, "LOCAL", "RS"))
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"corrupted file outside bounding box")
        outside_filename = "2A.GPM.DPR.V9-20211125.20200706-S170044-E183317.036098.V08A.HDF5"
        outside = Path(dl.get_filepath_from_filename(outside_filename, "LOCAL", "RS"))
        outside.write_bytes(b"corrupted file outside time range")
        assert (
            dl.download_archive(
                "2A-DPR",
                START,
                END,
                version=8,
                storage="EARTHDATA",
                bounding_box=(-10, 40, 10, 50),
                verbose=False,
            )
            == []
        )
        with xr.open_dataset(existing) as ds:
            assert ds.attrs["downloaded"] == "yes"
        assert outside.read_bytes() == b"corrupted file outside time range"
    assert cloud[2].call_count == 2
    assert cloud[2].call_args_list[0].args[0] == [f"https://data.example.org/{FILENAME}"]
    assert cloud[2].call_args_list[1].args[0] == [dl.get_filepath_from_filename(SECOND_FILENAME, "EARTHDATA", "RS")]


def test_download_by_filename_skips_existing_without_search(cloud, tmp_path):
    with gpm.config.set({"base_dir": str(tmp_path / "GPM")}):
        assert dl.download_files(FILENAME, storage="EARTHDATA", verbose=False) == []
        assert dl.download_files(FILENAME, storage="EARTHDATA", verbose=False) is None
        cloud[1].assert_not_called()
        cloud[0].assert_called_once_with()
        assert cloud[2].call_args.args[0] == [dl.get_filepath_from_filename(FILENAME, "EARTHDATA", "RS")]
        assert cloud[2].call_count == 1


def test_archive_corruption_retry_stays_on_earthdata(cloud, tmp_path, mocker):
    retry_download = mocker.spy(dl, "download_files")
    real_download = cloud[2].side_effect
    calls = []

    def initially_corrupted(granules, local_path, **kwargs):
        calls.append(local_path)
        if len(calls) == 1:
            path = Path(local_path) / FILENAME
            path.write_bytes(b"corrupted")
            return [path]
        return real_download(granules, local_path, **kwargs)

    cloud[2].side_effect = initially_corrupted
    with gpm.config.set({"base_dir": str(tmp_path / "GPM")}):
        assert (
            dl.download_archive(
                "2A-DPR",
                START,
                END,
                version=8,
                storage="EARTHDATA",
                verbose=False,
                bounding_box=(-10, 40, 10, 50),
                n_threads=2,
                progress_bar=False,
            )
            == []
        )
        assert len(calls) == 2
        assert cloud[1].call_count == 2  # Initial daily discovery only; the retry uses URLs.
        assert cloud[2].call_args.kwargs["threads"] == 2
        assert retry_download.call_args.kwargs["bounding_box"] == (-10, 40, 10, 50)
        assert all(call.kwargs["bounding_box"] == (-10, 40, 10, 50) for call in cloud[1].call_args_list)


@pytest.mark.parametrize("remove_corrupted", [True, False])
def test_filename_corruption_retry_keeps_options(cloud, tmp_path, remove_corrupted, mocker):
    retry_download = mocker.spy(dl, "download_files")
    real_download = cloud[2].side_effect
    calls = []

    def initially_corrupted(granules, local_path, **kwargs):
        calls.append(local_path)
        if len(calls) == 1:
            path = Path(local_path) / FILENAME
            path.write_bytes(b"corrupted")
            return [path]
        return real_download(granules, local_path, **kwargs)

    cloud[2].side_effect = initially_corrupted
    with gpm.config.set({"base_dir": str(tmp_path / "GPM")}):
        assert (
            dl.download_files(
                FILENAME,
                storage="EARTHDATA",
                retry=2,
                verbose=False,
                remove_corrupted=remove_corrupted,
                bounding_box=(-10, 40, 10, 50),
            )
            == []
        )
    assert len(calls) == 2
    assert retry_download.call_count == 2
    assert all(call.kwargs["bounding_box"] == (-10, 40, 10, 50) for call in retry_download.call_args_list)
    cloud[1].assert_not_called()
    assert all(isinstance(url, str) for call in cloud[2].call_args_list for url in call.args[0])


def test_empty_archive(cloud):
    cloud[1].return_value = []
    assert dl.download_archive("2A-DPR", START, END, version=8, storage="EARTHDATA", verbose=False) is None
    cloud[2].assert_not_called()


@pytest.mark.parametrize(
    ("function", "date_args"),
    [
        (dl.download_daily_data, {"year": 2020, "month": 7, "day": 6}),
        (dl.download_monthly_data, {"year": 2020, "month": 7}),
    ],
)
def test_date_wrappers_forward_cloud_options(cloud, tmp_path, function, date_args):
    with gpm.config.set({"base_dir": str(tmp_path / "GPM")}):
        assert (
            function(
                "2A-DPR",
                **date_args,
                version=8,
                storage="EARTHDATA",
                verbose=False,
                bounding_box=(-10, 40, 10, 50),
            )
            == []
        )
    assert cloud[1].call_args.kwargs["bounding_box"] == (-10, 40, 10, 50)
