import pytest
import os
import datetime
import ftplib
from typing import Any
from pytest_mock.plugin import MockerFixture
from gpm_api.io import download as dl
from gpm_api.io.products import available_products


def test_construct_curl_cmd(
    username: str,
    password: str,
    server_paths: list[str],
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
    server_paths: list[str],
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


# May be better to test child functions... (get_disk_directory, get_start_time_from_filepaths)

# def test_convert_pps_to_disk_filepaths(
#     server_paths: list[str],
#     products: list[str],
#     product_categories: list[str],
#     product_types: list[str],
#     versions: list[int],
#     tmpdir: str,
# ) -> None:
#     """Test that the PPS filepaths are correctly converted to disk filepaths"""

#     base_dir = os.path.join(
#         tmpdir,
#         datetime.datetime.utcnow().isoformat(),
#     )

#     # for product_type in product_types:
#     #     for product in available_products(product_type=product_type):
#     #         for version in versions:
#     disk_filepaths = dl.convert_pps_to_disk_filepaths(
#         pps_filepaths=server_paths,
#         base_dir=base_dir,
#         product=product,
#         product_type=product_type,
#         version=version,
#     )
#     for i, server_path in enumerate(server_paths):
#         # Remote URI scheme
#         split_no_uri = server_path.split("://")[-1]

#         _uri, _gpm, y, m, d, categ, file = split_no_uri.split("/")

#         # Assertions assume the list order is identical
#         if product_type == "RS":
#             assert disk_filepaths[i] == os.path.join(
#                 base_dir,
#                 "GPM",
#                 product_type,
#                 f"V0{version}",
#                 categ,
#                 product,
#                 y,
#                 m,
#                 d,
#                 file,
#             )
#         elif product_type == "NRT":
#             assert disk_filepaths[i] == os.path.join(
#                 base_dir,
#                 "GPM",
#                 product_type,
#                 categ,
#                 product,
#                 y,
#                 m,
#                 d,
#                 file,
#             )

# def _download_with_ftlib(server_path, disk_path, username, password):
#     # Infer hostname
#     hostname = server_path.split("/", 3)[2]  # remove ftps:// and select host

#     # Remove hostname from server_path
#     server_path = server_path.split("/", 3)[3]

#     # Connect to the FTP server using FTPS
#     ftps = ftplib.FTP_TLS(hostname)

#     # Login to the FTP server using the provided username and password
#     ftps.login(username, password)  # /gpmdata base directory

#     # Download the file from the FTP server
#     try:
#         with open(disk_path, "wb") as file:
#             ftps.retrbinary(f"RETR {server_path}", file.write)
#     except EOFError:
#         return f"Impossible to download {server_path}"

#     # Close the FTP connection
#     ftps.close()
#     return None


def test_download_with_ftplib(
    mocker: MockerFixture,
    server_paths: dict[str, dict[str, Any]],
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
