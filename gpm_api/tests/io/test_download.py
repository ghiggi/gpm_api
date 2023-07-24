import pytest
import os
import datetime
from gpm_api.io import download as dl


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
