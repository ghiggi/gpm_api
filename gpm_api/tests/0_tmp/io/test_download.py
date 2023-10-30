#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 18:08:16 2022

@author: ghiggi
"""
import datetime
import time

import numpy as np

from gpm_api.io.download import curl_pps_cmd, download_archive, run, wget_pps_cmd

####-------------------------------------------------------------------------.
#### Test gpm_api.download function
product = "2A-DPR"
product_type = "RS"
start_time = datetime.datetime(2020, 7, 5, 0, 2, 0)
end_time = datetime.datetime(2020, 7, 6, 0, 4, 0)
version = 7

verbose = True
progress_bar = True
force_download = False
check_integrity = True
remove_corrupted = False  # True
n_threads = 4
transfer_tool = "curl"  # works
transfer_tool = "wget"  # buggy ... especially with lot of threads

l_corrupted = download_archive(
    product=product,
    start_time=start_time,
    end_time=end_time,
    product_type=product_type,
    version=version,
    n_threads=n_threads,
    transfer_tool=transfer_tool,
    progress_bar=progress_bar,
    force_download=force_download,
    check_integrity=check_integrity,
    remove_corrupted=remove_corrupted,
    verbose=verbose,
)

####-------------------------------------------------------------------------.
#### Test single file download with curl and wget
remote_filepath = "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5"
local_filepath1 = "/tmp/dummy1.hdf"
local_filepath2 = "/tmp/dummy2.hdf"

username = "gionata.ghiggi@epfl.ch"
password = username

### WGET
t_i = time.time()
cmd = wget_pps_cmd(remote_filepath, local_filepath1, username, password)
print(cmd)
run([cmd], n_threads=10, progress_bar=True, verbose=True)
t_f = time.time()
t_elapsed = np.round(t_f - t_i, 2)
print(t_elapsed, "seconds")  # 96.6 seconds

### CURL cmd
t_i = time.time()
cmd = curl_pps_cmd(remote_filepath, local_filepath2, username, password)
print(cmd)
run([cmd], n_threads=10, progress_bar=True, verbose=True)
t_f = time.time()
t_elapsed = np.round(t_f - t_i, 2)
print(t_elapsed, "seconds")  # 102.6 seconds


####-------------------------------------------------------------------------.
#### Test multiple download at once
from gpm_api.io.download import get_fpaths_from_fnames, filter_download_list
from gpm_api.io.find import find_filepaths

pps_filepaths = find_filepaths(
    storage="pps",
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    verbose=verbose,
)

local_filepaths = get_fpaths_from_fnames(
    pps_filepaths,
    storage="local",
    product_type=product_type,
)

pps_filepaths, local_filepaths = filter_download_list(
    local_filepaths=local_filepaths,
    remote_filepaths=pps_filepaths,
    force_download=force_download,
)

#### WGET
import tempfile

# Open temporary file
temp_files = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt")
# Create download file list to be read by wget or curl
for pps_path, local_filepath in zip(pps_filepaths[0:2], local_filepaths[0:2]):
    temp_files.write(f"{pps_path} {local_filepath}\n")
temp_files.seek(0)

# Print the path to the temporary file
print(temp_files.name)


password = username
cmd = "".join(
    [
        "wget ",
        "-4 ",
        "--ftp-user=",
        username,
        " ",
        "--ftp-password=",
        password,
        " ",
        "-e robots=off ",  # allow wget to work ignoring robots.txt file
        "-np ",  # prevents files from parent directories from being downloaded
        "-R .html,.tmp ",  # comma-separated list of rejected extensions
        "-nH ",  # don't create host directories
        "-c ",  # continue from where it left
        "--read-timeout=",
        "10",
        " ",  # if no data arriving for 10 seconds, retry
        "--tries=",
        "5",
        " ",  # retry 5 times (0 forever)
        "--show-progress ",
        # "-P ",
        # "4",
        # " ",
        "-i ",
        temp_files.name,
    ]
)

# cmd

# wget -4 --ftp-user=gionata.ghiggi@epfl.ch --ftp-password=gionata.ghiggi@epfl.ch --show-progress -P 1 -i /tmp/tmptj56j27u.txt

# Remove the temporary file
temp_file.close()

###----------------------------
#### - CURL
import tempfile

# Open temporary file
temp_files = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt")
# Create download file list to be read by wget or curl
for pps_path, local_filepath in zip(pps_filepaths[0:2], local_filepaths[0:2]):
    pps_path = pps_path.replace("ftps", "ftp", 1)  # REQUIRED BY CURL
    temp_files.write(f"url = {pps_path}\noutput = {local_filepath}\n")
temp_files.seek(0)

password = username
cmd = "".join(
    [
        "curl ",
        "--verbose ",
        "--ipv4 ",
        "--insecure ",
        "--user ",
        username,
        ":",
        password,
        " ",
        "--ftp-ssl ",
        # Custom settings
        "--header 'Connection: close' ",
        "--connect-timeout 20 ",
        "--retry 5 ",
        "--retry-delay 10 ",
        # "-m 4 ", # number of parallel files  # this cause stuffs to not work
        "-n ",
        "-K ",
        temp_files.name,
    ]
)

print(cmd)

#### ------------------------------------------------------------------------
#### Test download monthly data
from gpm_api.io.download import download_monthly_data

product = "2A-DPR"
product_type = "RS"
year = 2020
month = 7
version = 7
verbose = True
progress_bar = True
force_download = False
check_integrity = True
remove_corrupted = False  # True
n_threads = 4
transfer_tool = "curl"  # works
transfer_tool = "wget"  # buggy ... especially with lot of threads

l_corrupted = download_monthly_data(
    product=product,
    year=year,
    month=month,
    product_type=product_type,
    version=version,
    n_threads=n_threads,
    transfer_tool=transfer_tool,
    progress_bar=progress_bar,
    force_download=force_download,
    check_integrity=check_integrity,
    remove_corrupted=remove_corrupted,
    verbose=verbose,
)


#### ------------------------------------------------------------------------
#### Test download GPM IMERG
import datetime
import gpm_api


start_time = datetime.datetime.strptime("2019-07-13 11:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2019-07-13 13:00:00", "%Y-%m-%d %H:%M:%S")
product = "IMERG-FR"  # 'IMERG-ER' 'IMERG-LR'
product_type = "RS"
version = 6
n_threads = 1
force_download = False
verbose = True
progress_bar = True
check_integrity = True
transfer_tool = "curl"
remove_corrupted = True


# Download the data
gpm_api.download(
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    force_download=force_download,
    verbose=verbose,
    progress_bar=progress_bar,
    check_integrity=check_integrity,
    n_threads=n_threads,
)
