# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""Check GPM-API configuration files."""
import os  # noqa

import pytest
from unittest import mock


def test_donfig_takes_environment_variable():
    """Test that the donfig config file takes the environment defaults."""
    from importlib import reload

    import gpm

    with mock.patch.dict("os.environ", {"GPM_BASE_DIR": "/my_path_to/GPM"}):
        reload(gpm._config)
        reload(gpm)
        assert gpm.config.get("base_dir") == "/my_path_to/GPM"


def test_donfig_takes_config_yaml_file(tmp_path, mocker):
    """Test that the donfig config file takes the YAML defaults."""
    from importlib import reload

    import gpm

    # Mock to save config YAML at custom location
    config_fpath = str(tmp_path / ".config_gpm.yaml")
    mocker.patch("gpm.configs._define_config_filepath", return_value=config_fpath)

    # Initialize config YAML
    gpm.configs.define_configs(base_dir="test_dir/GPM")

    reload(gpm._config)
    reload(gpm)
    assert gpm.config.get("base_dir") == "test_dir/GPM"


CONFIGS_TEST_KWARGS = {
    "base_dir": "test_base_dir",
    "username_pps": "test_username_pps",
    "password_pps": "test_password_pps",
    "username_earthdata": "test_username_earthdata",
    "password_earthdata": "password_earthdata",
}


@pytest.mark.parametrize("key_value", list(CONFIGS_TEST_KWARGS.items()))
def test_donfig_context_manager(key_value):
    """Test that the donfig context manager works as expected."""
    import gpm

    key = key_value[0]
    value = key_value[1]

    # Assert donfig key context manager
    with gpm.config.set({key: value}):
        assert gpm.config.get(key) == value

    # # Assert if not initialized, defaults to None
    # assert gpm.config.get(key) is None

    # Now initialize
    gpm.config.set({key: value})
    assert gpm.config.get(key) == value

    # Now try context manager again
    with gpm.config.set({key: "new_value"}):
        assert gpm.config.get(key) == "new_value"
    assert gpm.config.get(key) == value
