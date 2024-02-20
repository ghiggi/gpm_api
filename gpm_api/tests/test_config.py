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
import os

import pytest

CONFIGS_TEST_KWARGS = {
    "base_dir": "test_base_dir",
    "username_pps": "test_username_pps",
    "password_pps": "test_password_pps",
    "username_earthdata": "test_username_earthdata",
    "password_earthdata": "password_earthdata",
}


def test_define_configs(tmp_path, mocker):
    """Test define_configs function."""
    import gpm_api

    # Mock to save config YAML at custom location
    config_filepath = str(tmp_path / ".config_gpm_api.yml")
    mocker.patch("gpm_api.configs._define_config_filepath", return_value=config_filepath)

    # Define config YAML
    gpm_api.configs.define_configs(**CONFIGS_TEST_KWARGS)
    assert os.path.exists(tmp_path / ".config_gpm_api.yml")


def test_read_configs(tmp_path, mocker):
    """Test read_configs function."""
    from gpm_api.configs import define_configs, read_configs

    # Mock to save config YAML at custom location
    config_filepath = str(tmp_path / ".config_gpm_api.yml")
    mocker.patch("gpm_api.configs._define_config_filepath", return_value=config_filepath)

    # Define config YAML
    define_configs(**CONFIGS_TEST_KWARGS)
    assert os.path.exists(tmp_path / ".config_gpm_api.yml")

    # Read config YAML
    config_dict = read_configs()
    assert isinstance(config_dict, dict)
    print(config_dict)
    assert config_dict["base_dir"] == "test_base_dir"


def test_update_gpm_api_configs(tmp_path, mocker):
    """Test define_configs function in 'update' mode."""
    import gpm_api
    from gpm_api.utils.yaml import read_yaml

    # Mock to save config YAML at custom location
    config_filepath = str(tmp_path / ".config_gpm_api.yml")
    mocker.patch("gpm_api.configs._define_config_filepath", return_value=config_filepath)

    # Initialize
    gpm_api.configs.define_configs(**CONFIGS_TEST_KWARGS)
    assert os.path.exists(config_filepath)

    # Read
    config_dict = read_yaml(config_filepath)
    assert config_dict["base_dir"] == "test_base_dir"

    # Update
    gpm_api.configs.define_configs(
        base_dir="new_test_base_dir", username_pps="new_username", password_pps="new_password"
    )
    assert os.path.exists(config_filepath)
    config_dict = read_yaml(config_filepath)
    assert config_dict["base_dir"] == "new_test_base_dir"
    assert config_dict["username_pps"] == "new_username"


def test_get_base_dir():
    """Test get_base_dir function."""
    import gpm_api
    from gpm_api.configs import get_base_dir

    # Check that if input is not None, return the specified base_dir
    assert get_base_dir(base_dir="test/gpm_api") == "test/gpm_api"

    # Check that if no config YAML file specified (base_dir=None), raise error
    with gpm_api.config.set({"base_dir": None}):
        with pytest.raises(ValueError):
            get_base_dir()

    # Set base_dir in the donfig config and check it return it !
    gpm_api.config.set({"base_dir": "another_test_dir/gpm_api"})
    assert get_base_dir() == "another_test_dir/gpm_api"

    # Now test that return the one from the temporary gpm_api.config donfig object
    with gpm_api.config.set({"base_dir": "new_test_dir/gpm_api"}):
        assert get_base_dir() == "new_test_dir/gpm_api"

    # And check it return the default one
    assert get_base_dir() == "another_test_dir/gpm_api"


@pytest.mark.parametrize("key_value", list(CONFIGS_TEST_KWARGS.items()))
def test_get_argument_value(key_value):
    import gpm_api
    from gpm_api import configs as configs_module

    key = key_value[0]
    value = key_value[1]

    function_name = f"get_{key}"
    function = getattr(configs_module, function_name)

    # Check raise error if key is not specified
    gpm_api.config.set({key: None})
    with pytest.raises(ValueError):
        function()

    # Check returns the value specified in the donfig file
    gpm_api.config.set({key: value})
    assert function() == value
