import xarray as xr
from datatree import DataTree

from gpm_api.dataset import attrs


# Tests for public functions ###################################################


def test_decode_string() -> None:
    """Test decode_string"""

    # Try string including keys (separated by '='). Dataset: 1A GMI
    string = "InputFileNames=GPM.GMIS.20150801.131141073_20150801.131639202.001.SCANRAW\nGPM.GMIS.20150801.131641077_20150801.132139205.001.SCANRAW\nGPM.GMIS.20150801.132141080_20150801.132639209.001.SCANRAW\nGPM.GMIS.20150801.132641084_20150801.133139212.001.SCANRAW\nGPM.GMIS.20150801.133141087_20150801.133639216.001.SCANRAW\nGPM.GMIS.20150801.133641091_20150801.134139219.001.SCANRAW\nGPM.GMIS.20150801.134141094_20150801.134639223.001.SCANRAW\nGPM.GMIS.20150801.134641098_20150801.135139226.001.SCANRAW\nGPM.GMIS.20150801.135141101_20150801.135639230.001.SCANRAW\nGPM.GMIS.20150801.135641105_20150801.140139233.001.SCANRAW\nGPM.GMIS.20150801.140141108_20150801.140639237.001.SCANRAW\nGPM.GMIS.20150801.140641112_20150801.141139240.001.SCANRAW\nGPM.GMIS.20150801.141141115_20150801.141639244.001.SCANRAW\nGPM.GMIS.20150801.141641119_20150801.142139247.001.SCANRAW\nGPM.GMIS.20150801.142141122_20150801.142639251.001.SCANRAW\nGPM.GMIS.20150801.142641126_20150801.143139254.001.SCANRAW\nGPM.GMIS.20150801.143141129_20150801.143639258.001.SCANRAW\nGPM.GMIS.20150801.143641133_20150801.144139261.001.SCANRAW\nGPM.GMIS.20150801.144141136_20150801.144639265.001.SCANRAW\nGPM.GMIS.20150801.144641140_20150801.145139268.001.SCANRAW;\nInputAlgorithmVersions=n/a;\nInputGenerationDateTimes=n/a;\n"

    res = attrs.decode_string(string)

    assert isinstance(res, dict), "String not returned as dictionary"
    assert list(res.keys()) == [
        "InputFileNames",
        "InputAlgorithmVersions",
        "InputGenerationDateTimes",
    ]

    for key, props in res.items():
        assert "\t" not in props, "Tab not removed"
        assert "\n" not in props, "New line not removed"
        assert ";" not in props, "Semicolon not removed"

    # Try without = separator
    string = "2016067040550_52442_CS_2B-GEOPROF_GRANULE_P1_R05_E06_F00.hdf"
    res = attrs.decode_string(string)

    assert isinstance(res, str), "String not returned as string"
    assert res == string, "String altered when should be returned as is"


def test_decode_attrs():
    """Test decode_attrs"""

    nested_dict_string = "\tsubkey_1=value_1;\n\tsubkey_2=value_2;\n"
    initial_dict = {
        "key_1": nested_dict_string,
        "key_2": "value_2",
    }
    expected_dict = {
        "key_1": {
            "subkey_1": "value_1",
            "subkey_2": "value_2",
        },
        "key_2": "value_2",
    }
    returned_dict = attrs.decode_attrs(initial_dict)
    assert returned_dict == expected_dict


def test_get_granule_attrs(monkeypatch):
    """Test get_granule_attrs"""

    # Mock valid keys
    monkeypatch.setattr(
        "gpm_api.dataset.attrs.STATIC_GLOBAL_ATTRS",
        ("key_1", "key_2", "key_3"),
    )
    monkeypatch.setattr(
        "gpm_api.dataset.attrs.GRANULE_ONLY_GLOBAL_ATTRS",
        (),
    )
    monkeypatch.setattr(
        "gpm_api.dataset.attrs.DYNAMIC_GLOBAL_ATTRS",
        (),
    )

    # Test with non-nested dictionary
    dt = DataTree()
    dt.attrs = {
        "key_1": "value_1",
        "invalid_key": "value_2",
    }
    expected_dict = {
        "key_1": "value_1",
    }
    returned_dict = attrs.get_granule_attrs(dt)
    assert returned_dict == expected_dict

    # Test with nested dictionary
    dt.attrs = {
        "base_key_1": "\tkey_1=value_1;\n\tkey_2=value_2;\n",
        "base_key_2": "\tkey_3=value_3;\n\tinvalid_key=value_4;\n",
    }
    expected_dict = {
        "key_1": "value_1",
        "key_2": "value_2",
        "key_3": "value_3",
    }
    returned_dict = attrs.get_granule_attrs(dt)
    assert returned_dict == expected_dict


def test_add_history():
    """Test add_history"""

    ds = xr.Dataset()
    attrs.add_history(ds)
    assert "history" in ds.attrs


# Tests for internal functions #################################################


def test_has_nested_dictionary():
    """Test _has_nested_dictionary"""

    non_nested_dict = {
        "key_1": "value_1",
        "key_2": "value_2",
    }
    assert not attrs._has_nested_dictionary(non_nested_dict)

    nested_dict = {
        "key_1": {
            "subkey_1": "value_1",
            "subkey_2": "value_2",
        },
        "key_2": "value_2",
    }
    assert attrs._has_nested_dictionary(nested_dict)
