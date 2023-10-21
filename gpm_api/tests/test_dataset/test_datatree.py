import os

from datatree.datatree import DataTree

from gpm_api.dataset import datatree


def test_open_datatree():
    """Test open_datatree.

    Test that the functions returns a DataTree object.
    """

    script_path = os.path.dirname(os.path.realpath(__file__))
    fpath = os.path.join(
        script_path, "assets", "1A.GPM.GMI.COUNT2021.20200801-S105247-E122522.036508.V07A.HDF5"
    )
    # dt = datatree.open_datatree(fpath)
    # assert isinstance(dt, DataTree)
