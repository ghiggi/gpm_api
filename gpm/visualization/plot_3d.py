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
"""This module contains functions for 3D visualization of GPM-API RADAR data."""
import numpy as np

# TODO:
# - Isosurface contour buggy at low reflectivity
#   --> Should I replace values as 0-1 at each round?
# - 3D terrain
# - surface on bin surface height


def create_pyvista_2d_surface(data_array, spacing=(1, 1, 1), origin=(0, 0, 0)):
    """Create pyvista ImageData object from 2D xr.DataArray."""
    import pyvista as pv

    dimensions = (data_array.shape[0], data_array.shape[1], 1)
    surf = pv.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
    )
    data = data_array.to_numpy()
    data = data[:, ::-1]
    surf.point_data.set_array(data.flatten(order="F"), name=data_array.name)
    surf.set_active_scalars(data_array.name)
    return surf


def create_pyvista_3d_volume(data_array, spacing=(1, 1, 0.25), origin=(0, 0, 0)):
    """Create pyvista ImageData object from 3D xr.DataArray."""
    import pyvista as pv

    # Remove vertical areas without values
    data_array = data_array.gpm.slice_range_with_valid_data()

    # Create ImageData object
    # - TODO: scale (factor)
    dimensions = data_array.shape
    vol = pv.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
    )
    data = data_array.to_numpy()
    data = data[:, ::-1, ::-1]
    vol.point_data.set_array(data.flatten(order="F"), name=data_array.name)
    vol.set_active_scalars(data_array.name)
    return vol


def get_slider_button_positions(i, num_buttons, spacing_factor=0.04):
    """Return the pointa and pointb parameters for pl.add_slider_widget."""
    if num_buttons < 1:
        raise ValueError("Number of buttons must be at least 1")

    # Define margin
    min_pointa = 0.025
    max_pointb = 0.98

    # Define allowable buttons width
    total_width = max_pointb - min_pointa - spacing_factor * (num_buttons - 1)

    # Define button width
    button_width = total_width / num_buttons

    # Define pointa and pointb
    start_x = min_pointa + i * (button_width)
    if i > 0:
        start_x = start_x + spacing_factor
    end_x = start_x + button_width
    pointa = (start_x, 0.1)
    pointb = (end_x, 0.1)
    return pointa, pointb


class OpacitySlider:
    """Opacity Slider for pyvista pl.add_slider_widget."""

    def __init__(self, pl_actor):
        self.pl_actor = pl_actor

    def __call__(self, value):
        self.pl_actor.GetProperty().SetOpacity(value)


def add_3d_isosurfaces(
    vol,
    pl,
    isovalues=[30, 40, 50],
    opacities=[0.3, 0.5, 1],
    method="contour",
    style="surface",
    add_sliders=False,
    **mesh_kwargs,
):
    """Add 3D isosurface to a pyvista plotter object.

    If add_sliders=True, isosurface opacity can be adjusted interactively.

    """
    # Checks
    if len(isovalues) != len(opacities):
        raise ValueError("Expected same number of isovalues and opacities values.")
    # TODO: check there are values larger than max isovalues

    # Define opacity dictionary
    dict_opacity = dict(zip(isovalues, opacities))

    # Precompute isosurface
    dict_isosurface = {isovalue: vol.contour([isovalue], method=method) for isovalue in isovalues}
    n_isosurfaces = len(dict_isosurface)
    # Add isosurfaces
    for i, (isovalue, isosurface) in enumerate(dict_isosurface.items()):
        pl_actor = pl.add_mesh(
            isosurface,
            opacity=dict_opacity[isovalue],
            style=style,
            **mesh_kwargs,
        )
        if add_sliders:
            # Define opacity slider
            opacity_slider = OpacitySlider(pl_actor)
            # Define slicer button position
            pointa, pointb = get_slider_button_positions(i=i, num_buttons=n_isosurfaces)
            # Add slider widget
            pl.add_slider_widget(
                callback=opacity_slider,
                rng=[0, 1],
                value=dict_opacity[isovalue],
                title=f"Isovalue={isovalue}",
                pointa=pointa,
                pointb=pointb,
                style="modern",
            )


class IsosurfaceSlider:
    def __init__(self, vol, method="contour", isovalue=None):
        """Define pyvista slider for 3D isosurfaces."""
        self.vol = vol
        vmin, vmax = vol.get_data_range()
        self.vmin = vmin
        self.vmax = vmax
        self.method = method
        if isovalue is None:
            isovalue = vmin + (vmax - vmin) / 2
        self.isovalue = isovalue
        self.isosurface = vol.contour([isovalue], method=method)

    def __call__(self, value):
        self.isovalue = value
        self.update()

    def update(self):
        result = self.vol.contour([self.isovalue], method=self.method)
        self.isosurface.copy_from(result)


def add_3d_isosurface_slider(vol, pl, method="contour", isovalue=None, **mesh_kwargs):
    """Add a 3D isosurface slider enabling to slide through the 3D volume."""
    isosurface_slider = IsosurfaceSlider(vol, method=method, isovalue=isovalue)
    isosurface = isosurface_slider.isosurface
    pl.add_mesh(isosurface, **mesh_kwargs)
    pl.add_slider_widget(
        callback=isosurface_slider,
        rng=vol.get_data_range(),
        value=isosurface_slider.isovalue,
        title="Isosurface",
        pointa=(0.4, 0.9),
        pointb=(0.9, 0.9),
        style="modern",
    )


class OrthogonalSlicesSlider:
    def __init__(self, vol, x=1, y=1, z=1):
        """Define pyvista sliders for 3D orthogonal slices."""
        self.vol = vol
        self.slices = vol.slice_orthogonal(x=x, y=y, z=z)
        # Set default parameters
        self.kwargs = {
            "x": x,
            "y": y,
            "z": z,
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        result = self.vol.slice_orthogonal(**self.kwargs)
        self.slices[0].copy_from(result[0])
        self.slices[1].copy_from(result[1])
        self.slices[2].copy_from(result[2])


def add_3d_orthogonal_slices(vol, pl, x=None, y=None, z=None, add_sliders=False, **mesh_kwargs):
    """Add 3D orthogonal slices with interactive sliders."""
    # Define bounds
    x_rng = vol.bounds[0:2]
    y_rng = vol.bounds[2:4]
    z_rng = vol.bounds[4:6]

    # Define default values if not provided
    # - If value is 0, means no slice plotted !
    if x is None:
        x = int(np.diff(x_rng) / 2)
    if y is None:
        y = int(np.diff(y_rng) / 2)
    if z is None:
        z = int(z_rng[0] + 0.01)

    # Define orthogonal slices (and sliders)
    if add_sliders:
        orthogonal_slices_slider = OrthogonalSlicesSlider(vol)
        orthogonal_slices = orthogonal_slices_slider.slices
    else:
        orthogonal_slices = vol.slice_orthogonal(x=x, y=y, z=z, progress_bar=False)

    # Display orthogonal slices
    pl.add_mesh(orthogonal_slices, **mesh_kwargs)

    # Add slider widgets
    if add_sliders:
        pl.add_slider_widget(
            callback=lambda value: orthogonal_slices_slider("x", int(value)),
            rng=x_rng,
            value=x,
            title="Along-Track",
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
            style="modern",
        )
        pl.add_slider_widget(
            callback=lambda value: orthogonal_slices_slider("y", int(value)),
            rng=y_rng,
            value=y,
            title="Cross-Track",
            pointa=(0.35, 0.1),
            pointb=(0.64, 0.1),
            style="modern",
        )
        pl.add_slider_widget(
            callback=lambda value: orthogonal_slices_slider("z", value),
            rng=z_rng,
            value=z,
            title="Elevation",
            pointa=(0.67, 0.1),
            pointb=(0.98, 0.1),
            style="modern",
        )
