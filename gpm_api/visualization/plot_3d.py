#!/usr/bin/env python3
"""
Created on Fri Aug 18 15:34:19 2023

@author: ghiggi
"""

import pyvista as pv


def create_pyvista_2d_surface(data_array, spacing=(1, 1, 1), origin=(0, 0, -4)):

    # Create ImageData object
    dimensions = (data_array.shape[0], data_array.shape[1], 1)
    surf = pv.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
    )
    data = data_array.values
    data = data[:, ::-1]
    surf.point_data.set_array(data.flatten(order="F"), name=data_array.name)
    surf.set_active_scalars(data_array.name)
    return surf


def create_pyvista_3d_volume(data_array, spacing=(1, 1, 0.25), origin=(0, 0, 0)):

    # Remove vertical areas without values
    data_array = data_array.gpm_api.slice_range_with_valid_data()

    # Create ImageData object
    dimensions = data_array.shape
    vol = pv.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
    )
    data = data_array.values
    data = data[:, ::-1, ::-1]
    vol.point_data.set_array(data.flatten(order="F"), name=data_array.name)
    vol.set_active_scalars(data_array.name)
    return vol


def plot_3d_radar_isosurface(
    ds,
    radar_frequency="Ku",
    cmap="Spectral_r",
    clim=[10, 50],
    isovalues=[30, 40, 50],
    opacities=[0.3, 0.5, 1],
    method="contour",
    background_color="#282727",
    add_colorbar=False,
):
    # Checks
    # - Smaller than ... to not freeze laptop
    # - check length opacities = len isovalues
    # - check there are values larger than max isovalues

    # Retrieve pyvista surface and volume data objects
    da_2d = ds["zFactorFinalNearSurface"].sel(radar_frequency=radar_frequency).compute()
    surf = create_pyvista_2d_surface(da_2d, spacing=(1, 1, 1), origin=(0, 0, -3))

    da_3d = ds["zFactorFinal"].sel(radar_frequency=radar_frequency).compute()
    vol = create_pyvista_3d_volume(da_3d, spacing=(1, 1, 0.25), origin=(0, 0, 0))

    # Define opacity dictionary
    dict_opacity = dict(zip(isovalues, opacities))

    # Precompute isosurface
    dict_isosurface = {isovalue: vol.contour([isovalue], method=method) for isovalue in isovalues}

    # Display
    pl = pv.Plotter()
    pl.background_color = background_color
    pl.add_mesh(surf, opacity=1, cmap=cmap, show_scalar_bar=False, style="surface")
    for isovalue, isosurface in dict_isosurface.items():
        pl.add_mesh(
            isosurface,
            opacity=dict_opacity[isovalue],
            clim=clim,
            cmap=cmap,
            show_scalar_bar=add_colorbar,
            style="surface",
        )
    pl.show()
