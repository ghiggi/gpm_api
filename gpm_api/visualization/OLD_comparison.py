#!/usr/bin/env python3
"""
Created on Fri Sep  9 12:14:03 2022

@author: ghiggi
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import gpm_api


def compare_products(
    base_dir,
    start_time,
    end_time,
    bbox,
    product_variable_1,
    product_variable_2,
    version=7,
    product_type="RS",
):

    #### Preprocess bbox
    # - Enlarge so to be sure to include all data when cropping
    # TODO: Robustify to deal when close to the antimeridian
    bbox_extent = bbox.copy()
    bbox = [bbox[0] - 5, bbox[1] + 5, bbox[2] - 5, bbox[3] + 5]

    ####----------------------------------------------------------------------.
    #### Load data
    product_var_dict = {}
    product_var_dict.update()
    list_tuple = list(product_variable_1.items()) + list(product_variable_2.items())

    list_da = []
    for product, variable in list_tuple:
        # Open dataset
        ds = gpm_api.open_dataset(
            base_dir=base_dir,
            product=product,
            start_time=start_time,
            end_time=end_time,
            # Optional
            variables=variable,
            version=version,
            product_type=product_type,
            chunks="auto",
            decode_cf=True,
            prefix_group=False,
        )

        # Crop dataset
        ds = ds.gpm_api.crop(bbox)
        # Append to list
        list_da.append(ds[variable])

    ####----------------------------------------------------------------------.
    #### Define figure
    dpi = 300
    figsize = (7, 2.8)
    crs_proj = ccrs.PlateCarree()

    # TODO: 3 subplots... colormap on third axis

    # Create figure
    fig, axs = plt.subplots(
        1,
        2,
        subplot_kw={"projection": crs_proj},
        gridspec_kw={"width_ratios": [0.44, 0.56]},
        figsize=figsize,
        dpi=dpi,
    )

    ####----------------------------------------------------------------------.
    #### First Map
    ax = axs[0]

    # Retrieve DataArray
    da = list_da[0]

    # - Plot map
    da.gpm_api.plot(ax=ax, add_colorbar=False)

    # - Set title
    title = da.gpm_api.title(time_idx=0, add_timestep=False)
    ax.set_title(title)

    # - Set extent
    ax.set_extent(bbox_extent)

    ####----------------------------------------------------------------------.
    #### Second Map
    ax = axs[1]

    # Retrieve DataArray
    da = list_da[1]

    # - Plot map
    da.gpm_api.plot(ax=ax, add_colorbar=True)

    # TODO: REMOVE Y AXIS
    # ax.set_yticklabels(None)
    #  ax.axes.get_yaxis().set_visible(False)

    # - Set title
    title = da.gpm_api.title(time_idx=0, add_timestep=False)
    ax.set_title(title)

    # - Set extent
    ax.set_extent(bbox_extent)

    ax.get_ygridlines()

    ####----------------------------------------------------------------------.
    fig.tight_layout()
    return fig
    ####----------------------------------------------------------------------.


# def compare_products(base_dir,
#                      start_time,
#                      end_time,
#                      bbox,
#                      product_variable_1,
#                      product_variable_2,
#                      version = 7,
#                      product_type = "RS",
#                     ):

#     #### Preprocess bbox
#     # - Enlarge so to be sure to include all data when cropping
#     # TODO: Robustify to deal when close to the antimeridian
#     bbox_extent = bbox.copy()
#     bbox = [bbox[0] - 5, bbox[1] + 5, bbox[2] - 5, bbox[3] + 5]

#     ####----------------------------------------------------------------------.
#     #### Load data
#     product_var_dict = {}
#     product_var_dict.update()
#     list_tuple = list(product_variable_1.items()) + list(product_variable_2.items())

#     list_da = []
#     for product, variable in list_tuple:
#         # Open dataset
#         ds = gpm_api.open_dataset(base_dir=base_dir,
#                          product=product,
#                          start_time=start_time,
#                          end_time=end_time,
#                          # Optional
#                          variables=variable,
#                          version=version,
#                          product_type=product_type,
#                          chunks="auto",
#                          decode_cf = True,
#                          prefix_group = False)

#         # Crop dataset
#         ds = ds.gpm_api.crop(bbox)
#         # Append to list
#         list_da.append(ds[variable])

#     ####----------------------------------------------------------------------.
#     #### Define figure
#     dpi = 300
#     figsize = (7, 2.8)
#     crs_proj = ccrs.PlateCarree()

#     from mpl_toolkits.axes_grid1 import AxesGrid
#     from cartopy.mpl.geoaxes import GeoAxes

#     # Create figure
#     fig = plt.figure(figsize=figsize, dpi=dpi)

#     axes_class = (GeoAxes,
#                   dict(map_projection=crs_proj))

#     axs = AxesGrid(fig, 111, axes_class=axes_class,
#                    nrows_ncols=(1, 2),
#                    axes_pad=0.6,
#                    cbar_location='right',
#                    cbar_mode='single',
#                    cbar_pad=0.2,
#                    cbar_size='6%',
#                    label_mode='')  # note the empty label_mode
#     ####----------------------------------------------------------------------.
#     #### First Map
#     ax = axs[0]

#     # Retrieve DataArray
#     da = list_da[0]

#     # - Plot map
#     p = da.gpm_api.plot(ax=ax, add_colorbar=False)

#     # - Set title
#     title = da.gpm_api.title(time_idx=0, add_timestep=False)
#     ax.set_title(title)

#     # - Set extent
#     ax.set_extent(bbox_extent)

#     ####----------------------------------------------------------------------.
#     #### Second Map
#     ax = axs[1]

#     # Retrieve DataArray
#     da = list_da[1]

#     # - Plot map
#     p = da.gpm_api.plot(ax=ax, add_colorbar=False)

#     # TODO: REMOVE Y AXIS
#     # ax.set_yticklabels(None)
#     #  ax.axes.get_yaxis().set_visible(False)

#     # - Set title
#     title = da.gpm_api.title(time_idx=0, add_timestep=False)
#     ax.set_title(title)

#     # - Set extent
#     ax.set_extent(bbox_extent)

#     ax.get_ygridlines()

#     ####----------------------------------------------------------------------.
#     ### Add colorbar
#     axs.cbar_axes[0].colorbar(p)

#     ####----------------------------------------------------------------------.
#     fig.tight_layout()
#     return fig
#     ####----------------------------------------------------------------------.
