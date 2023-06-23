#!/usr/bin/env python3
"""
Created on Fri Jan 13 16:41:17 2023

@author: ghiggi
"""
# Software: ximage (already on pypi), ximager, x-image, xbatcher package?

## Labels
# ximage.label_object
# ximage.get_label_stats
# ximage.plot_label            [FULL IMAGE]
# ximage.plot_label_patches()  [LABEL PATCHES]

####--------------------------------------------------------------------------.
###########################
#### Image-Extraction #####
###########################
#### LabelsPatchGenerator
# Around Label point:
#     - Centroid , Center of Mass] [REQUIRE LABEL]
#     - Min, Max[OPTIONAL LABEL, BUT COULD BE USED TO MASK OUTSIDE LABEL]

# Over image/ labels bbox
# --> Sliding
# --> Tiling / Splitting
# --> RandomSampling (number)
# --> BlockSampling  (number)

####--------------------------------------------------------------------------.
#### Tiling/Sliding/Sampling
# xarray patch sampling
# - sampling: RandomSampling, BlockSampling
# - number_patches
# - random_state

# xarray sliding (rolling)
# - size
# - step / stride = 1

# xarray tiling
# - size
# - step / stride = 1
# - overlaps = 0
# --> tile_merging (when overlap --> choose [max, min, avg, sd])

# --> Add to dask-image tool ?
# --> Wrapper into xarray package

# Discussion https://github.com/xarray-contrib/xbatcher/discussions/new?category=ideas
# -->  xbatcher for image/ndarray tiling/sliding and patch sampling


# xbatcher
# - input_overlap (1 for sliding, 0 for split)
# - https://xbatcher.readthedocs.io/en/latest/
# - https://xbatcher.readthedocs.io/en/latest/demo.html
# - BatchSchema: https://xbatcher.readthedocs.io/en/latest/generated/xbatcher.BatchSchema.html
# - https://github.com/xarray-contrib/xbatcher/discussions/78
# --> https://figshare.com/articles/presentation/Xbatcher_-_A_Python_Package_That_Simplifies_Feeding_Xarray_Data_Objects_to_Machine_Learning_Libraries/22264072/1

# Patch coordinate (0,0), (0,1), ...

####--------------------------------------------------------------------------.
# tiler
# - https://github.com/the-lay/tiler

# patchify
# - https://github.com/dovahcrow/patchify.py

# EMPatches
# - https://github.com/Mr-TalhaIlyas/EMPatches

# numpy.sliding_window_view (sliding)
# - no step/striding
# - https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html

# dask-array.sliding_window_view (sliding)
# - no step/striding
# - https://docs.dask.org/en/latest/generated/dask.array.lib.stride_tricks.sliding_window_view.html

# skimage.util.view_as_windows (sliding)
# - has step / striding
# - https://scikit-image.org/docs/stable/api/skimage.util.html#view-as-windows

# cucim.skimage.util.view_as_windows (sliding)
# - has step / striding
# - https://docs.rapids.ai/api/cucim/stable/api/#cucim.skimage.util.view_as_windows

# skimage.util.view_as_blocks (tiling)
# - no overlap allowed
# - https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.view_as_blocks

# cucim.skimage.util.view_as_blocks
# - no overlap allowed
# - https://docs.rapids.ai/api/cucim/stable/api/#cucim.skimage.util.view_as_blocks

# pytorch fold/unfold
# --> https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
# --> https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html#torch.Tensor.unfold

# pytorch_toolbel - ImageSlicer
# https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/inference/tiles.py#L54

# Blended-Tiling
# https://github.com/ProGamerGov/blended-tiling

# sklearn.feature_extraction.PatchExtractor
# - https://github.com/scikit-learn/scikit-learn/blob/baf828ca1/sklearn/feature_extraction/image.py#L453

# torchgeo.samplers.tile_to_chips
# - https://torchgeo.readthedocs.io/en/stable/api/samplers.html#torchgeo.samplers.tile_to_chips

#### Samplers
# Sampling tasks
# - https://eo-learn.readthedocs.io/en/latest/eotasks.html#ml-tools
# - https://torchgeo.readthedocs.io/en/stable/api/samplers.html

####--------------------------------------------------------------------------.
