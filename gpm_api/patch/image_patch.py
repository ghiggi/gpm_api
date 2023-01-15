#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:41:17 2023

@author: ghiggi
"""
####--------------------------------------------------------------------------.
# TODO: to place in a separate xpatch / ximage package 
## Labels
# ximage.label_object
# ximage.get_label_stats
# ximage.plot_label            [FULL IMAGE]
# ximage.plot_label_patches()  [LABEL PATCHES]

## Image 

####--------------------------------------------------------------------------.
#################################
#### Patch-Image-Extraction #####
#################################
# sliding_window_view
# numpy 
# - https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
# dask-array 
# https://docs.dask.org/en/latest/generated/dask.array.lib.stride_tricks.sliding_window_view.html
# skimage.util.view_as_windows
# - https://scikit-image.org/docs/stable/api/skimage.util.html#view-as-windows

# sklearn.feature_extraction.PatchExtractor 
# - https://github.com/scikit-learn/scikit-learn/blob/baf828ca1/sklearn/feature_extraction/image.py#L453

# patchify 
# - https://github.com/dovahcrow/patchify.py

# EMPatches 
# - https://github.com/Mr-TalhaIlyas/EMPatches

# xbatcher 
# - input_overlap (1 for sliding, 0 for split)
# - https://xbatcher.readthedocs.io/en/latest/
# - https://github.com/xarray-contrib/xbatcher/discussions/78

# --> generator 
# --> xarray patch tensor 

# ImagePatchGenerator 
# --> Around Label point: 
#     - Centroid , Center of Mass] [REQUIRE LABEL]
#     - Min, Max[OPTIONAL LABEL, BUT COULD BE USED TO MASK OUTSIDE LABEL]
# --> Splitting 
# --> Sliding 
# --> RandomSampling (number)
# --> BlockSampling 

# --> Extract patch sample tensor 

####--------------------------------------------------------------------------.