import os
from multiprocessing import Pool

import numpy as np
from shapely.strtree import STRtree

# TODO:
# aggregator.subset (return aggregator over smaller area, stree not recomputed !)
# Currently using STRTree
# Maybe use pykdtree to filter out by distance first ?

# Point Aggregator
# LineAggregator
# --> Useful also to rasterize !

# Other libraries
# - rasterstats: https://pythonhosted.org/rasterstats/
# - xarray-spatial: https://xarray-spatial.readthedocs.io/en/stable/user_guide/zonal.html
# - pyscissor: https://github.com/nzahasan/pyscissor
# - zonal_statistics: https://corteva.github.io/geocube/stable/examples/zonal_statistics.html


# --------------------------------------------------------------------.
# def _process_subset(arg):
#     """Function to process a subset of target polygons."""
#     str_tree, subset = arg
#     return {i: _get_dst_agg_info(str_tree, target_poly) for i, target_poly in subset}

# # Split the target polygons into subsets
# subsets = np.array_split(list(enumerate(self.target_polygons)), n_cpus)
# args = [(self.str_tree, subset) for subset in subsets]

# # Use multiprocessing to process subsets in parallel
# # multiprocessing.set_start_method('fork')
# with Pool(processes=n_cpus) as pool:
#     results = pool.map(_process_subset, args)

# # Merge the results into a single dictionary
# dict_info = {i: result for subset in results for i, result in subset.items()}


def _get_dst_agg_info(str_tree, target_poly):
    """
    Get intersecting source polygons and their intersection areas with the target polygon.

    Parameters
    ----------
    target_poly : Polygon
        Target polygon.

    Returns
    -------
    tuple
        Indices of intersecting source polygons and their intersection areas.
    """
    indices = str_tree.query(target_poly, predicate="intersects")
    intersection_areas = []
    for src_poly in str_tree.geometries.take(indices).tolist():
        poly_intersection = src_poly.intersection(target_poly)
        intersection_area = poly_intersection.area
        intersection_areas.append(intersection_area)
    return indices, np.array(intersection_areas)


def _get_intersection_area(source_poly, target_poly):
    poly_intersection = source_poly.intersection(target_poly)
    intersection_area = poly_intersection.area
    return intersection_area


def _get_target_poly_info(target_poly, source_indices, source_polygons):
    intersection_areas = [
        _get_intersection_area(source_poly=source_polygons[source_index], target_poly=target_poly)
        for source_index in source_indices
    ]
    return [source_indices, np.array(intersection_areas)]


def _create_dict_info(dict_indices, target_polygons, source_polygons):
    dict_info = {
        target_index: _get_target_poly_info(
            target_poly=target_polygons[target_index],
            source_indices=source_indices,
            source_polygons=source_polygons,
        )
        for target_index, source_indices in dict_indices.items()
    }
    return dict_info


def _create_dict_info_parallel(arg):
    return _create_dict_info(*arg)


def create_dict_info(str_tree, target_polygons, source_polygons, parallel=True):
    # Retrieve target-source intersection indices
    # - Target/Sources Indices are not returned for empty and non-intersecting polygons !
    target_indices, source_indices = str_tree.query(target_polygons, predicate="intersects")

    list_source_indices = np.split(source_indices, np.unique(target_indices, return_index=True)[1])[1:]
    target_indices = np.unique(target_indices)

    # Check there are some geometries intersecting
    if target_indices.size == 0:
        raise ValueError("No intersecting geometries.")

    # Now retrieve also the intersection areas and return the dictionary info
    if parallel:  # split and process in parallel n_cpus/2 tasks
        n_cpus = int(os.cpu_count() / 2)
        target_indices_subsets = np.array_split(target_indices, n_cpus)
        list_dict_indices = []
        for target_indices_subset in target_indices_subsets:
            list_sources_indices_subset = [list_source_indices[i] for i in target_indices_subset]
            list_dict_indices.append(dict(zip(target_indices_subset, list_sources_indices_subset)))

        # Use multiprocessing to process subsets in parallel
        with Pool(processes=n_cpus) as pool:
            subsets = [[subset, target_polygons, source_polygons] for subset in list_dict_indices]
            results = pool.map(_create_dict_info_parallel, subsets)

        # with dask.config.set(pool=Pool(n_cpus)):
        # delayed_tasks = [dask.delayed(_create_dict_info)(dict_indices=subset,
        #                                                  target_polygons=target_polygons,
        #                                                  source_polygons=source_polygons)
        #                  for subset in list_dict_indices]
        # results = dask.compute(delayed_tasks)[0]

        # Merge the results into a single dictionary
        dict_info = {i: result for subset in results for i, result in subset.items()}

    else:
        dict_indices = dict(zip(target_indices, list_source_indices))
        dict_info = _create_dict_info(
            dict_indices,
            target_polygons=target_polygons,
            source_polygons=source_polygons,
        )
    return dict_info


class PolyAggregator:
    def __init__(self, source_polygons, target_polygons, parallel=False):
        """
        Initialize the PolyAggregator.

        Parameters
        ----------
        source_polygons : list of shapely.Polygon
            List of source polygons.
        target_polygons : list of shapelyPolygon
            List of target polygons.
        parallel : bool, optional
            Whether to run in parallel. Default is False.
        use_multiprocessing : bool, optional
            Whether to use multiprocessing (if parallel is True). Default is False.
        """
        self.source_polygons = np.array(source_polygons)
        self.target_polygons = np.array(target_polygons)
        self.parallel = parallel
        self.str_tree = STRtree(self.source_polygons)
        self._dict_info = create_dict_info(
            str_tree=self.str_tree,
            target_polygons=self.target_polygons,
            source_polygons=self.source_polygons,
            parallel=self.parallel,
        )

    @property
    def n_target_polygons(self):
        return self.target_polygons.size

    @property
    def n_source_polygons(self):
        return self.source_polygons.size

    @property
    def target_intersecting_indices(self):
        return np.array(list(self._dict_info.keys()))

    @property
    def target_non_intersecting_indices(self):
        intersecting_indices = self.target_intersecting_indices
        target_indices = np.arange(0, self.n_target_polygons)
        return target_indices[np.isin(target_indices, intersecting_indices, invert=True)]

    def _normalize_weights(self, weights):
        """
        Normalize the weights.

        Parameters
        ----------
        weights : array-like
            Array of weights.

        Returns
        -------
        array
            Normalized weights.
        """
        total = np.sum(weights)
        if total == 0:
            return np.zeros_like(weights)
        return weights / total

    def _combine_weights(self, primary_weights, area_weights):
        """
        Combine primary weights with area weights.

        Parameters
        ----------
        primary_weights : array-like or None
            Array of primary weights or None.
        area_weights : array-like
            Array of area weights.

        Returns
        -------
        array
            Combined weights.
        """
        if primary_weights is None:
            return area_weights
        primary_weights = self._normalize_weights(primary_weights)
        return self._normalize_weights(primary_weights * area_weights)

    def _agg(self, func, values, weights, intersection_areas, area_weighted=True, skip_na=True):
        # Discard NaN values if asked
        if skip_na:
            is_nan = np.isnan(values)
            values = values[~is_nan]
            weights = weights[~is_nan]
            intersection_areas = intersection_areas[~is_nan]
        # Prepare weights
        if area_weighted:
            area_weights = self._normalize_weights(intersection_areas)
        else:
            area_weights = np.ones_like(intersection_areas, dtype="float")
        combined_weights = self._combine_weights(weights, area_weights)
        # Deal with no indices
        if values.size == 0:
            values = np.array([np.nan])
            combined_weights = np.array([np.nan])
        # Aggregate
        result = func(values, weights=combined_weights)
        return result

    def apply(self, func, values, weights=None, area_weighted=True, skip_na=True):
        """
        Apply a custom aggregation function to the data.

        Parameters
        ----------
        func : callable
            Aggregation function to apply.
        values : list or array-like, optional
            Array of values to aggregate. Default is None, which uses `self.data`.
        weights : list or array-like, optional
            Array of weights. Default is None.
        area_weighted : bool, optional
            Whether to weight by the intersection area. Default is True.
        skip_na : bool, optional
            Whether to discard NaN values before applying the aggregation function. Default is True.

        Returns
        -------
        list
            List of aggregated values for each target polygon.
        """
        # Check inputs
        values = np.asanyarray(values)
        expected_nvalues = len(self.source_polygons)
        if len(values) != expected_nvalues:
            raise ValueError(f"Expecting {expected_nvalues} values. Got {len(values)}.")
        if weights is None:
            weights = np.ones_like(values, dtype="float")
        weights = np.asanyarray(weights)
        if len(weights) != expected_nvalues:
            raise ValueError(f"Expecting {expected_nvalues} weights values. Got {len(weights)}.")
        # Aggregate intersecting geometries
        results = [
            self._agg(
                func,
                values=values[indices],
                weights=weights[indices],
                intersection_areas=intersection_areas,
                area_weighted=area_weighted,
                skip_na=skip_na,
            )
            for indices, intersection_areas in self._dict_info.values()
        ]
        out = np.vstack(results).squeeze()

        # Add missing values for non intersecting geometries
        if out.ndim == 1:
            arr = np.zeros(self.n_target_polygons) * np.nan
            arr = arr.astype(out.dtype)  # ensure datetime becomes NaT
            arr[self.target_intersecting_indices] = out
        elif out.ndim == 2:
            shape = (self.n_target_polygons, out.shape[1])
            arr = np.zeros(shape) * np.nan
            arr = arr.astype(out.dtype)  # ensure datetime becomes NaT
            arr[self.target_intersecting_indices, :] = out
        else:
            raise NotImplementedError
        return arr

    def fraction_covered_area(self):
        """Compute the fraction of covered area of each target polygon by the source polygons."""
        fca = [
            np.round(np.sum(intersection_areas) / self.target_polygons.take(i).area, 6)
            for i, (_, intersection_areas) in enumerate(self._dict_info.values())
        ]
        return fca

    def counts(self):
        """Compute the number of source polygons intersecting each target polygon."""
        return [len(indices) for i, (indices, _) in enumerate(self._dict_info.values())]

    def first(self, values, skip_na=True):
        def func(values, weights="dummy"):  # noqa
            return values.take(0)

        return self.apply(func=func, values=values, weights=None, area_weighted=False, skip_na=skip_na)

    def sum(self, values, weights=None, area_weighted=True, skip_na=True):
        def func(values, weights):
            return np.sum(values * weights)

        return self.apply(func=func, values=values, weights=weights, area_weighted=area_weighted, skip_na=skip_na)

    def average(self, values, weights=None, area_weighted=True, skip_na=True):
        def func(values, weights):
            return np.average(values, weights=weights)

        return self.apply(func=func, values=values, weights=weights, area_weighted=area_weighted, skip_na=skip_na)

    def variance(self, values, weights=None, area_weighted=True, skip_na=True):
        def func(values, weights):
            mean = np.average(values, weights=weights)
            return np.average((values - mean) ** 2, weights=weights)

        return self.apply(func=func, values=values, weights=weights, area_weighted=area_weighted, skip_na=skip_na)

    def max(self, values, skip_na=True):
        def func(values, weights="dummy"):  # noqa
            return np.max(values)

        return self.apply(func=func, values=values, weights=None, area_weighted=False, skip_na=skip_na)

    def min(self, values, skip_na=True):
        def func(values, weights="dummy"):  # noqa
            return np.min(values)

        return self.apply(func=func, values=values, weights=None, area_weighted=False, skip_na=skip_na)
