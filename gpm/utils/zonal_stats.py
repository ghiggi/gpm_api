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
"""This module contains tools for zonal statistics computations."""
import numpy as np
import shapely
from shapely.strtree import STRtree


def split_dict(dictionary, npartitions):
    """
    Splits a dictionary into n_parts of approximately equal size.

    Parameters
    ----------
    input_dict : str
        The dictionary to split..
    npartitions : int
        The number of parts to split the dictionary into..

    Returns
    -------
    list_dicts : list
        A list of dictionaries.

    """
    # Convert dictionary items to a NumPy array
    keys = np.array(list(dictionary.keys()))

    # Calculate indices to split the array into n_parts
    split_keys = np.array_split(np.arange(len(keys)), npartitions)

    # Use the indices to create subarrays
    list_dicts = [{k: dictionary[k] for k in keys} for keys in split_keys]

    return list_dicts


def create_target_dictionary(target_indices, values):
    list_values = np.split(values, np.unique(target_indices, return_index=True)[1])[1:]
    target_indices_ids = np.unique(target_indices)
    # Create dict_indices {target: values}
    dict_indices = dict(zip(target_indices_ids, list_values, strict=False))
    return dict_indices


def create_list_indices(source_polygons, target_polygons):
    # Retrieve target-source intersection indices
    # - Target/Sources Indices are not returned for empty and non-intersecting polygons !
    # - Intersects also include touching ! Intersection polygons becomes LineString !
    str_tree = STRtree(source_polygons)
    target_indices, source_indices = str_tree.query(target_polygons, predicate="intersects")
    # Check there are some geometries intersecting
    if target_indices.size == 0:
        raise ValueError("No intersecting geometries.")
    # Create list indices
    # - first row: target indices
    # - second row: source indices
    list_indices = np.vstack((target_indices, source_indices))
    return list_indices


class InverseDistanceWeights:
    """
    Class for calculating weights for Inverse Distance Weighting.

    Parameters
    ----------
    order : int, float or numpy.ndarray
        The order(s) of the inverse distance weighting.
    """

    def __init__(self, order=1):
        if np.any(order < 1) or not np.all(np.isfinite(order)):
            raise ValueError("'order' must have finite integers greater than or equal to 1.")
        self.order = np.array(order)

    def set_size(self, n):
        if self.order.ndim == 0:
            self.order = np.full(n, self.order)
        if np.size(self.order) != n:
            raise ValueError(f"Size of 'order' array must be {n}.")

    def get_weights(self, idx, distances):
        return 1 / np.power(distances, self.order[idx])


class BarnesGaussianWeights:
    """
    Class for calculating weights for Barnes Gaussian Weighting.

    Parameters
    ----------
    kappa : float or numpy.ndarray
        The smoothing parameter(s).
    """

    def __init__(self, kappa):
        if np.any(kappa <= 0) or not np.all(np.isfinite(kappa)):
            raise ValueError("'kappa' must have finite positive values.")
        self.kappa = np.array(kappa)

    def set_size(self, n):
        if self.kappa.ndim == 0:
            self.kappa = np.full(n, self.kappa)
        if len(self.kappa) != n:
            raise ValueError(f"Size of 'kappa' array must be {n}.")

    def get_weights(self, idx, distances):
        return np.exp(-np.power(distances, 2) / np.power(self.kappa[idx], 2))


class CressmanWeights:
    """
    Class for calculating weights using Cressman Weighting.

    Parameters
    ----------
    max_distance : float or array-like
        The maximum allowable distance(s). If scalar, it will be replicated.
    """

    def __init__(self, max_distance):
        if np.any(max_distance <= 0) or not np.all(np.isfinite(max_distance)):
            raise ValueError("'max_distance' must have finite positive values.")
        self.max_distance = np.array(max_distance)

    def set_size(self, n):
        if self.max_distance.ndim == 0:
            self.max_distance = np.full(n, self.max_distance)
        if len(self.max_distance) != n:
            raise ValueError(f"Size of 'max_distance' array must be {n}.")

    def get_weights(self, idx, distances):
        max_distance = self.max_distance[idx]
        max_dist_sq = max_distance**2
        dist_sq = distances**2
        weights = (max_dist_sq - dist_sq) / (max_dist_sq + dist_sq)
        weights[distances > max_distance] = 0
        return weights


DISTANCE_WEIGHTING_CLASSES = [InverseDistanceWeights, BarnesGaussianWeights, CressmanWeights]
DISTANCE_WEIGHTING_CLASSES_IMPLEMENTED = ["InverseDistanceWeights", "BarnesGaussianWeights", "CressmanWeights"]


class PolyAggregator:
    def __init__(self, source_polygons, target_polygons, parallel=False):
        """
        Initialize the PolyAggregator.

        Parameters
        ----------
        source_polygons : list of shapely.Polygon
            List of source polygons.
        target_polygons : list of shapely.Polygon
            List of target polygons.
        parallel : bool, optional
            Whether to run in parallel. Default is False.
        use_multiprocessing : bool, optional
            Whether to use multiprocessing (if parallel is True). Default is False.
        """
        self.source_polygons = np.array(source_polygons)
        self.target_polygons = np.array(target_polygons)
        self.parallel = parallel

        self.list_indices = create_list_indices(
            source_polygons=self.source_polygons,
            target_polygons=self.target_polygons,
        )

        self.dict_indices = create_target_dictionary(target_indices=self.list_indices[0], values=self.list_indices[1])

        # Fields to be optionally populated
        self._source_polygons_areas = None
        self._target_polygons_areas = None
        self._target_centroids = None
        self._source_centroids = None
        self._list_intersection_areas = None
        self._list_distances = None
        self._dict_intersection_areas = None
        self._dict_distances = None

    def _update_aggregator(self, valid_indices):
        # Update list indices
        self.list_indices = self.list_indices[:, valid_indices]
        # Update dict indices
        self.dict_indices = create_target_dictionary(target_indices=self.list_indices[0], values=self.list_indices[1])
        if self._list_intersection_areas is not None:
            self._list_intersection_areas = self._list_intersection_areas[valid_indices]
            self._dict_intersection_areas = create_target_dictionary(
                target_indices=self.list_indices[0],
                values=self._list_intersection_areas,
            )
        if self._list_distances is not None:
            self._list_distances = self._list_distances[valid_indices]
            self._dict_distances = create_target_dictionary(
                target_indices=self.list_indices[0],
                values=self._list_distances,
            )

    @property
    def target_polygons_areas(self):
        if self._target_polygons_areas is None:
            self._target_polygons_areas = shapely.area(self.target_polygons)
        return self._target_polygons_areas

    @property
    def source_polygons_areas(self):
        if self._source_polygons_areas is None:
            self._source_polygons_areas = shapely.area(self.source_polygons)
        return self._source_polygons_areas

    @property
    def target_centroids(self):
        if self._target_centroids is None:
            self._target_centroids = shapely.centroid(self.target_polygons)
        return self._target_centroids

    @property
    def source_centroids(self):
        if self._source_centroids is None:
            self._source_centroids = shapely.centroid(self.source_polygons)
        return self._source_centroids

    @property
    def list_distances(self):
        if self._list_distances is None:
            # Compute distance between polygon centroids
            distance = shapely.distance(
                self.target_centroids[self.list_indices[0]],
                self.source_centroids[self.list_indices[1]],
            )
            self._list_distances = distance
        return self._list_distances

    @property
    def list_intersection_areas(self):
        if self._list_intersection_areas is None:
            self._list_intersection_areas = self._compute_list_intersection_areas()
        return self._list_intersection_areas

    @property
    def dict_intersection_areas(self):
        if self._dict_intersection_areas is None:
            self._dict_intersection_areas = create_target_dictionary(
                target_indices=self.list_indices[0],
                values=self.list_intersection_areas,
            )
        return self._dict_intersection_areas

    @property
    def dict_distances(self):
        if self._dict_distances is None:
            self._dict_distances = create_target_dictionary(
                target_indices=self.list_indices[0],
                values=self.list_distances,
            )
        return self._dict_distances

    def _compute_list_intersection_areas(self):
        # Compute intersection polygons
        intersection_geom = shapely.intersection(
            self.target_polygons[self.list_indices[0]],
            self.source_polygons[self.list_indices[1]],
        )
        # Check valid intersection (intersecting Polygon ... not line of touching polygon)
        valid_intersection = np.array([geom.geom_type == "Polygon" for geom in intersection_geom])

        # Update aggregator lists and dictionaries
        self._update_aggregator(valid_indices=valid_intersection)

        # Compute intersection areas
        valid_intersection_geom = intersection_geom[valid_intersection]
        areas = shapely.area(valid_intersection_geom)
        return areas

    @property
    def n_target_polygons(self):
        return self.target_polygons.size

    @property
    def n_source_polygons(self):
        return self.source_polygons.size

    @property
    def target_intersecting_indices(self):
        return np.array(list(self.dict_indices.keys()))

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

    def _agg(self, func, target_idx, values, weights, area_weighting=True, distance_weighting=False, skip_na=True):

        source_shape = values.shape

        # Retrieve area_weights
        if area_weighting:
            area_weights = self.dict_intersection_areas[target_idx]
        else:
            area_weights = np.ones_like(values[:, 0], dtype="float")

        # Retrieve distance_weights
        if distance_weighting:
            distance_weights = distance_weighting.get_weights(idx=target_idx, distances=self.dict_distances[target_idx])
        else:
            distance_weights = np.ones_like(values[:, 0], dtype="float")

        # Discard NaN and Inf values if asked
        if skip_na:
            is_valid = np.all(np.isfinite(values), axis=1)
            values = values[is_valid]
            weights = weights[is_valid]
            distance_weights = distance_weights[is_valid]
            area_weights = area_weights[is_valid]

        # Normalize weights
        area_weights = self._normalize_weights(area_weights)
        distance_weights = self._normalize_weights(distance_weights)
        weights = self._normalize_weights(weights)

        # Combine weights
        # --> Potentially here provided weighting parameters for weights ...
        list_weights = [weights, area_weights, distance_weights]

        combined_weights = self._normalize_weights(np.prod(list_weights, axis=0))

        # Deal with no indices
        if values.size == 0 or np.sum(combined_weights) == 0:
            values = np.ones((1, source_shape[1]), dtype="float") * np.nan
            combined_weights = np.array([np.nan])

        # Squeeze to 1D if possible
        values = np.atleast_1d(np.squeeze(values))  # or pass axis to func !

        # Aggregate
        result = func(values, weights=combined_weights)
        return result

    def apply(self, func, values, weights=None, area_weighting=True, distance_weighting=False, skip_na=True):
        """
        Apply a custom aggregation function to the data.

        Parameters
        ----------
        func : callable
            Aggregation function to apply.
        values : list or array-like, optional
            Array of source values to aggregate.
        weights : list or array-like, optional
            Array of source weights. Default is None.
        area_weighting : bool, optional
            Whether to weight by the intersection area. Default is True.
        distance_weighting : bool or gpm.utils.zonal_stats.BaseDistanceWeights class, optional
             Whether to weight by distance between poylgon centroids. Default is False.
             Currently accepted classes are InverseDistanceWeights, BarnesGaussianWeights, CressmanWeights.
        skip_na : bool, optional
            Whether to discard NaN values before applying the aggregation function. Default is True.

        Returns
        -------
        list
            List of aggregated values for each target polygon.
        """
        # Check inputs
        values = np.asanyarray(values)
        values = np.atleast_2d(values).T
        expected_nvalues = len(self.source_polygons)
        if values.shape[0] != expected_nvalues:
            raise ValueError(f"Expecting {expected_nvalues} values. Got {values.shape[0]}.")

        # Check custom weights
        if weights is None:
            weights = np.ones_like(values[:, 0], dtype="float")
        weights = np.asanyarray(weights)
        if len(weights) != expected_nvalues:
            raise ValueError(f"Expecting {expected_nvalues} weights values. Got {len(weights)}.")

        # Check distance_weighting
        if distance_weighting:
            if not isinstance(distance_weighting, tuple(DISTANCE_WEIGHTING_CLASSES)):
                raise ValueError(
                    f"The accepted distance_weighting classes are {DISTANCE_WEIGHTING_CLASSES_IMPLEMENTED}",
                )
            distance_weighting.set_size(len(self.target_polygons))

        # Aggregate intersecting geometries
        results = [
            self._agg(
                func,
                target_idx=target_idx,
                values=values[source_indices],
                weights=weights[source_indices],
                area_weighting=area_weighting,
                distance_weighting=distance_weighting,
                skip_na=skip_na,
            )
            for target_idx, source_indices in self.dict_indices.items()
        ]
        # - atleast_1d required when only 1 target_polygon
        out = np.atleast_1d(np.vstack(results).squeeze())

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
        # Compute fraction covered area
        target_areas = shapely.area(self.target_polygons.take(list(self.dict_intersection_areas)))
        out = [
            np.round(np.sum(intersection_areas) / target_areas[i], 6)
            for i, intersection_areas in enumerate(self.dict_intersection_areas.values())
        ]
        # Infill for no intersection with target polygons
        arr = np.zeros(self.n_target_polygons) * np.nan
        arr[self.target_intersecting_indices] = out
        return arr

    def counts(self):
        """Compute the number of source polygons intersecting each target polygon."""
        out = [len(indices) for indices in self.dict_indices.values()]
        # Infill for no intersection with target polygons
        arr = np.zeros(self.n_target_polygons) * np.nan
        arr[self.target_intersecting_indices] = out
        return arr

    def first(self, values, skip_na=True):
        def func(values, weights="dummy"):  # noqa
            return values[0]

        return self.apply(func=func, values=values, weights=None, area_weighting=False, skip_na=skip_na)

    def sum(self, values, weights=None, area_weighting=True, distance_weighting=False, skip_na=True):
        def func(values, weights):
            return np.sum(values * weights)

        return self.apply(
            func=func,
            values=values,
            weights=weights,
            area_weighting=area_weighting,
            distance_weighting=distance_weighting,
            skip_na=skip_na,
        )

    def average(self, values, weights=None, area_weighting=True, distance_weighting=False, skip_na=True):
        def func(values, weights):
            return np.average(values, weights=weights)

        return self.apply(
            func=func,
            values=values,
            weights=weights,
            area_weighting=area_weighting,
            distance_weighting=distance_weighting,
            skip_na=skip_na,
        )

    def mean(self, values, weights=None, area_weighting=True, distance_weighting=False, skip_na=True):
        return self.average(
            values=values,
            weights=weights,
            area_weighting=area_weighting,
            distance_weighting=distance_weighting,
            skip_na=skip_na,
        )

    def var(self, values, weights=None, area_weighting=True, distance_weighting=False, skip_na=True):
        def func(values, weights):
            mean = np.average(values, weights=weights)
            return np.average((values - mean) ** 2, weights=weights)

        return self.apply(
            func=func,
            values=values,
            weights=weights,
            area_weighting=area_weighting,
            distance_weighting=distance_weighting,
            skip_na=skip_na,
        )

    def std(self, values, weights=None, area_weighting=True, distance_weighting=False, skip_na=True):
        return np.sqrt(
            self.var(
                values=values,
                weights=weights,
                area_weighting=area_weighting,
                distance_weighting=distance_weighting,
                skip_na=skip_na,
            ),
        )

    def max(self, values, skip_na=True):
        def func(values, weights="dummy"):  # noqa
            return np.max(values)

        return self.apply(func=func, values=values, weights=None, area_weighting=False, skip_na=skip_na)

    def min(self, values, skip_na=True):
        def func(values, weights="dummy"):  # noqa
            return np.min(values)

        return self.apply(func=func, values=values, weights=None, area_weighting=False, skip_na=skip_na)
