from functools import wraps
import numpy as np
import pandas as pd 
import polars as pl
from gpm.utils.geospatial import (
    _check_size, 
    check_extent, 
    Extent,
    get_geographic_extent_around_point,
    get_country_extent,
    get_continent_extent,
    get_extent_around_point,
)
####--------------------------------------------------------.
#### Grids
## Unstructured
# - Planar 
# - Spherical
# --> Regular Geometry 
# --> Irregular Geometry 

## Structured
# 2D on the sphere 
# - SwathDefinition
# - CRS: geographic

# 2D on planar projection
# - AreaDefinition: define cell size (resolution) based on shape and extent !
# - CRS: geographic / projected

####--------------------------------------------------------.
#### Partitioning
# 2D Partitioning:
# --> XYPartitioning: size 
# --> GeographicPartitioning
# --> TilePartitioning
# - Last partition not granted to have same size as the other
# - If size = resolution, share same property as an AreaDefinition 
# - Directory: 1 level (TilePartitioning) or 2 levels (XYPartitioning/GeographicPartitioning)
# - CRS: geographic / projected
# - Query by extent: OK 
# - Query by geometry: First subset by extent, then intersect quadmesh?

# KdTree/RTree Partitioning
# --> https://rtree.readthedocs.io/en/latest/

# SphericalPartitioning
# --> H3Partitioning (Uber’s Hexagonal Hierarchical) 
# --> S2Partitioning (Google Geometry) (Hierarchical, discrete, global grid system) (Square cells)
# --> HealpixPartitioning
# --> GaussianGrid, CubedSphere, ... (other hierarchical, discrete, and global grid system)
# --> Hexbin, the process of taking coordinates and binning them into hexagonal cells in analytics or mapping software.
# --> Geohash: a system for encoding locations using a string of characters, creating a hierarchical, square grid system (a quadtree).

# - Directory: 1 level
# - Query by extent: Define polygon and query by geometry ???  
# - Query by geometry: First subset by extent, then intersect quadmesh?

# GeometryPartitioning
# - Can include all partitioning 

# GeometryBucket --> GeometryIntersection, save geometries in GeoParquet


#### CentroidBucket 

#### GeometryBucket 
# - Exploit kdtree and dask 
# - Borrow/Replace from pyresample bucket 
# - Output to xvec/datatree?
# - Intermediate for remapping (weighted fraction, fraction_overlap, conservative)

####--------------------------------------------------------.
#### Bucket Improvements
# Readers 
# - gpm.bucket.read_around_point(bucket_dir, lon, lat, distance, size, **polars_kwargs) 
# -->  compute distance on subset and select below threshold 
# -->  https://stackoverflow.com/questions/76262681/i-need-to-create-a-column-with-the-distance-between-two-coordinates-in-polars
# - gpm.bucket.read_within_extent(bucket_dir, extent, **polars_kwargs) 
# - gpm.bucket.read_within_country(bucket_dir, country, **polars_kwargs) 
# - gpm.bucket.read_within_continent(bucket_dir, continent, **polars_kwargs) 

# Core:
# - list_partitions_within_extent
# - list_filepaths_within_extent
# - read_within_extent !

# Routines
# - Routine to repartition in smaller partitions (disaggregate bucket) 
# - Routine to repartition in larger partitions (aggregate bucket) 
 
# Analysis  
# - Group by overpass 
# - Reformat to dataset / Generator 

# Writers 
# - RENAME: write_partitioned_dataset into write_arrow_dataset 

####--------------------------------------------------------.
#### Query directories 
# Format 
# - hive: xbin=xx/ybin=yy or tile_id=id
# - xx/yy or id 
## Option1 (faster if small query and lot of directories)
# - Create directories paths 
# - Check if they exists 
## Option2 (might be faster for large queries)
## - List directories
## - Filter by lon/lat
# Core:
# - list_partitions_within_extent
# - list_filepaths_within_extent
# - read_within_extent !

# Notes
# - https://h3geo.org/docs/comparisons/s2
# - H3: https://github.com/uber/h3-py  # https://pypi.org/project/h3/
#  --> h3.latlng_to_cell(lat, lng, resolution)
 
####---------------------------------------------------------------------------------------------
# ProjectionPartitioning
# Enable distance in meters and compute planar or on sphere  
# def get_partitions_around_point(self, x, y, distance=None, size=None):
#     extent = get_extent_around_point(x, y, distance=distance, size=size)
#     return self.get_partitions_by_extent(extent=extent)

# Requires pyproj CRS. Backproject x,y to lon/lat
# def get_partitions_by_country(self, name, padding=None):
#     extent = get_country_extent(name, padding=padding) 
#     return self.get_partitions_by_extent(extent=extent)

# Requires pyproj CRS. Backproject x,y to lon/lat
# def get_partitions_by_continent(self, name, padding=None):
#     extent = get_continent_extent(name, padding=padding) 
#     return self.get_partitions_by_extent(extent=extent)

####---------------------------------------------------------------------------------------------


def check_valid_dataframe(func):
   """
   Decorator to check if the first argument or the keyword argument 'df'
   is a `pandas.DataFrame` or a `polars.DataFrame`.
   """
   @wraps(func)
   def wrapper(*args, **kwargs):
       # Check if 'df' is in kwargs, otherwise assume it is the first positional argument
       df = kwargs.get('df', args[1] if len(args) == 2 else None)
       # Validate the DataFrame
       if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
           raise TypeError("The 'df' argument must be either a pandas.DataFrame or a polars.DataFrame.")
       return func(*args, **kwargs)
   return wrapper


def check_valid_x_y(df, x, y):
    """Check if the x and y columns are in the dataframe."""
    if y not in df: 
        raise ValueError(f"y='{y}' is not a column of the dataframe.")
    if x not in df: 
        raise ValueError(f"x='{x}' is not a column of the dataframe.")


def ensure_xy_without_nan_values(df, x, y, remove_invalid_rows=True):
    """Ensure valid coordinates in the dataframe."""
    # Remove NaN vales 
    if remove_invalid_rows: 
        if isinstance(df, pd.DataFrame):
            return df.dropna(subset=[x,y])
        else: 
            return df.filter(~pl.col(x).is_null() | ~pl.col(y).is_null())
        
    # Check no NaN values
    if isinstance(df, pd.DataFrame): 
        indices = df[[x, y]].isna().any(axis=1)
    else:    
        indices = (df[x].is_null() | df[x].is_null())
    if indices.any(): 
         rows_indices = np.where(indices)[0].tolist()
         raise ValueError(f"Null values present in columns {x} and {y} at rows: {rows_indices}")
    return df 


def ensure_valid_partitions(df, xbin, ybin, remove_invalid_rows=True):
    """Ensure valid partitions labels in the dataframe."""
    # Remove NaN values 
    if remove_invalid_rows: 
        if isinstance(df, pd.DataFrame):
            return df.dropna(subset=[xbin,ybin])
        else: 
            df = df.filter(~pl.col(xbin).is_in(["outside_right", "outside_left"]))
            df = df.filter(~pl.col(ybin).is_in(["outside_right", "outside_left"]))
            df = df.filter(~pl.col(xbin).is_null() | ~pl.col(ybin).is_null())
            return df
        
   # Check no invalid partitions (NaN or polars outside_right/outside_left)      
    if isinstance(df, pd.DataFrame): 
        indices = df[[xbin, ybin]].isna().any(axis=1)
    else:
        indices = (df[xbin].is_in(["outside_right", "outside_left"]) | 
                   df[ybin].is_in(["outside_right", "outside_left"])
        )
    if indices.any(): 
         rows_indices = np.where(indices)[0].tolist()
         raise ValueError(f"Out of extent x,y coordinates at rows: {rows_indices}") 
         

def get_array_combinations(x,y):
    """Return all the combinations between the two array."""
    # Create the mesh grid
    grid1, grid2 = np.meshgrid(x, y)
    # Stack and reshape the grid arrays to get combinations
    combinations = np.vstack([grid1.ravel(), grid2.ravel()]).T
    return combinations 


def get_n_decimals(number):
    """Get the number of decimals of a number."""
    number_str = str(number)
    decimal_index = number_str.find(".")

    if decimal_index == -1:
        return 0  # No decimal point found

    # Count the number of characters after the decimal point
    return len(number_str) - decimal_index - 1


def get_breaks(size, vmin, vmax):
    """Define partitions edges."""
    breaks = np.arange(vmin, vmax, size)
    if breaks[-1] != vmax:
        breaks = np.append(breaks, np.array([vmax]))
    return breaks


def get_midpoints(size, vmin, vmax): 
    """Define partitions midpoints."""
    breaks = get_breaks(size, vmin=vmin, vmax=vmax)
    midpoints = breaks[0:-1] + size / 2
    return midpoints


def get_labels(size, vmin, vmax):
    """Define partitions labels (rounding partitions midpoints)."""
    n_decimals = get_n_decimals(size)
    midpoints = get_midpoints(size, vmin, vmax)
    return midpoints.round(n_decimals + 1).astype(str)


def get_breaks_and_midpoints(size, vmin, vmax):
    """Return the partitions edges and partitions midpoints."""
    breaks = get_breaks(size, vmin=vmin, vmax=vmax)
    midpoints = get_midpoints(size, vmin=vmin, vmax=vmax)
    return breaks, midpoints


def get_breaks_and_labels(size, vmin, vmax):
    """Return the partitions edges and partitions labels."""
    breaks = get_breaks(size, vmin=vmin, vmax=vmax)
    labels = get_labels(size, vmin=vmin, vmax=vmax)
    return breaks, labels


def query_labels(values, breaks, labels): 
    """Return the partition labels for the specified coordinates."""
    # TODO: flag to raise error for NaN, None?
    values = np.atleast_1d(np.asanyarray(values)).astype(float)
    return pd.cut(values, bins=breaks, labels=labels, include_lowest=True, right=True)
         

def query_midpoints(values, breaks, midpoints): 
    """Return the partition midpoints for the specified coordinates."""
    values = np.atleast_1d(np.asanyarray(values)).astype(float)
    return pd.cut(values, bins=breaks, labels=midpoints, include_lowest=True, right=True).astype(float)


def add_pandas_xy_partitions(df, size, extent, x, y, xbin, ybin, remove_invalid_rows=True):
    """Add partitions labels to a pandas DataFrame based on x, y coordinates."""
    # Check x,y names
    check_valid_x_y(df, x, y)
    # Check/remove rows with NaN x,y columns  
    df = ensure_xy_without_nan_values(df, x=x, y=y, remove_invalid_rows=remove_invalid_rows)
    # Retrieve breaks and labels (N and N+1) 
    cut_x_breaks, cut_x_labels = get_breaks_and_labels(size[0], vmin=extent[0], vmax=extent[1])
    cut_y_breaks, cut_y_labels = get_breaks_and_labels(size[1], vmin=extent[2], vmax=extent[3])
    # Add partitions labels columns
    df[xbin] = query_labels(df[x].to_numpy(), breaks=cut_x_breaks, labels=cut_x_labels)
    df[ybin] = query_labels(df[y].to_numpy(), breaks=cut_y_breaks, labels=cut_y_labels)
    # Check/remove rows with invalid partitions (NaN)  
    df = ensure_valid_partitions(df, xbin=xbin, ybin=ybin, remove_invalid_rows=remove_invalid_rows)
    return df


def add_polars_xy_partitions(df, x, y, size, extent, xbin, ybin, remove_invalid_rows=True):
    """Add partitions to a polars DataFrame based on x, y coordinates."""
    # Check x,y names
    check_valid_x_y(df, x, y)
    # Check/remove rows with null x,y columns  
    df = ensure_xy_without_nan_values(df, x=x, y=y, remove_invalid_rows=remove_invalid_rows)
    # Retrieve breaks and labels (N and N+1)
    cut_x_breaks, cut_x_labels = get_breaks_and_labels(size[0], vmin=extent[0], vmax=extent[1])
    cut_y_breaks, cut_y_labels = get_breaks_and_labels(size[1], vmin=extent[2], vmax=extent[3])
    # Add outside labels for polars cut function 
    cut_x_labels = ["outside_left", *cut_x_labels, "outside_right"]
    cut_y_labels = ["outside_left", *cut_y_labels, "outside_right"]
    # Deal with left inclusion
    cut_x_breaks[0] = cut_x_breaks[0] - 1e-8
    cut_y_breaks[0] = cut_y_breaks[0] - 1e-8
    # Add partitions columns
    df = df.with_columns(
        pl.col(x).cut(cut_x_breaks, labels=cut_x_labels, left_closed=False).alias(xbin),
        pl.col(y).cut(cut_y_breaks, labels=cut_y_labels, left_closed=False).alias(ybin),
    )        
    # Check/remove rows with invalid partitions (out of extent or Null)  
    df = ensure_valid_partitions(df, xbin=xbin, ybin=ybin, remove_invalid_rows=remove_invalid_rows)
    return df 



def add_polars_xy_tile_partitions(df, size, extent, x,y, tile_id):
    check_valid_x_y(df, x, y)
    raise NotImplementedError()


def add_pandas_xy_tile_partitions(df, size, extent, x,y, tile_id):
    check_valid_x_y(df, x, y)
    raise NotImplementedError()  
    

def df_to_xarray(df, xbin, ybin, size, extent, new_x=None, new_y=None): 
    """Convert dataframe to xarray Dataset based on specified partitions midpoints.
    
    The partitioning cells not present in the dataframe are set to NaN.    
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    if set(df.index.names) == set([xbin, ybin]):
        df = df.reset_index()
        
    # Ensure index is float or string 
    df[xbin] = df[xbin].astype(float)
    df[ybin] = df[ybin].astype(float)
    df = df.set_index([xbin, ybin])
 
    # Create an empty DataFrame with the MultiIndex
    x_midpoints = get_midpoints(size[0], vmin=extent[0], vmax=extent[1])
    y_midpoints = get_midpoints(size[1], vmin=extent[2], vmax=extent[3])
    multi_index = pd.MultiIndex.from_product(
        [x_midpoints, y_midpoints],
        names=[xbin, ybin],
    )
    empty_df = pd.DataFrame(index=multi_index)

    # Create final dataframe
    df_full = empty_df.join(df, how="left")
        
    # Reshape to xarray
    ds = df_full.to_xarray()
    ds[xbin] = ds[xbin].astype(float)
        
    # Rename dictionary
    rename_dict = {}
    if new_x is not None: 
        rename_dict[xbin] = new_x 
    if new_y is not None: 
        rename_dict[ybin] = new_y
    ds = ds.rename(rename_dict)
    return ds


class XYPartitioning:
    """
    Handales partitioning of data into x and y bins.
       
    Parameters:
    ----------
    xbin : float
        Identifier for the x bin.
    ybin : float
        Identifier for the y bin.
    size : int, float, tuple, list
        The size value(s) of the bins. 
        The function interprets the input as follows:
        - int or float: The same size is enforced in both x and y directions.
        - tuple or list: The bin size for the x and y directions.
    extent : list
        The extent for the partitioning specified as [xmin, xmax, ymin, ymax].

    """
    def __init__(self, xbin, ybin, size, extent):
        # Define extent 
        self.extent = check_extent(extent)
        # Define bin size
        self.size = _check_size(size)
        # Define bin names
        self.xbin = xbin  
        self.ybin = ybin
        # Define breaks, midpoints and labels 
        self.x_breaks = get_breaks(size=self.size[0], vmin=self.extent.xmin, vmax=self.extent.xmax)
        self.y_breaks = get_breaks(size=self.size[1], vmin=self.extent.ymin, vmax=self.extent.ymax)
        self.x_midpoints = get_midpoints(size=self.size[0], vmin=self.extent.xmin, vmax=self.extent.xmax)
        self.y_midpoints = get_midpoints(size=self.size[1], vmin=self.extent.ymin, vmax=self.extent.ymax)
        self.x_labels = get_labels(size=self.size[0], vmin=self.extent.xmin, vmax=self.extent.xmax)
        self.y_labels = get_labels(size=self.size[1], vmin=self.extent.ymin, vmax=self.extent.ymax)
        # Define info
        self.shape = (len(self.x_labels), len(self.y_labels))
        self.n_partitions = len(self.x_labels) * len(self.y_labels)
        self.n_x = self.shape[0] 
        self.n_y = self.shape[1] 
        
    @property
    def partitions(self):
        return [self.xbin, self.ybin]
    
    @check_valid_dataframe
    def add_partitions(self, df, x, y, remove_invalid_rows=True): 
        if isinstance(df, pd.DataFrame): 
            return add_pandas_xy_partitions(df=df, x=x, y=y,
                                            size=self.size, 
                                            extent=self.extent,
                                            xbin=self.xbin, 
                                            ybin=self.ybin,
                                            remove_invalid_rows=remove_invalid_rows)
        return add_polars_xy_partitions(df=df, x=x, y=y,
                                        size=self.size, 
                                        extent=self.extent,
                                        xbin=self.xbin, 
                                        ybin=self.ybin,
                                        remove_invalid_rows=remove_invalid_rows)
    
    @check_valid_dataframe
    def to_xarray(self, df, new_x=None, new_y=None):
        return df_to_xarray(df,
                            xbin=self.xbin,
                            ybin=self.ybin, 
                            size=self.size, extent=self.extent, 
                            new_x=new_x, 
                            new_y=new_y)
    
    def to_dict(self): 
        dictionary = {"name": self.__class__.__name__,
                      "extent": list(self.extent), 
                      "size": self.size, 
                      "xbin": self.xbin, 
                      "ybin": self.ybin}
        return dictionary
        
    def query_x_labels(self, x):
        """Return the x partition labels for the specified x coordinates."""
        return query_labels(x, breaks=self.x_breaks, labels=self.x_labels).astype(str)
    
    def query_y_labels(self, y):
        """Return the y partition labels for the specified y coordinates."""
        return query_labels(y, breaks=self.y_breaks, labels=self.y_labels).astype(str)
    
    def query_labels(self, x, y): 
        """Return the partition labels for the specified x,y coordinates."""
        return self.query_x_labels(x), self.query_y_labels(y)
    
    def query_x_midpoints(self, x):
        """Return the x partition midpoints for the specified x coordinates."""
        return query_midpoints(x, breaks=self.x_breaks, midpoints=self.x_midpoints)
    
    def query_y_midpoints(self, y):
        """Return the y partition midpoints for the specified y coordinates."""
        return query_midpoints(y, breaks=self.y_breaks, midpoints=self.y_midpoints)
    
    def query_midpoints(self, x, y): 
        """Return the partition midpoints for the specified x,y coordinates."""
        return self.query_x_midpoints(x), self.query_y_midpoints(y)

    def get_partitions_by_extent(self, extent):
        """Return the partitions labels containing data within the extent."""
        extent = check_extent(extent)
        # Define valid query extent (to be aligned with partitioning extent)
        query_extent = [max(extent.xmin, self.extent.xmin), min(extent.xmax, self.extent.xmax),
                        max(extent.ymin, self.extent.ymin), min(extent.ymax, self.extent.ymax)]
        query_extent = Extent(*query_extent)
        # Retrieve midpoints
        xmin, xmax = self.query_x_midpoints([query_extent.xmin, query_extent.xmax])
        ymin, ymax = self.query_y_midpoints([query_extent.ymin, query_extent.ymax])
        # Retrieve univariate x and y labels within the extent
        x_labels = self.x_labels[np.logical_and(self.x_midpoints >= xmin, self.x_midpoints <= xmax)]
        y_labels = self.y_labels[np.logical_and(self.y_midpoints >= ymin, self.y_midpoints <= ymax)]
        # Retrieve combination of all (x,y) labels within the extent
        combinations = get_array_combinations(x_labels, y_labels)
        return combinations

    def get_partitions_around_point(self, x, y, distance=None, size=None):
        extent = get_extent_around_point(x, y, distance=distance, size=size)
        return self.get_partitions_by_extent(extent=extent)

    @property
    def quadmesh(self):
        """Return the quadrilateral mesh.
        
        A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are defined via a (M+1, N+1) 
        grid of vertices.
        
        The quadrilateral mesh is accepted by `matplotlib.pyplot.pcolormesh`, `matplotlib.collections.QuadMesh` 
        `matplotlib.collections.PolyQuadMesh`.
        
        np.naddary
            Quadmesh array of shape (M+1, N+1, 2)
        """
        x_corners, y_corners = np.meshgrid(self.x_breaks, self.y_breaks)
        return np.stack((x_corners, y_corners), axis=2)
    
    # to_yaml  
    # to_shapely
    # to_spherically (geographic)
    # to_geopandas [lat_bin, lon_bin, geometry]
    


class GeographicPartitioning(XYPartitioning):
    """
    Handles geographic partitioning of data based on longitude and latitude bin sizes within a defined extent.
    
    The last bin size (in lon and lat direction) might not be of size ``size` !
    
    Parameters:
    ----------
    size : float
        The uniform size for longitude and latitude binning.
    xbin : str, optional
        Name of the longitude bin, default is 'lon_bin'.
    ybin : str, optional
        Name of the latitude bin, default is 'lat_bin'.
    extent : list, optional
        The geographical extent for the partitioning specified as [xmin, xmax, ymin, ymax].
        Default is the whole earth: [-180, 180, -90, 90].

    Inherits:
    ----------
    XYPartitioning
    """
    def __init__(self, size, xbin="lon_bin", ybin="lat_bin", extent=[-180, 180, -90, 90]):
        super().__init__(xbin=xbin, ybin=ybin, size=size, extent=extent)    
    
    def get_partitions_around_point(self, lon, lat, distance=None, size=None):
        extent = get_geographic_extent_around_point(lon=lon, lat=lat, 
                                                    distance=distance,
                                                    size=size, 
                                                    distance_type="geographic")
        return self.get_partitions_by_extent(extent=extent)
    
    def get_partitions_by_country(self, name, padding=None):
        extent = get_country_extent(name, padding=padding) 
        return self.get_partitions_by_extent(extent=extent)
    
    def get_partitions_by_continent(self, name, padding=None):
        extent = get_continent_extent(name, padding=padding) 
        return self.get_partitions_by_extent(extent=extent)
    
    @check_valid_dataframe
    def to_xarray(self, df, new_x="lon", new_y="lat"):
        return df_to_xarray(df,
                            xbin=self.xbin,
                            ybin=self.ybin, 
                            size=self.size, 
                            extent=self.extent, 
                            new_x=new_x, 
                            new_y=new_y)



class TilePartitioning:
    """
    Handles partitioning of data into tiles within a specified extent.

    Parameters:
    ----------
    size : float
        The size of the tiles.
    extent : list
        The extent for the partitioning specified as [xmin, xmax, ymin, ymax].
    tile_id : str, optional
        Identifier for the tile bin. The default is ``'tile_id'``.

    """
    def __init__(self, size, extent, tile_id="tile_id"):
        self.size = _check_size(size)
        self.extent = check_extent(extent)
        self.tile_id = tile_id
    
    @property
    def bins(self):
        return [self.tile_id]
    
    @check_valid_dataframe
    def add_partitions(self, df, x, y): 
        if isinstance(df, pd.DataFrame): 
            return add_pandas_xy_tile_partitions(df, x=x, y=x,
                                                 size=self.size, 
                                                 extent=self.extent,
                                                 tile_id=self.tile_id, 
                                                 )
        return add_polars_xy_tile_partitions(df, x=x, y=x,
                                             size=self.size, 
                                             extent=self.extent,
                                             tile_id=self.tile_id, 
                                            )