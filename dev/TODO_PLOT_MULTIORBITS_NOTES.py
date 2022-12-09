# pcolormesh is QuadMesh
# Cells that cross the antimeridian are a part of a different PolyCollection, stored into the _wrapped_collection_fix attribute.
# So pcolormesh plots the QuadMesh and the antimeridian-crossing PolyCollection separately
# Antimeridian-crossing cell are masked, so that pcolormesh() do not plot them
# Masked collections ...

# Open Issues
# - https://github.com/SciTools/cartopy/issues/1419GeoQuadMesh
# - https://github.com/SciTools/cartopy/pull/1622 [SOLVED]

im.get_array()
im.set_array()
im.set_clim()
im.get_clim()

im._wrapped_collection_fix  # Polygons to be drawn with PolyCollection
im._wrapped_collection_fix.set_clim()

im = ax.pcolormesh(some_data)
for i in range(10):
    im.set_array(other_data)
    plt.savefig(...)

# ... support set_array() with pcolormesh
# update a map with new data using set_array() as typically done in MPL animations

# pcolor supports masked arrays for X and Y.
# pcolormesh DOES NOT SUPPORT masked arrays for X and Y.

# pcolormesh patch removing overlapping cells

# get_array() is only returning the Quadmesh cells, and yes those have masked elements where the 'wrap' occurred.

# pcolormesh change in MPL 3.3,
# please supply explicit cell edges to pcolormesh
# data = data[:-1, :-1] you will get the behavior of the old MPL style.


# Cartopy 0.20.3
