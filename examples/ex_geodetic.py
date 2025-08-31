"""
.. _example_geodetic:

Geodetic Objects
================

The library provides utilities to manage geodetic coordinates. While other
libraries offer more exhaustive geodetic functionalities, ``pyinterp`` includes
these objects because its C++ core requires geodetic information to be passed
from Python.

This example demonstrates how to use the primary geodetic objects available in
``pyinterp``:

- :py:class:`pyinterp.geodetic.Spheroid`: Defines the reference ellipsoid.
- :py:class:`pyinterp.geodetic.Coordinates`: Handles coordinate system
  transformations.
- :py:class:`pyinterp.geodetic.Point`: Represents a single geographic point.
- :py:class:`pyinterp.geodetic.Box`: Represents a rectangular geographic area.
- :py:class:`pyinterp.geodetic.LineString`: Represents a sequence of points.
- :py:class:`pyinterp.geodetic.Polygon`: Represents a closed geographic shape.
- :py:class:`pyinterp.geodetic.Crossover`: Calculates the intersection of two
  lines.

Let's start by importing the necessary libraries.
"""
# %%
import timeit

import cartopy.crs
import cartopy.feature
import matplotlib.pyplot
import numpy

import pyinterp.geodetic

# %%
# World Geodetic System (WGS)
# ---------------------------
# The :py:class:`pyinterp.geodetic.Spheroid` class describes the reference
# ellipsoid used for calculations. By default, it represents the WGS84 system.
wgs84 = pyinterp.geodetic.Spheroid()
print(wgs84)

# %%
# You can also define other ellipsoids, such as GRS80, by providing the
# semi-major axis and the inverse of flattening.
grs80 = pyinterp.geodetic.Spheroid((6378137, 1 / 298.257222101))
print(grs80)

# %%
# Coordinate System Transformations
# ---------------------------------
# The :py:class:`pyinterp.geodetic.Coordinates` class is used internally to
# convert between geodetic latitude, longitude, and altitude (LLA) and
# Earth-Centered, Earth-Fixed (ECEF) coordinates.
#
# Here, we measure the performance of transforming a large number of points from
# the WGS84 to the GRS80 coordinate system.
generator = numpy.random.Generator(numpy.random.PCG64(0))
lon = generator.uniform(-180.0, 180.0, 1_000_000)
lat = generator.uniform(-90.0, 90.0, 1_000_000)
alt = generator.uniform(-10_000, 100_000, 1_000_000)

# %%
# Create coordinate system handlers for WGS84 and GRS80
a = pyinterp.geodetic.Coordinates(wgs84)
b = pyinterp.geodetic.Coordinates(grs80)

# %%
# Time the transformation
elapsed = timeit.timeit('a.transform(b, lon, lat, alt, num_threads=0)',
                        number=10,
                        globals={
                            'a': a,
                            'b': b,
                            'lon': lon,
                            'lat': lat,
                            'alt': alt
                        })
print(f'Transformation took: {float(elapsed) / 10:.6f} seconds')

# %%
# Geodetic Point
# --------------
# A :py:class:`pyinterp.geodetic.Point` represents a single location defined by
# its longitude and latitude in degrees.
paris = pyinterp.geodetic.Point(2.3488, 48.8534)
new_york = pyinterp.geodetic.Point(-73.9385, 40.6643)

# %%
# Points can be serialized to and from the Well-Known Text (WKT) format.
print(f'WKT representation of Paris: {paris.wkt()}')
print('Is the WKT representation of Paris equal to the original point? '
      f'{pyinterp.geodetic.Point.read_wkt(paris.wkt()) == paris}')

# %%
# Distance Calculations
# ---------------------
# You can calculate the distance between two points using different geodesic
# algorithms: `Andoyer`, `Thomas`, or `Vincenty`. The distance is returned in
# meters.
for strategy in ['andoyer', 'thomas', 'vincenty']:
    distance = paris.distance(new_york, strategy=strategy, wgs=wgs84)
    print(f'Distance between Paris and New York ({strategy}): '
          f'{distance * 1e-3:.3f} km')

# %%
# The library also provides a vectorized function,
# :py:func:`pyinterp.geodetic.coordinate_distances`, for calculating distances
# over large arrays of coordinates efficiently.
lon1 = numpy.arange(0, 10, 1, dtype=numpy.float64)
lat1 = numpy.arange(0, 10, 1, dtype=numpy.float64)
lon2 = lon1 + 1.0
lat2 = lat1 + 1.0

distances = pyinterp.geodetic.coordinate_distances(lon1,
                                                   lat1,
                                                   lon2,
                                                   lat2,
                                                   strategy='vincenty',
                                                   wgs=wgs84,
                                                   num_threads=1)
print('Vectorized distance calculations:')
for i in range(len(distances)):
    print(f'Distance between ({lon1[i]:.1f}, {lat1[i]:.1f}) and '
          f'({lon2[i]:.1f}, {lat2[i]:.1f}): {distances[i]:.3f} m')

# %%
# Geodetic Box and Polygon
# ------------------------
# A :py:class:`pyinterp.geodetic.Box` defines a rectangular area from two corner
# points.
box = pyinterp.geodetic.Box(new_york, paris)
print(f'Box WKT: {box.wkt()}')

# %%
# A :py:class:`pyinterp.geodetic.Polygon` is a more general shape defined by a
# series of points. A box can be converted to a polygon.
polygon = pyinterp.geodetic.Polygon.read_wkt(box.wkt())
print(f'Polygon WKT: {polygon.wkt()}')

# %%
# You can perform various geometric operations, such as calculating the area in
# square meters or getting the envelope (bounding box).
print(f'Area of the polygon: {polygon.area(wgs=wgs84) * 1e-6:.2f} km²')

simple_polygon = pyinterp.geodetic.Polygon.read_wkt(
    'POLYGON((0 0, 0 7, 4 2, 2 0, 0 0))')
print(f'Envelope of a simple polygon: {simple_polygon.envelope()}')

# %%
# Selecting Points within a Polygon
# ---------------------------------
# Polygons are useful for selecting points that fall within a specific area.
# Here, we define a polygon for the Gulf of Mexico and check which points from a
# grid are inside it.
gulf_of_mexico = pyinterp.geodetic.Polygon.read_wkt(
    'POLYGON ((-97.5 20, -97.5 30, -82.5 30, -82.5 20, -90 17.5, -97.5 20))')

# %%
# Create a grid of points
lon = numpy.arange(-100, -80, 2, dtype=numpy.float64)
lat = numpy.arange(15, 35, 2, dtype=numpy.float64)
mx, my = numpy.meshgrid(lon, lat)

# %%
# Use the `covered_by` method to get a mask of points inside the polygon.
mask = gulf_of_mexico.covered_by(mx.ravel(), my.ravel())
mask = mask.reshape(mx.shape)

# %%
# Now, let's visualize the polygon and the selected points.
fig = matplotlib.pyplot.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=cartopy.crs.PlateCarree())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_extent([-100, -80, 15, 35])

# Plot the polygon boundary
poly_lon, poly_lat = zip(*((pt.lon, pt.lat) for pt in gulf_of_mexico.outer))
poly_lon = numpy.array(poly_lon)
poly_lat = numpy.array(poly_lat)

ax.plot(poly_lon, poly_lat, color='red', transform=cartopy.crs.Geodetic())

# Plot the points, coloring them based on whether they are inside the polygon
ax.scatter(mx[mask],
           my[mask],
           color='green',
           label='Inside',
           transform=cartopy.crs.PlateCarree())
ax.scatter(mx[~mask],
           my[~mask],
           color='gray',
           label='Outside',
           transform=cartopy.crs.PlateCarree())
ax.legend()

# %%
# Crossover Detection
# ===================
# The :py:class:`pyinterp.geodetic.Crossover` class is used to find the
# intersection point between two line segments, which is particularly useful for
# finding crossovers between satellite tracks.
#
# We'll define two simple line segments (half-orbits).
lon1 = numpy.array([234.068, 234.142], dtype=numpy.float64)
lat1 = numpy.array([-67.117, -67.163], dtype=numpy.float64)
lon2 = numpy.array([234.061, 234.135], dtype=numpy.float64)
lat2 = numpy.array([-67.183, -67.138], dtype=numpy.float64)

# %%
# Create the Crossover object from two LineString objects.
crossover = pyinterp.geodetic.Crossover(
    pyinterp.geodetic.LineString(lon1, lat1),
    pyinterp.geodetic.LineString(lon2, lat2))

# %%
# Check if an intersection exists.
intersection_point = None
if crossover.exists():
    print('A crossover exists between the two lines.')
    # Search for the crossover point
    intersection_point = crossover.search()
    if intersection_point:
        print(f'The intersection point is: {intersection_point}')

        # Find the indices of the nearest points on each line to the
        # intersection
        nearest_indices = crossover.nearest(intersection_point)
        if nearest_indices:
            print('Nearest point on line #1: '
                  f'{crossover.half_orbit_1[nearest_indices[0]]}')
            print('Nearest point on line #2: '
                  f'{crossover.half_orbit_2[nearest_indices[1]]}')
else:
    print('No crossover found.')

# %%
# Finally, we visualize the two lines and their intersection point.
fig = matplotlib.pyplot.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=cartopy.crs.PlateCarree())
ax.add_feature(cartopy.feature.LAND)

# Plot the lines
ax.plot(lon1,
        lat1,
        '-o',
        color='red',
        label='Line 1',
        transform=cartopy.crs.Geodetic())
ax.plot(lon2,
        lat2,
        '-o',
        color='blue',
        label='Line 2',
        transform=cartopy.crs.Geodetic())
ax.set_extent([
    min(lon1.min(), lon2.min()) - 0.01,
    max(lon1.max(), lon2.max()) + 0.01,
    min(lat1.min(), lat2.min()) - 0.01,
    max(lat1.max(), lat2.max()) + 0.01
])

# Plot the intersection point
if intersection_point:
    ax.plot(intersection_point.lon,
            intersection_point.lat,
            'o',
            color='green',
            markersize=10,
            label='Intersection')

ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.legend()
