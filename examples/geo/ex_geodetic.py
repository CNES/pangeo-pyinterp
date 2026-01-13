""".. _example_geodetic:

Geometric Objects
=================

The library provides comprehensive utilities to manage geometric coordinates in
both geographic (lon/lat on a spheroid) and cartesian (x/y plane) coordinate
systems. The new geometry module offers a unified interface for working with
spatial data.

This example demonstrates the primary geometric objects available in
``pyinterp``:

**Geographic Geometry** (``pyinterp.geometry.geographic``):

- :py:class:`~pyinterp.geometry.geographic.Point`: Geographic point (lon, lat)
- :py:class:`~pyinterp.geometry.geographic.Segment`: Line segment between two
    points
- :py:class:`~pyinterp.geometry.geographic.Box`: Rectangular geographic area
- :py:class:`~pyinterp.geometry.geographic.LineString`: Sequence of geographic
     points
- :py:class:`~pyinterp.geometry.geographic.Ring`: Closed line forming a
     boundary
- :py:class:`~pyinterp.geometry.geographic.Polygon`: Closed geographic shape
     with holes
- :py:class:`~pyinterp.geometry.geographic.MultiPoint`: Collection of
     geographic points
- :py:class:`~pyinterp.geometry.geographic.MultiLineString`: Collection of
     geographic linestrings
- :py:class:`~pyinterp.geometry.geographic.MultiPolygon`: Collection of
     geographic polygons
- :py:class:`~pyinterp.geometry.geographic.Spheroid`: Reference ellipsoid
- :py:class:`~pyinterp.geometry.geographic.Coordinates`: Coordinate
     transformations
- :py:class:`~pyinterp.geometry.geographic.algorithms`: Geometric algorithms

**Cartesian Geometry** (``pyinterp.geometry.cartesian``):

- :py:class:`~pyinterp.geometry.cartesian.Point`: Cartesian point (x, y)
- :py:class:`~pyinterp.geometry.cartesian.Segment`: Line segment between two
    points
- :py:class:`~pyinterp.geometry.cartesian.Box`: Rectangular cartesian area
- :py:class:`~pyinterp.geometry.cartesian.LineString`: Sequence of cartesian
     points
- :py:class:`~pyinterp.geometry.cartesian.Ring`: Closed line forming a boundary
- :py:class:`~pyinterp.geometry.cartesian.Polygon`: Planar polygon
- :py:class:`~pyinterp.geometry.cartesian.MultiPoint`: Collection of cartesian
     points
- :py:class:`~pyinterp.geometry.cartesian.MultiLineString`: Collection of
     cartesian linestrings
- :py:class:`~pyinterp.geometry.cartesian.MultiPolygon`: Collection of
     cartesian polygons
- And corresponding algorithms for cartesian operations

**Satellite Utilities** (``pyinterp.geometry.satellite``):

- :py:func:`~pyinterp.geometry.satellite.find_crossovers`: Find crossover
     points between tracks

Let's start by importing the necessary libraries.
"""  # noqa: D400, D415

# %%
import json
import timeit
from typing import cast

import cartopy.crs
import cartopy.feature
import matplotlib.pyplot
import numpy

from pyinterp.geometry import cartesian, geographic, satellite


# %%
# World Geodetic System (WGS)
# ---------------------------
# The :py:class:`~pyinterp.core.geometry.geographic.Spheroid` class describes
# the reference ellipsoid used for calculations. By default, it represents the
# WGS84 system.
wgs84 = geographic.Spheroid()
print(f"WGS84 semi-major axis: {wgs84.semi_major_axis} m")
print(f"WGS84 flattening: {wgs84.flattening}")
print(f"WGS84 mean radius: {wgs84.mean_radius():.2f} m")
print(
    f"WGS84 equatorial circumference: {wgs84.equatorial_circumference():.2f} m"
)
print(f"WGS84 authalic radius: {wgs84.authalic_radius():.2f} m")

# %%
# You can also define other ellipsoids, such as GRS80, by providing the
# semi-major axis and flattening.
grs80 = geographic.Spheroid(6378137.0, 1 / 298.257222101)
print(f"\nGRS80 semi-major axis: {grs80.semi_major_axis} m")
print(f"GRS80 flattening: {grs80.flattening}")
print(
    "Difference in semi-major axis: "
    f"{abs(wgs84.semi_major_axis - grs80.semi_major_axis):.6f} m"
)

# %%
# Coordinate System Transformations
# ---------------------------------
# The :py:class:`~pyinterp.core.geometry.geographic.Coordinates` class converts
# between geodetic latitude, longitude, and altitude (LLA) and Earth-Centered,
# Earth-Fixed (ECEF) coordinates.
#
# Here, we demonstrate LLA to ECEF conversion and back.
generator = numpy.random.Generator(numpy.random.PCG64(0))
lon = generator.uniform(-180.0, 180.0, 100_000)
lat = generator.uniform(-90.0, 90.0, 100_000)
alt = generator.uniform(-10_000, 100_000, 100_000)

coords_wgs84 = geographic.Coordinates(wgs84)

# Convert to ECEF
x, y, z = coords_wgs84.lla_to_ecef(lon, lat, alt, num_threads=0)
print(f"\nConverted {len(lon)} points to ECEF")
print(f"X range: [{x.min():.2f}, {x.max():.2f}] m")
print(f"Y range: [{y.min():.2f}, {y.max():.2f}] m")
print(f"Z range: [{z.min():.2f}, {z.max():.2f}] m")

# Convert back to LLA
lon_restored, lat_restored, alt_restored = coords_wgs84.ecef_to_lla(
    x, y, z, num_threads=0
)
print(
    "Round-trip error (lon): "
    f"{numpy.abs(lon - lon_restored).max():.2e} degrees"
)
print(
    "Round-trip error (lat): "
    f"{numpy.abs(lat - lat_restored).max():.2e} degrees"
)
print(f"Round-trip error (alt): {numpy.abs(alt - alt_restored).max():.2e} m")

# %%
# Benchmark coordinate transformations between different spheroids
coords_grs80 = geographic.Coordinates(grs80)

elapsed = timeit.timeit(
    "coords_wgs84.transform(coords_grs80, lon, lat, alt, num_threads=0)",
    number=10,
    globals={
        "coords_wgs84": coords_wgs84,
        "coords_grs80": coords_grs80,
        "lon": lon,
        "lat": lat,
        "alt": alt,
    },
)
print(
    f"\nWGS84 to GRS80 transformation took: {float(elapsed) / 10:.6f} seconds"
)

# %%
# Geographic Point
# ----------------
# A :py:class:`~pyinterp.core.geometry.geographic.Point` represents a single
# location defined by its longitude and latitude in degrees.
paris = geographic.Point(2.3488, 48.8534)
new_york = geographic.Point(-73.9385, 40.6643)
london = geographic.Point(-0.1276, 51.5074)
tokyo = geographic.Point(139.6917, 35.6895)

print(f"\nParis: {paris}")
print(f"New York: {new_york}")
print(f"London: {london}")
print(f"Tokyo: {tokyo}")

# %%
# Serialization: WKT and GeoJSON
# ------------------------------
# Points (and all geometries) can be serialized to Well-Known Text (WKT) and
# GeoJSON formats.

# WKT serialization
paris_wkt = geographic.algorithms.to_wkt(paris)
print(f"\nWKT representation of Paris: {paris_wkt}")

paris_restored = geographic.algorithms.from_wkt(paris_wkt)
print(f"Is restored point equal to original? {paris == paris_restored}")

# GeoJSON serialization
paris_geojson = geographic.algorithms.to_geojson(paris)
print(f"\nGeoJSON representation of Paris: {paris_geojson}")

# Parse and pretty-print the GeoJSON
geojson_obj = json.loads(paris_geojson)
print(f"Pretty GeoJSON:\n{json.dumps(geojson_obj, indent=2)}")

# Deserialize from GeoJSON
paris_from_geojson = geographic.algorithms.from_geojson(paris_geojson)
print(
    f"Is point from GeoJSON equal to original? {paris == paris_from_geojson}"
)

# %%
# Distance Calculations and Azimuth
# ---------------------------------
# Calculate the distance between two points using different geodesic
# algorithms. The distance is returned in meters.
print("\nDistance calculations between Paris and New York:")
for strategy in [
    geographic.algorithms.ANDOYER,
    geographic.algorithms.THOMAS,
    geographic.algorithms.VINCENTY,
    geographic.algorithms.KARNEY,
]:
    distance = geographic.algorithms.distance(
        paris, new_york, spheroid=wgs84, strategy=strategy
    )
    print(f"  {strategy.name:12s}: {distance * 1e-3:.3f} km")

# %%
# Calculate azimuth (forward bearing) from one point to another
azimuth_pny = geographic.algorithms.azimuth(
    paris, new_york, spheroid=wgs84, strategy=geographic.algorithms.VINCENTY
)
print(f"\nAzimuth from Paris to New York: {numpy.degrees(azimuth_pny):.2f}°")

azimuth_nyp = geographic.algorithms.azimuth(
    new_york, paris, spheroid=wgs84, strategy=geographic.algorithms.VINCENTY
)
print(f"Azimuth from New York to Paris: {numpy.degrees(azimuth_nyp):.2f}°")

# %%
# MultiPoint: Collections of Points
# ---------------------------------
# :py:class:`~pyinterp.core.geometry.geographic.MultiPoint` represents a
# collection of points.
cities = geographic.MultiPoint(
    numpy.array(
        [paris.lon, new_york.lon, london.lon, tokyo.lon], dtype=numpy.float64
    ),
    numpy.array(
        [paris.lat, new_york.lat, london.lat, tokyo.lat], dtype=numpy.float64
    ),
)

print(f"\nMultiPoint with {len(cities)} cities")
print(f"Number of geometries: {geographic.algorithms.num_geometries(cities)}")

# Serialize to GeoJSON
cities_geojson = geographic.algorithms.to_geojson(cities)
cities_obj = json.loads(cities_geojson)
print(f"Cities GeoJSON type: {cities_obj['type']}")
print(f"Number of coordinates: {len(cities_obj['coordinates'])}")

# %%
# Geographic Box and Spatial Predicates
# --------------------------------------
# A :py:class:`~pyinterp.core.geometry.geographic.Box` defines a rectangular
# area from two corner points.
box = geographic.Box((new_york.lon, new_york.lat), (paris.lon, paris.lat))
print(
    f"\nBox min corner: ({box.min_corner.lon:.4f}, {box.min_corner.lat:.4f})"
)
print(f"Box max corner: ({box.max_corner.lon:.4f}, {box.max_corner.lat:.4f})")

# Calculate the centroid of the box
centroid = geographic.algorithms.centroid(box)
print(f"Box centroid: ({centroid.lon:.4f}, {centroid.lat:.4f})")

# %%
# Test various spatial predicates
test_point_inside = geographic.Point(0.0, 45.0)
test_point_outside = geographic.Point(100.0, 0.0)

print(
    f"\nSpatial predicates for point ({test_point_inside.lon}, "
    f"{test_point_inside.lat}):"
)
covered_by = geographic.algorithms.covered_by(test_point_inside, box)
within = geographic.algorithms.within(test_point_inside, box)
intersects = geographic.algorithms.intersects(test_point_inside, box)
print(f"  covered_by box: {covered_by}")
print(f"  within box: {within}")
print(f"  intersects box: {intersects}")

print(
    f"\nSpatial predicates for point ({test_point_outside.lon}, "
    f"{test_point_outside.lat}):"
)
covered_by = geographic.algorithms.covered_by(test_point_outside, box)
within = geographic.algorithms.within(test_point_outside, box)
intersects = geographic.algorithms.intersects(test_point_outside, box)
disjoint = geographic.algorithms.disjoint(test_point_outside, box)
print(f"  covered_by box: {covered_by}")
print(f"  within box: {within}")
print(f"  intersects box: {intersects}")
print(f"  disjoint from box: {disjoint}")
# %%
# LineString: Sequences of Connected Points
# -----------------------------------------
# A :py:class:`~pyinterp.core.geometry.geographic.LineString` represents a
# sequence of connected points forming a path.
lon_array = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=numpy.float64)
lat_array = numpy.array([0.0, 1.0, 0.5, 1.5, 1.0], dtype=numpy.float64)
line = geographic.LineString(lon_array, lat_array)
print(f"\nLineString has {len(line)} points")
print(f"Number of segments: {geographic.algorithms.num_segments(line)}")

# Calculate the length of the line
line_length = geographic.algorithms.length(line, spheroid=wgs84)
print(f"Line length: {line_length * 1e-3:.3f} km")

# %%
# Access individual points in the LineString
print("\nPoints in LineString:")
for i, point in enumerate(line):
    print(f"  Point {i}: ({point.lon:.2f}, {point.lat:.2f})")

# %%
# Calculate curvilinear distance along the line (cumulative distance from
# start)
distances = geographic.algorithms.curvilinear_distance(line, spheroid=wgs84)
print("\nCurvilinear distances along line (km):")
for i, dist in enumerate(distances):
    print(f"  Point {i}: {dist * 1e-3:.3f} km from start")

# %%
# Line interpolation: get a point at a specific distance along the line
interp_distance = line_length * 0.5  # Halfway along the line
interp_point = geographic.algorithms.line_interpolate(
    line, interp_distance, spheroid=wgs84
)
print(
    f"\nPoint at {interp_distance * 1e-3:.3f} km: "
    f"({interp_point.lon:.4f}, {interp_point.lat:.4f})"
)

# %%
# Serialize LineString to GeoJSON
line_geojson = geographic.algorithms.to_geojson(line)
line_obj = json.loads(line_geojson)
print(f"\nLineString GeoJSON type: {line_obj['type']}")
print(f"Number of coordinate pairs: {len(line_obj['coordinates'])}")

# %%
# Ring and Polygon: Closed Shapes
# -------------------------------
# A :py:class:`~pyinterp.core.geometry.geographic.Ring` is a closed boundary,
# and a :py:class:`~pyinterp.core.geometry.geographic.Polygon` is defined by an
# outer ring and optional inner rings (holes).

# Create a simple polygon
outer_ring_lon = numpy.array([0.0, 0.0, 4.0, 4.0, 0.0], dtype=numpy.float64)
outer_ring_lat = numpy.array([0.0, 7.0, 7.0, 0.0, 0.0], dtype=numpy.float64)
outer_ring = geographic.Ring(outer_ring_lon, outer_ring_lat)
polygon = geographic.Polygon(outer_ring)
num_points = geographic.algorithms.num_points(polygon)
num_interior_rings = geographic.algorithms.num_interior_rings(polygon)

print(f"\nPolygon with {num_points} points")
print(f"Number of interior rings: {num_interior_rings}")

# %%
# Validate the polygon
is_valid = geographic.algorithms.is_valid(polygon)
print(f"Is polygon valid? {is_valid}")

is_simple = geographic.algorithms.is_simple(polygon)
print(f"Is polygon simple? {is_simple}")

is_empty = geographic.algorithms.is_empty(polygon)
print(f"Is polygon empty? {is_empty}")

# %%
# Calculate geometric properties
poly_area = geographic.algorithms.area(polygon, spheroid=wgs84)
poly_perimeter = geographic.algorithms.perimeter(polygon, spheroid=wgs84)
print(f"\nPolygon area: {poly_area * 1e-6:.2f} km²")
print(f"Polygon perimeter: {poly_perimeter * 1e-3:.3f} km")

# %%
# Get the envelope (bounding box) of the polygon
envelope = geographic.algorithms.envelope(polygon)
min_corner = envelope.min_corner
max_corner = envelope.max_corner
print(
    f"Envelope: ({min_corner.lon:.1f}, {min_corner.lat:.1f}) to "
    f"({max_corner.lon:.1f}, {max_corner.lat:.1f})"
)

# %%
# Polygon with Holes
# ------------------
# Create a polygon with an interior hole

# Outer boundary
outer_lon = numpy.array([0.0, 0.0, 10.0, 10.0, 0.0], dtype=numpy.float64)
outer_lat = numpy.array([0.0, 10.0, 10.0, 0.0, 0.0], dtype=numpy.float64)
outer = geographic.Ring(outer_lon, outer_lat)

# Interior hole
hole_lon = numpy.array([3.0, 3.0, 7.0, 7.0, 3.0], dtype=numpy.float64)
hole_lat = numpy.array([3.0, 7.0, 7.0, 3.0, 3.0], dtype=numpy.float64)
hole = geographic.Ring(hole_lon, hole_lat)

# Create polygon with hole
polygon_with_hole = geographic.Polygon(outer, [hole])
num_interior_rings = geographic.algorithms.num_interior_rings(
    polygon_with_hole
)
print("\nPolygon with hole:")
print(f"  Number of interior rings: {num_interior_rings}")

area_with_hole = geographic.algorithms.area(polygon_with_hole, spheroid=wgs84)
area_outer = geographic.algorithms.area(
    geographic.Polygon(outer), spheroid=wgs84
)
area_hole = geographic.algorithms.area(
    geographic.Polygon(hole), spheroid=wgs84
)
area_difference = abs((area_outer - area_hole) - area_with_hole)
print(f"  Outer area: {area_outer * 1e-6:.2f} km²")
print(f"  Hole area: {area_hole * 1e-6:.2f} km²")
print(f"  Effective area: {area_with_hole * 1e-6:.2f} km²")
print(f"  Difference: {area_difference * 1e-6:.6f} km²")

# %%
# Serialize polygon to GeoJSON
polygon_geojson = geographic.algorithms.to_geojson(polygon_with_hole)
polygon_obj = json.loads(polygon_geojson)
print(f"\nPolygon GeoJSON type: {polygon_obj['type']}")
print(f"Number of rings: {len(polygon_obj['coordinates'])}")

# %%
# Selecting Points within a Polygon
# ---------------------------------
# Polygons are useful for selecting points that fall within a specific area.
# Here, we define a polygon for the Gulf of Mexico and check which points from
# a grid are inside it.
gulf_wkt = (
    "POLYGON((-97.5 20, -97.5 30, -82.5 30, -82.5 20, -90 17.5, -97.5 20))"
)
gulf_of_mexico = cast(
    geographic.Polygon,
    geographic.algorithms.from_wkt(gulf_wkt),
)

# %%
# Create a grid of points
lon_grid = numpy.arange(-100, -80, 2, dtype=numpy.float64)
lat_grid = numpy.arange(15, 35, 2, dtype=numpy.float64)
mx, my = numpy.meshgrid(lon_grid, lat_grid)

# %%
# Use vectorized functions to test point containment
multipoint = geographic.MultiPoint(mx.ravel(), my.ravel())

# Test multiple predicates
mask_covered = geographic.algorithms.for_each_point_covered_by(
    multipoint, gulf_of_mexico
)
mask_within = geographic.algorithms.for_each_point_within(
    multipoint, gulf_of_mexico
)

print("\nGrid statistics:")
print(f"  Total points: {len(mask_covered)}")
print(f"  Points covered by polygon: {mask_covered.sum()}")
print(f"  Points within polygon: {mask_within.sum()}")

mask = mask_covered.reshape(mx.shape)

# %%
# Calculate distances from each point to the polygon boundary
distances_to_polygon = geographic.algorithms.for_each_point_distance(
    multipoint, gulf_of_mexico, spheroid=wgs84
)
print("Distance statistics (km):")
print(f"  Min: {distances_to_polygon.min() * 1e-3:.2f}")
print(f"  Max: {distances_to_polygon.max() * 1e-3:.2f}")
print(f"  Mean: {distances_to_polygon.mean() * 1e-3:.2f}")

# %%
# Visualize the polygon and the selected points
fig = matplotlib.pyplot.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=cartopy.crs.PlateCarree())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_extent([-100, -80, 15, 35])

# Plot the polygon boundary
poly_lon = numpy.array([pt.lon for pt in gulf_of_mexico.outer])
poly_lat = numpy.array([pt.lat for pt in gulf_of_mexico.outer])
ax.plot(
    poly_lon,
    poly_lat,
    color="red",
    linewidth=2,
    transform=cartopy.crs.Geodetic(),
    label="Gulf of Mexico",
)

# Plot the points
ax.scatter(
    mx[mask],
    my[mask],
    color="green",
    s=50,
    label="Inside",
    transform=cartopy.crs.PlateCarree(),
    zorder=3,
)
ax.scatter(
    mx[~mask],
    my[~mask],
    color="gray",
    s=20,
    label="Outside",
    transform=cartopy.crs.PlateCarree(),
    alpha=0.5,
)
ax.legend()
matplotlib.pyplot.title("Point-in-Polygon Test")

# %%
# Satellite Crossover Detection
# -----------------------------
# The :py:func:`~pyinterp.core.geometry.satellite.find_crossovers` function
# finds intersection points between two satellite tracks. This requires at
# least three points per track to determine the direction of propagation.
#
# This is crucial for satellite altimetry where you need to find where two
# satellite passes intersect.

# Create two crossing satellite tracks with realistic geometry
# Track 1: ascending pass
lon1 = numpy.array(
    [234.00, 234.05, 234.10, 234.15, 234.20], dtype=numpy.float64
)
lat1 = numpy.array(
    [-67.30, -67.20, -67.10, -67.00, -66.90], dtype=numpy.float64
)

# Track 2: descending pass that crosses track 1
lon2 = numpy.array(
    [233.95, 234.05, 234.15, 234.25, 234.35], dtype=numpy.float64
)
lat2 = numpy.array(
    [-66.90, -67.00, -67.10, -67.20, -67.30], dtype=numpy.float64
)

print(f"\nTrack 1: {len(lon1)} points")
print(f"Track 2: {len(lon2)} points")

# %%
# Find crossovers between the two tracks
# The predicate parameter defines the maximum distance threshold for
# considering a potential crossover
crossovers = satellite.find_crossovers(
    lon1,
    lat1,
    lon2,
    lat2,
    predicate=50_000,  # Maximum distance in meters to consider a crossover
    spheroid=wgs84,
    strategy=geographic.algorithms.VINCENTY,
)

if crossovers:
    print(f"\nFound {len(crossovers)} crossover(s):")
    for i, crossover in enumerate(crossovers):
        point = crossover.point
        print(f"\nCrossover {i + 1}:")
        print(f"  Location: ({point.lon:.6f}°, {point.lat:.6f}°)")
        print(f"  Track 1 segment index: {crossover.index1}")
        print(f"  Track 2 segment index: {crossover.index2}")

        # Calculate the distance from the crossover point to nearby track
        # points
        p1 = geographic.Point(lon1[crossover.index1], lat1[crossover.index1])
        p2 = geographic.Point(lon2[crossover.index2], lat2[crossover.index2])
        d1 = geographic.algorithms.distance(
            crossover.point, p1, spheroid=wgs84
        )
        d2 = geographic.algorithms.distance(
            crossover.point, p2, spheroid=wgs84
        )
        print(f"  Distance to Track 1 point: {d1:.2f} m")
        print(f"  Distance to Track 2 point: {d2:.2f} m")
else:
    print("\nNo crossovers found.")

# %%
# Visualize the two tracks and their intersection point
fig = matplotlib.pyplot.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection=cartopy.crs.PlateCarree())
ax.add_feature(cartopy.feature.LAND, alpha=0.3)
ax.add_feature(cartopy.feature.OCEAN, alpha=0.3)

# Plot the tracks with direction arrows
ax.plot(
    lon1,
    lat1,
    "-o",
    color="red",
    linewidth=2,
    markersize=8,
    label="Track 1 (ascending)",
    transform=cartopy.crs.Geodetic(),
)
ax.plot(
    lon2,
    lat2,
    "-s",
    color="blue",
    linewidth=2,
    markersize=8,
    label="Track 2 (descending)",
    transform=cartopy.crs.Geodetic(),
)

# Add arrows to show direction
for i in range(len(lon1) - 1):
    ax.annotate(
        "",
        xy=(lon1[i + 1], lat1[i + 1]),
        xytext=(lon1[i], lat1[i]),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
        transform=cartopy.crs.PlateCarree(),
    )
for i in range(len(lon2) - 1):
    ax.annotate(
        "",
        xy=(lon2[i + 1], lat2[i + 1]),
        xytext=(lon2[i], lat2[i]),
        arrowprops={"arrowstyle": "->", "color": "blue", "lw": 2},
        transform=cartopy.crs.PlateCarree(),
    )

# Plot the crossover points
if crossovers:
    for i, crossover in enumerate(crossovers):
        ax.plot(
            crossover.point.lon,
            crossover.point.lat,
            "o",
            color="green",
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=2,
            label=f"Crossover {i + 1}" if i == 0 else "",
            transform=cartopy.crs.PlateCarree(),
            zorder=10,
        )
        ax.text(
            crossover.point.lon + 0.02,
            crossover.point.lat + 0.02,
            f"X{i + 1}",
            fontsize=12,
            fontweight="bold",
            transform=cartopy.crs.PlateCarree(),
        )

ax.set_extent(
    [
        min(lon1.min(), lon2.min()) - 0.05,
        max(lon1.max(), lon2.max()) + 0.05,
        min(lat1.min(), lat2.min()) - 0.05,
        max(lat1.max(), lat2.max()) + 0.05,
    ]
)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.legend(loc="upper right")
matplotlib.pyplot.title("Satellite Track Crossover Detection")

# %%
# Handling the International Date Line
# ------------------------------------
# Geographic boxes that span the International Date Line (180°/-180° longitude)
# require special handling.

dateline_box = geographic.Box((170.0, -30.0), (-170.0, 30.0))
min_corner = dateline_box.min_corner
max_corner = dateline_box.max_corner
print("\nDateline-crossing box:")
print(f"  Min corner: ({min_corner.lon}, {min_corner.lat})")
print(f"  Max corner: ({max_corner.lon}, {max_corner.lat})")

# %%
# Test points around the dateline
test_points = [
    (170, 0, "Left edge (170°E)"),
    (175, 0, "Eastern section (175°E)"),
    (180, 10, "On dateline (180°)"),
    (-180, -10, "On dateline (-180°)"),
    (-175, 0, "Western section (-175°W)"),
    (-170, 0, "Right edge (-170°W)"),
    (0, 0, "Prime meridian (gap)"),
    (160, 0, "West of box (160°E)"),
    (-160, 0, "East of box (-160°W)"),
]

print("\nPoint containment test:")
print(f"{'Longitude':<12} {'Latitude':<10} {'Within?':<10} {'Description'}")
print("-" * 70)

for lon, lat, description in test_points:
    point = geographic.Point(lon, lat)
    is_inside = geographic.algorithms.within(point, dateline_box)
    status = "Yes" if is_inside else "No"
    print(f"{lon:>9.1f}   {lat:>9.1f} {status:^10}  {description}")

# %%
# Visualize the dateline-crossing box
fig = matplotlib.pyplot.figure(figsize=(12, 8))
ax = fig.add_subplot(
    111, projection=cartopy.crs.PlateCarree(central_longitude=180)
)
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_extent([140, -140, -40, 40])

# Plot the box boundaries
box_lon = [170, 190, 190, 170, 170]  # 190 = -170 + 360
box_lat = [-30, -30, 30, 30, -30]
ax.plot(
    box_lon,
    box_lat,
    color="red",
    linewidth=3,
    transform=cartopy.crs.Geodetic(),
    label="Dateline-Crossing Box",
)

# Plot test points
for lon, lat, _ in test_points:
    point = geographic.Point(lon, lat)
    is_inside = geographic.algorithms.within(point, dateline_box)
    color = "green" if is_inside else "gray"
    marker = "o" if is_inside else "x"
    ax.plot(
        lon,
        lat,
        marker,
        color=color,
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=1,
        transform=cartopy.crs.PlateCarree(),
    )

ax.legend()
matplotlib.pyplot.title("Dateline-Crossing Box")

# %%
# Cartesian Geometry
# ------------------
# The library also provides cartesian geometry for planar operations.

# Create cartesian points
p1 = cartesian.Point(0.0, 0.0)
p2 = cartesian.Point(3.0, 4.0)
print("\nCartesian points:")
print(f"  P1: ({p1.x}, {p1.y})")
print(f"  P2: ({p2.x}, {p2.y})")

# Calculate Euclidean distance
cart_distance = cartesian.algorithms.distance(p1, p2)
print(f"  Euclidean distance: {cart_distance:.3f} units")

# %%
# Create a cartesian polygon
cart_outer_x = numpy.array([0.0, 10.0, 10.0, 0.0, 0.0], dtype=numpy.float64)
cart_outer_y = numpy.array([0.0, 0.0, 10.0, 10.0, 0.0], dtype=numpy.float64)
cart_ring = cartesian.Ring(cart_outer_x, cart_outer_y)
cart_polygon = cartesian.Polygon(cart_ring)

# Calculate cartesian properties
cart_area = cartesian.algorithms.area(cart_polygon)
cart_perimeter = cartesian.algorithms.perimeter(cart_polygon)
print("\nCartesian polygon:")
print(f"  Area: {cart_area:.2f} square units")
print(f"  Perimeter: {cart_perimeter:.2f} units")

# Test point containment
test_cart_point = cartesian.Point(5.0, 5.0)
is_inside_cart = cartesian.algorithms.covered_by(test_cart_point, cart_polygon)
print(
    f"  Is ({test_cart_point.x}, {test_cart_point.y}) inside? {is_inside_cart}"
)

# %%
# Converting Between Geographic and Cartesian
# -------------------------------------------
# You can convert geographic geometries to cartesian for certain operations.

geo_point = geographic.Point(10.0, 20.0)
cart_converted = geographic.algorithms.convert_to_cartesian(geo_point)
print("\nCoordinate conversion:")
print(f"  Geographic: ({geo_point.lon}°, {geo_point.lat}°)")
print(f"  Cartesian: ({cart_converted.x}, {cart_converted.y})")

# Convert a polygon
geo_ring_lon = numpy.array([0.0, 5.0, 5.0, 0.0, 0.0], dtype=numpy.float64)
geo_ring_lat = numpy.array([0.0, 0.0, 5.0, 5.0, 0.0], dtype=numpy.float64)
geo_ring = geographic.Ring(geo_ring_lon, geo_ring_lat)
geo_poly = geographic.Polygon(geo_ring)

geo_area = geographic.algorithms.area(geo_poly, wgs84)
cart_area = cartesian.algorithms.area(
    geographic.algorithms.convert_to_cartesian(geo_poly)
)

cart_poly_converted = geographic.algorithms.convert_to_cartesian(geo_poly)
print("\nPolygon conversion:")
print(f"  Geographic area: {geo_area * 1e-6:.2f} km²")
print(f"  Cartesian area: {cart_area:.2f} deg²")

# %%
# Advanced Geometric Operations
# -----------------------------

# Simplification: reduce the number of points in a geometry
dense_lon = numpy.linspace(0.0, 10.0, 100, dtype=numpy.float64)
dense_lat = numpy.sin(dense_lon * 0.5) + 45.0
dense_line = geographic.LineString(dense_lon, dense_lat)

simplified_line = geographic.algorithms.simplify(
    dense_line,
    max_distance=10000.0,
    spheroid=wgs84,  # 10 km tolerance
)

print("\nSimplification:")
print(f"  Original: {len(dense_line)} points")
print(f"  Simplified: {len(simplified_line)} points")
print(
    f"  Reduction: {(1 - len(simplified_line) / len(dense_line)) * 100:.1f}%"
)

# %%
# Densification: add intermediate points to a geometry
sparse_lon = numpy.array([0.0, 10.0], dtype=numpy.float64)
sparse_lat = numpy.array([0.0, 10.0], dtype=numpy.float64)
sparse_line = geographic.LineString(sparse_lon, sparse_lat)

densified_line = geographic.algorithms.densify(
    sparse_line,
    max_distance=100000.0,
    spheroid=wgs84,  # 100 km segments
)
densified_length = geographic.algorithms.length(densified_line, wgs84)
print("\nDensification:")
print(f"  Original: {len(sparse_line)} points")
print(f"  Densified: {len(densified_line)} points")
print(f"  Total length: {densified_length * 1e-3:.2f} km")

# %%
# Convex Hull: compute the smallest convex polygon containing all points
points_for_hull = geographic.MultiPoint(
    numpy.array([0.0, 1.0, 2.0, 1.0, 0.5], dtype=numpy.float64),
    numpy.array([0.0, 1.0, 0.0, -1.0, 0.5], dtype=numpy.float64),
)

hull = geographic.algorithms.convex_hull(points_for_hull, spheroid=wgs84)
print("\nConvex hull:")
print(f"  Input: {len(points_for_hull)} points")
print(f"  Hull: {geographic.algorithms.num_points(hull)} points")
print(f"  Hull area: {geographic.algorithms.area(hull, wgs84) * 1e-6:.4f} km²")

# %%
# Set Operations: union, intersection, difference
poly1_lon = numpy.array([0.0, 0.0, 5.0, 5.0, 0.0], dtype=numpy.float64)
poly1_lat = numpy.array([0.0, 5.0, 5.0, 0.0, 0.0], dtype=numpy.float64)
poly1 = geographic.Polygon(geographic.Ring(poly1_lon, poly1_lat))

poly2_lon = numpy.array([3.0, 3.0, 8.0, 8.0, 3.0], dtype=numpy.float64)
poly2_lat = numpy.array([0.0, 5.0, 5.0, 0.0, 0.0], dtype=numpy.float64)
poly2 = geographic.Polygon(geographic.Ring(poly2_lon, poly2_lat))

poly1_area = geographic.algorithms.area(poly1, wgs84)
poly2_area = geographic.algorithms.area(poly2, wgs84)

print("\nSet operations on polygons:")
print(f"  Polygon 1 area: {poly1_area * 1e-6:.4f} km²")
print(f"  Polygon 2 area: {poly2_area * 1e-6:.4f} km²")

# Test if polygons intersect
do_intersect = geographic.algorithms.intersects(poly1, poly2)
print(f"  Do they intersect? {do_intersect}")

if do_intersect:
    # Compute intersection
    intersection_list = geographic.algorithms.intersection(
        poly1, poly2, spheroid=wgs84
    )
    if intersection_list:
        intersection = intersection_list[0]
        area = geographic.algorithms.area(intersection, wgs84)
        print(f"  Intersection area: {area * 1e-6:.4f} km²")

    # Compute union
    union_list = geographic.algorithms.union(poly1, poly2, spheroid=wgs84)
    if union_list:
        union_area = sum(
            geographic.algorithms.area(p, wgs84) for p in union_list
        )
        print(f"  Union area: {union_area * 1e-6:.4f} km²")

    # Compute difference
    diff_list = geographic.algorithms.difference(poly1, poly2, spheroid=wgs84)
    if diff_list:
        diff_area = sum(
            geographic.algorithms.area(p, wgs84) for p in diff_list
        )
        print(f"  Difference (1-2) area: {diff_area * 1e-6:.4f} km²")

# %%
# Closest Points: find the nearest points between two geometries
line1 = geographic.LineString(
    numpy.array([0.0, 2.0], dtype=numpy.float64),
    numpy.array([0.0, 0.0], dtype=numpy.float64),
)
line2 = geographic.LineString(
    numpy.array([5.0, 7.0], dtype=numpy.float64),
    numpy.array([1.0, 1.0], dtype=numpy.float64),
)

closest = geographic.algorithms.closest_points(line1, line2, spheroid=wgs84)
distance = geographic.algorithms.distance(closest.a, closest.b, spheroid=wgs84)
print("\nClosest points between two lines:")
print(f"  Point on line 1: ({closest.a.lon:.4f}, {closest.a.lat:.4f})")
print(f"  Point on line 2: ({closest.b.lon:.4f}, {closest.b.lat:.4f})")
print(f"  Distance: {distance * 1e-3:.3f} km")

# %%
# Geometric Validation and Correction
# -----------------------------------

# Create a self-intersecting (invalid) polygon
invalid_lon = numpy.array([0.0, 2.0, 2.0, 0.0, 0.0], dtype=numpy.float64)
invalid_lat = numpy.array([0.0, 0.0, 2.0, 2.0, 0.0], dtype=numpy.float64)
# Make it self-intersecting by swapping two points
invalid_lon[2], invalid_lon[3] = invalid_lon[3], invalid_lon[2]
invalid_lat[2], invalid_lat[3] = invalid_lat[3], invalid_lat[2]

invalid_ring = geographic.Ring(invalid_lon, invalid_lat)
invalid_poly = geographic.Polygon(invalid_ring)

is_valid, reason = geographic.algorithms.is_valid(
    invalid_poly, return_reason=True
)
print("\nGeometry validation:")
print(f"  Is valid? {is_valid}")
if not is_valid:
    print(f"  Reason: {reason}")

# Try to correct the geometry
geographic.algorithms.correct(invalid_poly)
is_valid_after = geographic.algorithms.is_valid(invalid_poly)
print(f"  Is valid after correction? {is_valid_after}")
