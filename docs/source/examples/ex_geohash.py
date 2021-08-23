"""
*******
Geohash
*******

Geohashing is a geocoding method used to encode geographic coordinates (latitude
and longitude) into a short string of digits and letters delineating an area on
a map, which is called a cell, with varying resolutions. The more characters in
the string, the more precise the location.

Geohash Grid
============
"""
import timeit
import cartopy.crs
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot
import numpy
import pandas
#
import pyinterp.geohash
import pyinterp.geodetic


#%%
# Writing a visualization routine for GeoHash grids.
def _sort_colors(colors):
    """Sort colors by hue, saturation, value and name in descending order"""
    by_hsv = sorted(
        (tuple(matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color))),
         name) for name, color in colors.items())
    return [name for hsv, name in reversed(by_hsv)]


def _plot_box(ax, code, color, caption=True):
    """Plot a GeoHash bounding box"""
    box = pyinterp.geohash.bounding_box(code.decode())
    x0 = box.min_corner.lon
    x1 = box.max_corner.lon
    y0 = box.min_corner.lat
    y1 = box.max_corner.lat
    dx = x1 - x0
    dy = y1 - y0
    box = matplotlib.patches.Rectangle((x0, y0),
                                       dx,
                                       dy,
                                       alpha=0.5,
                                       color=color,
                                       ec="black",
                                       lw=1,
                                       transform=cartopy.crs.PlateCarree())
    ax.add_artist(box)
    if not caption:
        return
    rx, ry = box.get_xy()
    cx = rx + box.get_width() * 0.5
    cy = ry + box.get_height() * 0.5
    ax.annotate(code.decode(), (cx, cy),
                color='w',
                weight='bold',
                fontsize=16,
                ha='center',
                va='center')


def plot_geohash_grid(precision,
                      points=None,
                      box=None,
                      color_list=None,
                      inc=7):
    """Plot geohash bounding boxes"""
    color_list = color_list or matplotlib.colors.CSS4_COLORS
    fig = matplotlib.pyplot.figure(figsize=(24, 12))
    ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.PlateCarree())
    if box is not None:
        ax.set_extent([
            box.min_corner.lon, box.max_corner.lon, box.min_corner.lat,
            box.max_corner.lat
        ],
                      crs=cartopy.crs.PlateCarree())
    if points is not None:
        xmin, xmax = points['lon'].min(), points['lon'].max()
        ymin, ymax = points['lat'].min(), points['lat'].max()
        if box is None:
            ax.set_xlim((xmin - 4, xmax + 4))
            ax.set_ylim((ymin - 4, ymax + 4))

        ax.plot(points['lon'],
                points['lat'],
                color='black',
                marker=".",
                ls='',
                ms='0.5',
                transform=cartopy.crs.PlateCarree())
    colors = _sort_colors(color_list)
    ic = 0
    codes = pyinterp.geohash.bounding_boxes(box, precision=precision)

    color_codes = {codes[0][0]: colors[ic]}
    for item in codes:
        prefix = item[precision - 1]
        if prefix not in color_codes:
            ic += inc
            color_codes[prefix] = colors[ic % len(colors)]
        _plot_box(ax, item, color_codes[prefix], True)
    ax.stock_img()
    ax.coastlines()
    ax.grid()


#%%
# Bounds of geohash with a precision of 1 character.
plot_geohash_grid(1)

#%%
# Bounds of the geohash ``d`` with a precision of two characters.
plot_geohash_grid(2, box=pyinterp.geohash.bounding_box('d'))

#%%
# Bounds of the geohash ``dd`` with a precision of three characters.
plot_geohash_grid(3, box=pyinterp.geohash.bounding_box('dd'))

#%%
# Bounds of the geohash ``dds`` with a precision of four characters.
plot_geohash_grid(4, box=pyinterp.geohash.bounding_box('dds'))

#%%
# Encoding
# ========
#
# Generation of dummy data
SIZE = 1000000
lon = numpy.random.uniform(-180, 180, SIZE)
lat = numpy.random.uniform(-80, 80, SIZE)
measures = numpy.random.random_sample(SIZE)

#%%
# This algorithm is very fast, which makes it possible to process a lot of data quickly.
timeit.timeit("pyinterp.geohash.encode(lon, lat)",
              number=50,
              globals=dict(pyinterp=pyinterp, lon=lon, lat=lat)) / 50

#%%
# Geohash index
# =============
store = pyinterp.geohash.storage.MutableMapping()
index = pyinterp.geohash.index.init_geohash(store, precision=3)

#%%time
# The index can contain anything, as long as it's possible to serialize the data.
index.update(zip(index.encode(lon, lat), measures))
#%%
# Number of box filled in this index
len(index)

#%%
# Let's imagine that we want to retrieve the data in the following polygon:
#
# * ``POLYGON((-33.75 39.375,-33.75 45,-22.5 45,-22.5 39.375,-33.75 39.375))``
polygon = pyinterp.geodetic.Polygon.read_wkt(
    "POLYGON((-33.75 39.375,-33.75 45,-22.5 45,-22.5 39.375,-33.75 39.375))")
items = index.items(index.keys(polygon.envelope()))

#%%
# Density calculation
df = pandas.DataFrame(
    dict(lon=lon,
         lat=lat,
         measures=measures,
         geohash=pyinterp.geohash.encode(lon, lat, precision=3)))
df.set_index("geohash", inplace=True)
df = df.groupby("geohash").count()["measures"].rename("count").to_frame()
df["density"] = df["count"] / (
    pyinterp.geohash.area(df.index.values.astype('S')) / 1e6)
array = pyinterp.geohash.to_xarray(df.index.values.astype('S'), df.density)
array = array.where(array != 0, numpy.nan)

fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(111)
_ = array.plot(ax=ax)