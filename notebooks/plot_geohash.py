import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

import pyinterp.geohash as geohash


def _sort_colors(colors):
    """Sort colors by hue, saturation, value and name in descending order"""
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
                    for name, color in colors.items())
    return [name for hsv, name in reversed(by_hsv)]


def _plot_box(ax, code, color, caption=True):
    """Plot a GeoHash bounding box"""
    box = geohash.string.bounding_box(code)
    x0 = box.min_corner.lon
    x1 = box.max_corner.lon
    y0 = box.min_corner.lat
    y1 = box.max_corner.lat
    dx = x1 - x0
    dy = y1 - y0
    box = mpatch.Rectangle((x0, y0),
                           dx,
                           dy,
                           alpha=0.5,
                           color=color,
                           ec="black",
                           lw=1,
                           transform=ccrs.PlateCarree())
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


def grid(precision, points=None, box=None, color_list=None, inc=7):
    """Plot geohash bounding boxes"""
    color_list = color_list or mcolors.CSS4_COLORS
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if box is not None:
        ax.set_extent([
            box.min_corner.lon, box.max_corner.lon, box.min_corner.lat,
            box.max_corner.lat
        ],
                      crs=ccrs.PlateCarree())
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
                transform=ccrs.PlateCarree())
    colors = _sort_colors(color_list)
    ic = 0
    codes = geohash.string.bounding_boxes(box, precision=precision)
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