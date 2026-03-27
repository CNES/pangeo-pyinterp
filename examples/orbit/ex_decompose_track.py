"""Track Decomposition
********************

This example demonstrates how to use
:py:func:`~pyinterp.geometry.satellite.decompose_track` to split a satellite
ground track into meaningful segments.  The function supports two strategies:

- ``"latitude_bands"`` — fixed latitude thresholds producing up to three zones
  (south polar, mid-latitude, north polar).  Backward-compatible with the
  original ``create_processing_blocks`` behaviour.
- ``"monotonic"`` — splits at every latitude direction change (ascending ↔
  descending), producing tight bounding boxes for near-polar orbits.

We build a realistic track by propagating several consecutive passes from a
SWOT-like ephemeris, then explore the decomposition options.
"""  # noqa: D400, D415

# %%
import pathlib

import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import pyinterp.orbit
import pyinterp.tests
from pyinterp.geometry import satellite


# %%
# Loading the Ephemeris
# =====================
# Reuse the helper from the orbit example to load the test ephemeris file.
def load_test_ephemeris(
    filename: pathlib.Path,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.timedelta64]:
    """Load the ephemeris from a text file."""
    with open(filename, encoding="utf-8") as stream:
        lines = stream.readlines()

    def to_dict(comments: list[str]) -> dict[str, float]:
        result = {}
        for item in comments:
            if not item.startswith("#"):
                raise ValueError("Comments must start with #")
            key, value = item[1:].split("=")
            result[key.strip()] = float(value)
        return result

    settings = to_dict(lines[:2])
    del lines[:2]

    ephemeris = np.loadtxt(
        lines,
        delimiter=" ",
        dtype={
            "names": ("time", "longitude", "latitude", "height"),
            "formats": ("f8", "f8", "f8", "f8"),
        },
    )
    return (
        settings["height"],
        ephemeris["longitude"],
        ephemeris["latitude"],
        ephemeris["time"].astype("timedelta64[s]"),
        np.timedelta64(int(settings["cycle_duration"] * 86400.0 * 1e9), "ns"),
    )


# %%
# Building a Realistic Multi-Pass Track
# =======================================
# Compute the orbit and concatenate several consecutive nadir tracks so the
# resulting ground track spans all latitude zones.

orbit = pyinterp.orbit.calculate_orbit(
    *load_test_ephemeris(pyinterp.tests.ephemeris_path())
)

n_passes = 6  # six half-orbit arcs ≈ three full revolutions
lon_parts, lat_parts = [], []
for pass_number in range(1, n_passes + 1):
    pass_ = pyinterp.orbit.calculate_pass(pass_number, orbit)
    if pass_ is None:
        continue
    lon_parts.append(pass_.lon_nadir)
    lat_parts.append(pass_.lat_nadir)

lon = np.concatenate(lon_parts)
lat = np.concatenate(lat_parts)

print(f"Concatenated track : {len(lon)} points over {n_passes} passes")
print(f"Longitude range    : [{lon.min():.1f}°, {lon.max():.1f}°]")
print(f"Latitude range     : [{lat.min():.1f}°, {lat.max():.1f}°]")

# %%
# Decomposing with Latitude Bands
# =================================
# The ``"latitude_bands"`` strategy divides the track at the south and north
# latitude thresholds (default −50° / +50°).  Each segment is labelled with its
# dominant :py:class:`~pyinterp.geometry.satellite.LatitudeZone` and
# :py:class:`~pyinterp.geometry.satellite.OrbitDirection`.

segments_bands = satellite.decompose_track(lon, lat, strategy="latitude_bands")

print(f"\nLatitude-bands strategy → {len(segments_bands)} segment(s)")
for seg in segments_bands:
    print(f"  {seg}")

# %%
# Decomposing with Monotonic Segments
# =====================================
# The ``"monotonic"`` strategy splits the track every time the latitude changes
# direction.  This yields tighter bounding boxes and is well suited for
# near-polar orbits.

segments_mono = satellite.decompose_track(
    lon,
    lat,
    strategy="monotonic",
    opts=satellite.DecompositionOptions()
    .with_min_edge_size(1)
    .with_merge_area_ratio(0.0),
)

print(f"\nMonotonic strategy → {len(segments_mono)} segment(s)")
for seg in segments_mono:
    print(f"  {seg}")

# %%
# Custom Options with the Builder Pattern
# =========================================
# :py:class:`~pyinterp.geometry.satellite.DecompositionOptions` is configured
# with a fluent builder.  Here we restrict the polar zones to ±70° and expand
# each segment's bounding box by a 250 km swath width.

opts = (
    satellite.DecompositionOptions()
    .with_south_limit(-70.0)
    .with_north_limit(70.0)
    .with_swath_width_km(250.0)
)

segments_custom = satellite.decompose_track(
    lon, lat, strategy="latitude_bands", opts=opts
)

print(f"\nCustom options → {len(segments_custom)} segment(s)")
for seg in segments_custom:
    print(
        f"  indices [{seg.first_index}:{seg.last_index}]  "
        f"zone={seg.zone.name:<6}  orbit={seg.orbit.name}  "
        f"size={seg.size}  bbox={seg.bbox}"
    )

# %%
# Accessing Individual Segments
# ==============================
# Each :py:class:`~pyinterp.geometry.satellite.TrackSegment` exposes the index
# range so you can slice the original arrays directly.

print("\nSlicing the track for each segment (latitude_bands):")
for seg in segments_bands:
    sl = slice(seg.first_index, seg.last_index + 1)
    lon_seg = lon[sl]
    lat_seg = lat[sl]
    print(
        f"  zone={seg.zone.name:<6}  orbit={seg.orbit.name:<11}  "
        f"points={seg.size:4d}  "
        f"lat=[{lat_seg.min():+.1f}°, {lat_seg.max():+.1f}°]"
    )

# %%
# Visualizing the Segments
# =========================
# Plot each decomposition strategy separately:
# - latitude_bands: color by latitude zone (south/mid/north)
# - monotonic: color by segment index (zone is not the relevant split axis)

zone_colors = {
    satellite.LatitudeZone.SOUTH: "#1f77b4",  # blue
    satellite.LatitudeZone.MID: "#ff7f0e",  # orange
    satellite.LatitudeZone.NORTH: "#2ca02c",  # green
}

# Latitude-bands view: colors reflect dominant latitude zone.
fig = plt.figure(figsize=(12, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines(linewidth=0.5)
ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
ax.set_title("Latitude Bands Strategy")

for seg in segments_bands:
    sl = slice(seg.first_index, seg.last_index + 1)
    ax.plot(
        lon[sl],
        lat[sl],
        ".",
        color=zone_colors[seg.zone],
        markersize=2,
        transform=ccrs.Geodetic(),
    )

legend_patches = [
    mpatches.Patch(color=zone_colors[z], label=z.name.capitalize())
    for z in satellite.LatitudeZone
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=8)

plt.tight_layout()

# Monotonic view: colors reflect segment identity, not zone.
fig = plt.figure(figsize=(12, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines(linewidth=0.5)
ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
ax.set_title("Monotonic Strategy")

segment_colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(segments_mono)))

for index, seg in enumerate(segments_mono, start=1):
    sl = slice(seg.first_index, seg.last_index + 1)
    ax.plot(
        lon[sl],
        lat[sl],
        ".",
        color=segment_colors[index - 1],
        markersize=2,
        transform=ccrs.Geodetic(),
        label=f"S{index}",
    )

ax.legend(loc="lower left", fontsize=8, ncol=2)

plt.tight_layout()

# %%
