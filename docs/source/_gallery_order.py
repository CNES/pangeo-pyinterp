"""Order of the sections and examples of the gallery.

``conf.py`` refers to these objects by name rather than holding them, so the
Sphinx configuration stays picklable and incremental builds can reuse it.
"""

import pathlib


#: Gallery sections, in the order they are shown.
SECTIONS = ["core", "grid", "oi", "stats", "geo", "utilities", "orbit"]

#: Examples, in the order they are shown within their section.
EXAMPLES = [
    # core
    "ex_objects.py",
    "ex_axis.py",
    # grid
    "ex_xarray_interpolator.py",
    "ex_1d.py",
    "ex_2d.py",
    "ex_3d.py",
    "ex_4d.py",
    "ex_conservative_regrid.py",
    "ex_reduced_gaussian.py",
    "ex_unstructured.py",
    "ex_rtree3d_geodesic.py",
    # oi
    "ex_optimal_interpolation_2d_time.py",
    "ex_optimal_interpolation_geographic.py",
    # stats
    "ex_descriptive_statistics.py",
    "ex_binning.py",
    "ex_binning_geodetic.py",
    "ex_histogram.py",
    "ex_dask.py",
    # geo
    "ex_geodetic.py",
    "ex_geometry_primitives.py",
    "ex_geometry_spatial_predicates.py",
    "ex_geometry_operations.py",
    "ex_geometry_advanced.py",
    "ex_geometry_satellite.py",
    "ex_geohash.py",
    # utilities
    "ex_dateutils.py",
    "ex_fill_undef.py",
    # orbit
    "ex_orbit.py",
    "ex_decompose_track.py",
]


def section_key(dirname: str) -> int:
    """Return the rank of a gallery section."""
    name = pathlib.Path(dirname).name
    return SECTIONS.index(name) if name in SECTIONS else len(SECTIONS)


class ExampleKey:
    """Sort the examples of a section by their rank in :data:`EXAMPLES`."""

    def __init__(self, src_dir: str) -> None:
        """Initialize the sorter for the section stored in src_dir."""
        self.src_dir = src_dir

    def __repr__(self) -> str:
        """Return the representation sphinx-gallery records."""
        return f"{type(self).__name__}()"

    def __call__(self, fname: str) -> int:
        """Return the rank of an example; unknown ones come last."""
        name = pathlib.Path(fname).name
        return EXAMPLES.index(name) if name in EXAMPLES else len(EXAMPLES)
