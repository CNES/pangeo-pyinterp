Examples
--------

2D interpolation
================

.. _bivariate:

Bivariate
#########

Perform a bivariate interpolation of gridded data points.

The distribution contains a 2D field ``mss.nc`` that will be used in this help.
This file is located in the ``tests/dataset`` directory at the root of the
project.

.. warning ::

    This file is an old version of the sub-sampled quarter step MSS CNES/CLS. Do
    not use it for scientific purposes, download the latest updated
    high-resolution version instead `here <https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/mss.html>`_.

The first step is to load the data into memory:

.. code:: python

    import netCDF4
    import pyinterp.bivariate

    ds = netCDF4.Dataset("tests/dataset/mss.nc")

Afterwards, build the :py:class:`axes <pyinterp.core.Axis>` associated with the
grid:

.. code:: python

    import pyinterp.core

    x_axis = pyinterp.core.Axis(ds.variables["lon"][:], is_circle=True)
    y_axis = pyinterp.core.Axis(ds.variables["lat"][:])

Finally, we can build the object defining the :py:class:`grid
<pyinterp.grid.Grid2D>` to interpolate:

.. code:: python

    # The shape of the bivariate values must be (len(x_axis), len(y_axis))
    mss = ds.variables["mss"][:].T
    # The undefined values must be set to nan.
    mss[mss.mask] = float("nan")
    grid = pyinterp.grid.Grid2D(x_axis, y_axis, mss.data)

We will then build the coordinates on which we want to interpolate our grid:

.. code:: python

    import numpy as np

    # The coordinates used for interpolation are shifted to avoid using the
    # points of the bivariate function.
    mx, my = np.meshgrid(np.arange(-180, 180, 1) + 1 / 3.0,
                         np.arange(-89, 89, 1) + 1 / 3.0,
                         indexing='ij')

The grid is :py:meth:`interpolated <pyinterp.bivariate.bivariate>` to
the desired coordinates:

.. code:: python

    mss = pyinterp.bivariate.bivariate(
        grid, mx.flatten(), my.flatten()).reshape(mx.shape)

Values can be interpolated with several methods: *bilinear*, *nearest*, and
*inverse distance weighting*. Distance calculations, if necessary, are
calculated using the `Haversine formula
<https://en.wikipedia.org/wiki/Haversine_formula>`_.

An experimental module of the library simplifies the use of the library by
using xarray and CF information contained in dataset. This module
implements all the other interpolators of the regular grids presented below.

.. code:: python

    import pyinterp.backends.xarray
    import xarray as xr

    ds = xr.open_dataset("tests/dataset/mss.nc")
    interpolator = pyinterp.backends.xarray.Grid2D(ds, "mss")
    mss = interpolator.bivariate(dict(lon=mx.flatten(), lat=my.flatten()))

Bicubic
#######

Interpolating data points on two-dimensional regular grid. The interpolated
surface is smoother than the corresponding surfaces obtained by bilinear
interpolation. Bicubic interpolation is achieved by spline functions provided
by `GSL <https://www.gnu.org/software/gsl/>`_.

The interpolation :py:meth:`pyinterp.bicubic.bicubic` function has more
parameters in order to define the data frame used by the spline functions and
how to process the edges of the regional grids:

.. code:: python

    import pyinterp.bicubic

    mss = pyinterp.bicubic.bicubic(
        grid, mx.flatten(), my.flatten(), nx=3, ny=3).reshape(mx.shape)


It is also possible to simplify the interpolation of the dataset by using
xarray:

.. code:: python

    mss = interpolator.bicubic(dict(lon=mx.flatten(), lat=my.flatten()))

3D interpolation
================

Trivariate
##########

The :py:class:`trivariate <pyinterp.trivariate.Trivariate>` interpolation
allows to obtain values at arbitrary points in a 3D space of a function defined
on a grid.

The distribution contains a 3D field ``tcw.nc`` that will be used in this help.
This file is located in the ``tests/dataset`` directory at the root of the
project.


This method performs a bilinear interpolation in 2D space by considering the
axes of longitude and latitude of the grid, then performs a linear
interpolation in the third dimension. Its interface is similar to the
:py:class:`Bivariate <pyinterp.bivariate.Bivariate>` class except for a third
axis which is handled by this object.

.. code:: python

    import pyinterp.trivariate

    ds = netCDF4.Dataset("tests/dataset/tcw.nc")
    x_axis = pyinterp.core.Axis(ds.variables["longitude"][:], is_circle=True)
    y_axis = pyinterp.core.Axis(ds.variables["latitude"][:])
    z_axis = pyinterp.core.Axis(ds.variables["time"][:])
    # The shape of the bivariate values must be
    # (len(x_axis), len(y_axis), len(z_axis))
    tcw = ds.variables['tcw'][:].T
    # The undefined values must be set to nan.
    tcw[tcw.mask] = float("nan")
    grid = pyinterp.grid.Grid3D(
        x_axis, y_axis, z_axis, tcw.data)
    # The coordinates used for interpolation are shifted to avoid using the
    # points of the bivariate function.
    mx, my, mz = np.meshgrid(np.arange(-180, 180, 1) + 1 / 3.0,
                            np.arange(-89, 89, 1) + 1 / 3.0,
                            898500 + 3,
                            indexing='ij')
    tcw = pyinterp.trivariate.trivariate(
        grid, mx.flatten(), my.flatten(), mz.flatten()).reshape(mx.shape)

It is also possible to simplify the interpolation of the dataset by using
xarray:

.. code:: python

    ds = xr.open_dataset("tests/dataset/tcw.nc")
    interpolator = pyinterp.backends.xarray.Grid3D(ds, "tcw")
    tcw = interpolator.trivariate(
        dict(longitude=mx.flatten(), latitude=my.flatten(), time=mz.flatten()))

Unstructured grid
=================

The interpolation of this object is based on an :py:class:`R*Tree
<pyinterp.rtree.RTree>` structure. To begin with, we start by building this
object. By default, this object considers WGS-84 geodetic coordinate system.
But you can define another one using class :py:class:`System
<pyinterp.geodetic.System>`.

.. code:: python

    import pyinterp.rtree

    mesh = pyinterp.rtree.RTree()

Then, we will insert points into the tree. The class allows you to insert
points using two algorithms. The first one called :py:meth:`packing
<pyinterp.rtree.RTree.packing>` allows you to insert the values in the tree at
once. This mechanism is the recommended solution to create an optimized
in-memory structure, both in terms of construction time and queries. When this
is not possible, you can insert new information into the tree as you go along
using the :py:meth:`insert <pyinterp.rtree.RTree.insert>` method.

.. code:: python

    ds = netCDF4.Dataset("tests/dataset/mss.nc")
    # The shape of the bivariate values must be (len(longitude), len(latitude))
    mss = ds.variables['mss'][:].T
    mss[mss.mask] = float("nan")
    # Be careful not to enter undefined values in the tree.
    x_axis, y_axis = np.meshgrid(
        ds.variables['lon'][:], ds.variables['lat'][:], indexing='ij')
    mesh.packing(
        np.vstack((x_axis.flatten(), y_axis.flatten())).T,
        mss.data.flatten())

When the tree is created, you can :py:meth:`interpolate
<pyinterp.rtree.RTree.inverse_distance_weighting>` the data or make various
:py:meth:`queries <pyinterp.rtree.RTree.query>` on the tree.

.. code:: python

    mx, my = np.meshgrid(
        np.arange(-180, 180, 1) + 1 / 3.0,
        np.arange(-90, 90, 1) + 1 / 3.0,
        indexing="ij")
    mss, neighbors = mesh.inverse_distance_weighting(
        np.vstack((mx.flatten(), my.flatten())).T,
        within=False,
        radius=35434,
        k=8,
        num_threads=0)

Fill NaN values
===============

The undefined values in the grids do not allow interpolation of values located
in the neighborhood. This behavior is a concern when you need to interpolate
values near the land/sea mask of some maps. The library provides two functions
to fill the undefined values.

LOESS
#####

The :py:func:`first <pyinterp.fill.loess>` method applies a weighted local
regression to extrapolate the boundary between defined and undefined values. The
user must indicate the number of pixels on the X and Y axes to be considered in
the calculation. For example:

.. code:: python

    # Module that handles the filling of undefined values.
    import pyinterp.fill

    ds = netCDF4.Dataset("tests/dataset/mss.nc")
    x_axis = pyinterp.core.Axis(ds.variables["lon"][:], is_circle=True)
    y_axis = pyinterp.core.Axis(ds.variables["lat"][:])
    mss = ds.variables["mss"][:].T
    mss[mss.mask] = float("nan")
    grid = pyinterp.grid.Grid2D(x_axis, y_axis, mss.data)
    filled = pyinterp.fill.loess(grid, nx=3, ny=3, num_threads=4)

The image below illustrates the result:

.. image:: pictures/loess.png

Gauss-Seidel
############

The :py:func:`second <pyinterp.fill.gauss_seidel>` method consists of replacing
all undefined values (Nan) in a grid using the Gauss-Seidel method by
relaxation. This `link
<https://math.berkeley.edu/~wilken/228A.F07/chr_lecture.pdf>`_ contains more
information on the method used.

.. code:: python

    has_converged, filled = pyinterp.fill.gauss_seidel(grid)

The image below illustrates the result:

.. image:: pictures/gauss_seidel.png

Interpolation of a time series
==============================

This example shows how to interpolate a time series using the library.

In this example, we consider the time series of MSLA maps distributed by
AVISO/CMEMS. This series consists of a grid stored in netCDF format for each
weekly date distributed. The weekly file nomenclature is as follows:
``dt_global_allsat_phy_l4_YYYYYYMMDD_YYYYYMMDD.nc``. The first date defines the
date of the map snapshot and the second date defines the production date. To
manage this series of grids, we create the following object:

.. code:: python

    import datetime
    import re
    import os
    import dask.distributed
    import numpy as np
    import netCDF4
    import pandas as pd
    import pyinterp.core
    import pyinterp.trivariate


    class GridSeries:
        """Handling of MSLA AVISO maps.

        Args:
            dirname (str): Map storage directory.
        """

        def __init__(self, dirname):
            self.dirname = dirname
            self.df, self.dt = self._load_ts()

        def _load_ts(self):
            """Loading the time series into memory."""
            pattern = re.compile(
                r"dt_global_allsat_phy_l4_(\d{4})(\d{2})(\d{2})_\d{8}\.nc").search
            times = []
            files = []
            for root, dirs, items in os.walk(self.dirname):
                for item in items:
                    match = pattern(item)
                    if match is None:
                        continue
                    times.append(
                        datetime.datetime(int(match.group(1)), int(match.group(2)),
                                        int(match.group(3))))
                    files.append(os.path.join(root, item))
            times = np.array(times)
            files = np.array(files)
            indices = np.argsort(times)

            df = pd.DataFrame(data=dict(path=files[indices]), index=times[indices])
            frequency = set(pd.Series(np.diff(df.index)).dt.total_seconds())
            if len(frequency) != 1:
                raise RuntimeError(
                    "Time series does not have a constant step between two "
                    f"grids: {frequency}")
            return df, datetime.timedelta(seconds=frequency.pop())

        def load_dataset(self, varname, start, end):
            """Loading the time series into memory for the defined period.
            
            Args:
                varname (str): Name of the variable to be loaded into memory.
                start (datetime.datetime): Date of the first map to be loaded.
                end (datetime.datetime): Date of the last map to be loaded.

            Return:
                pyinterp.trivariate.Trivariate: The interpolator handling the
                interpolation of the grid series.
            """
            if start < self.df.index[0] or end > self.df.index[-1]:
                raise IndexError(
                    f"period [{start}, {end}] out of range [{self.df.index[0]}, "
                    f"{self.df.index[-1]}]")
            first = start - self.dt
            last = end + self.dt

            selected = self.df[(self.df.index >= first) & (self.df.index < last)]

            x_axis = y_axis = None
            t_axis = pyinterp.core.Axis(selected.index)

            var = []

            for item in selected["path"]:
                with netCDF4.Dataset(item) as ds:
                    if x_axis is None:
                        x_axis = pyinterp.core.Axis(ds.variables["longitude"][:])
                        y_axis = pyinterp.core.Axis(ds.variables["latitude"][:])

                    def _load(grid):
                        grid[grid.mask] = np.nan
                        return grid[0, :]

                    var.append(_load(ds.variables[varname][:]))

            var = np.stack(var).transpose(2, 1, 0)

            return pyinterp.trivariate.Trivariate(x_axis, y_axis, t_axis, var)

This object allows you to manage the time series of AVISO products, check the
continuity of the current time series and load a user-defined period.

Finally, an object is created that manages an ASCII file containing a time
series of floats. This file contains several columns defining the float
identifier, the date of the measurement, the longitude and the latitude of the
measurement.

.. code:: python

    def cnes_jd_to_datetime(seconds):
        """Convert a date expressed in seconds since 1950 into a calendar
        date."""
        return datetime.datetime.utcfromtimestamp(
            ((seconds / 86400.0) - 7305.0) * 86400.0)


    class DataFrame:
        """To handle a dataset to be interpolated from a time series.
        
        Args:
            path (str): Path to the file to be loaded into memory.
        """

        def __init__(self, path):
            df = pd.read_csv(path,
                            header=None,
                            sep=r"\s+",
                            usecols=[0, 1, 2, 3],
                            names=["id", "time", "lon", "lat"],
                            dtype=dict(id=np.uint32,
                                        time=np.float64,
                                        lon=np.float64,
                                        lat=np.float64))
            df.mask(df == 1.8446744073709552e+19, np.nan, inplace=True)
            df["time"] = df["time"].apply(cnes_jd_to_datetime)
            df.set_index('time', inplace=True)
            df["sla"] = np.nan
            self.df = df.sort_index()

        def periods(self, grid_series, frequency='D'):
            """Return the list of periods covering the time series loaded in
            memory."""
            period_start = self.df.groupby(
                self.df.index.to_period(frequency))["sla"].count().index

            for start, end in zip(period_start, period_start[1:]):
                start = start.to_timestamp()
                if start < grid_series.df.index[0]:
                    start = grid_series.df.index[0]
                end = end.to_timestamp()
                yield start, end
            yield end, self.df.index[-1] + grid_series.dt

        def interpolate(self, grid_series, varname, start, end):
            """Interpolate the time series over the defined period."""
            interpolator = grid_series.load_dataset(varname, start, end)
            mask = (self.df.index >= start) & (self.df.index < end)
            selected = self.df.loc[mask, ["lon", "lat"]]
            self.df.loc[mask, ["sla"]] = interpolator.evaluate(
                selected["lon"].values,
                selected["lat"].values,
                selected.index.values,
                interpolator="inverse_distance_weighting",
                num_threads=0)

This defined object, an interpolation function. This function performs the
following steps: selection of the 3D cube managing the time period to be
interpolated, interpolation of the values in the cable for all positions within
the processed interval.

Finally, we create the two objects to interpolate the time series composed by
the floats loaded in memory

.. code:: python

    grid_series = GridSeries("/work/ALT/odatis/AVISO")
    data = DataFrame("Dump_SVP_forEKValid_Stress.ascii")
    for start, end in data.periods(grid_series, frequency="M"):
        data.interpolate(grid_series, "sla", start, end)