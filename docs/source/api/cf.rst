CF Helpers
==========
Sets of the units defined by the Climate and Forecast (CF) conventions for
longitudes, latitudes and times. The xarray backend (see :doc:`grids`) uses
them to identify the axes of a ``DataArray`` from the ``units`` attribute of
its coordinates.

.. module:: pyinterp.cf

Classes
-------

.. autosummary::
   :toctree: _generated/

   AxisUnit
   AxisLatitudeUnit
   AxisLongitudeUnit
   AxisTimeUnit
