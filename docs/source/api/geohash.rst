Geohash
=======
Geohashes encode longitude/latitude positions as short strings identifying
cells of a hierarchical grid. Use :func:`~pyinterp.geohash.encode` and
:func:`~pyinterp.geohash.decode` on arrays of points,
:class:`~pyinterp.geohash.GeoHash` to handle a single cell, and
:func:`~pyinterp.geohash.to_xarray` to map values indexed by geohash onto a
longitude/latitude grid.

.. module:: pyinterp.geohash

Classes
-------

.. autosummary::
   :toctree: _generated/

   GeoHash

Functions
---------

.. autosummary::
   :toctree: _generated/

   area
   bounding_boxes
   decode
   encode
   to_xarray
   transform
   where
