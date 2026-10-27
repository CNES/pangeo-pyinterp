Axes
====
Axes hold the coordinates of each dimension of a grid. Use
:class:`~pyinterp.Axis` for numeric coordinates, periodic ones included (e.g.
longitudes with ``period=360``), and :class:`~pyinterp.TemporalAxis` for
``numpy.datetime64`` coordinates; the axes are then combined with the data in
a :class:`~pyinterp.Grid` (see :doc:`grids`).

.. currentmodule:: pyinterp

Classes
-------

.. autosummary::
    :toctree: _generated

    Axis
    TemporalAxis
