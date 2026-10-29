Orbit Propagation
=================
Tools to simulate the ground track of a satellite from its ephemeris. Start
with :func:`~pyinterp.orbit.calculate_orbit`, which interpolates the orbit over
a cycle, then extract a half-orbit with :func:`~pyinterp.orbit.calculate_pass`
and, for a swath mission, the across-track grid with
:func:`~pyinterp.orbit.calculate_swath`.

.. module:: pyinterp.orbit

Classes
-------

.. autosummary::
   :toctree: _generated/

   EquatorCoordinates
   Orbit
   Pass
   Swath

Functions
---------

.. autosummary::
   :toctree: _generated/

   calculate_orbit
   calculate_pass
   calculate_swath
