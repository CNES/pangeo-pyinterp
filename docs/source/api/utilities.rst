Date and Period Utilities
=========================
Vectorized helpers for ``numpy.datetime64`` arrays and time periods. The
``pyinterp.dateutils`` functions decompose dates without converting them to
``datetime`` objects, and :class:`~pyinterp.core.period.Period` /
:class:`~pyinterp.core.period.PeriodList` represent time intervals and
collections of them; both modules are available as ``pyinterp.dateutils`` and
``pyinterp.period``.

Date Utilities
--------------

.. currentmodule:: pyinterp.core.dateutils

.. autosummary::
   :toctree: _generated/

   date
   time
   weekday
   isocalendar
   timedelta_since_january

Periods
-------

.. currentmodule:: pyinterp.core.period

.. autosummary::
   :toctree: _generated/

   Period
   PeriodList
