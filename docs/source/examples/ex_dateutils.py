"""
********************
Numpy date utilities
********************

This library provides utility functions to perform conversions and get
information about numpy dates quickly.
"""
# %%
import datetime
import random

import numpy

import pyinterp


def make_date(samples=10000):
    """Generates random dates."""
    epoch = datetime.datetime(1970, 1, 1)
    delta = datetime.datetime.now() - datetime.datetime(1970, 1, 1)

    pydates = [epoch + random.random() * delta for _ in range(samples)]
    npdates = numpy.array(pydates).astype("datetime64[ns]")

    return npdates


# %%
dates = make_date()
dates

# %%
# Get the date part as a structured numpy array of three fields: ``year``,
# ``month`` and ``day``:
pyinterp.dateutils.date(dates)

# %%
# Get the time part as a structured numpy array of three fields: ``hour``,
# ``minute`` and ``second``:
pyinterp.dateutils.time(dates)

# %%
# Get the ISO calendar of the date as a structured numpy array of three fields:
# ``year``, ``weekday`` and ``week``:
pyinterp.dateutils.isocalendar(dates)

# %%
# Get the week day of the dates (Sunday is 0 ... Saturday is 6):
pyinterp.dateutils.weekday(dates)

# %%
# Get the timedelta from since January
pyinterp.dateutils.timedelta_since_january(dates)

# %%
# Get the dates as datetime.datetime array
pyinterp.dateutils.datetime(dates)
