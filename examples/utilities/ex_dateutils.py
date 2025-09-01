"""
.. _example_dateutils:

Date and Time Utilities
=======================

The ``pyinterp.dateutils`` module provides a set
of utility functions for working with dates and times in NumPy arrays. These
functions are designed to be fast and efficient, making it easy to perform
common date and time calculations.

This example will walk you through the various functions available in the
``pyinterp.dateutils`` module.
"""
# %%
# Generating Sample Data
# ----------------------
#
# First, let's create a set of random dates that we can use to demonstrate the
# functionality of the `dateutils` module.
import datetime
import random

import numpy

import pyinterp


def make_date(samples=10):
    """Generates random dates."""
    epoch = datetime.datetime(1970, 1, 1)
    delta = datetime.datetime.now() - datetime.datetime(1970, 1, 1)

    pydates = [epoch + random.random() * delta for _ in range(samples)]
    npdates = numpy.array(pydates).astype('datetime64[ns]')

    return npdates


dates = make_date()
print('Sample dates:')
print(dates)

# %%
# Extracting Date Components
# --------------------------
#
# You can extract the date components (year, month, and day) from a NumPy array
# of dates using the
# :py:func:`pyinterp.dateutils.date <pyinterp.core.dateutils.date>` function.
# This returns a structured NumPy array.
date_components = pyinterp.dateutils.date(dates)
print('\\nDate components:')
print(date_components)

# %%
# Extracting Time Components
# --------------------------
#
# Similarly, you can extract the time components (hour, minute, and second)
# using the :py:func:`pyinterp.dateutils.time <pyinterp.core.dateutils.time>`
# function.
time_components = pyinterp.dateutils.time(dates)
print('\\nTime components:')
print(time_components)

# %%
# ISO Calendar Information
# ------------------------
#
# The :py:func:`pyinterp.dateutils.isocalendar
# <pyinterp.core.dateutils.isocalendar>` function returns the ISO calendar
# information (year, week number, and weekday) for each date.
iso_calendar = pyinterp.dateutils.isocalendar(dates)
print('\\nISO calendar:')
print(iso_calendar)

# %%
# Weekday
# -------
#
# You can get the day of the week (where Sunday is 0 and Saturday is 6) using
# the :py:func:`pyinterp.dateutils.weekday <pyinterp.core.dateutils.weekday>`
# function.
weekday = pyinterp.dateutils.weekday(dates)
print('\\nWeekday (Sunday=0):')
print(weekday)

# %%
# Time Since January 1st
# ----------------------
#
# The :py:func:`pyinterp.dateutils.timedelta_since_january
# <pyinterp.core.dateutils.timedelta_since_january>` function calculates
# the time difference between each date and the first day of its corresponding
# year.
timedelta = pyinterp.dateutils.timedelta_since_january(dates)
print('\\nTime since January 1st:')
print(timedelta)

# %%
# Converting to datetime Objects
# ------------------------------
#
# Finally, you can convert a NumPy array of dates to an array of Python's
# native :py:class:`datetime.datetime` objects using the
# :py:func:`pyinterp.dateutils.datetime <pyinterp.core.dateutils.datetime>`
# function.
datetime_objects = pyinterp.dateutils.datetime(dates)
print('\\nDatetime objects:')
print(datetime_objects)
