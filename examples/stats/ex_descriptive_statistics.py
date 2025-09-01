"""
.. _example_descriptive_statistics:

Descriptive Statistics
======================

While NumPy provides a wide range of statistical functions, calculating multiple
statistical variables from the same array often requires multiple passes over
the data. The :py:class:`pyinterp.DescriptiveStatistics` class offers a more
efficient solution by computing several statistical variables in a single pass.
This approach is not only faster but also more numerically stable, thanks to its
incremental calculation algorithm.

.. note::

    This implementation is based on the following paper:

    Pébay, P., Terriberry, T.B., Kolla, H. et al.
    Numerically stable, scalable formulas for parallel and online
    computation of higher-order multivariate central moments
    with arbitrary weights.
    Comput Stat 31, 1305-1325,
    2016,
    https://doi.org/10.1007/s00180-015-0637-z
"""
# %%
# Basic Usage
# -----------
#
# Let's start by creating a random array and using it to initialize the
# :py:class:`pyinterp.DescriptiveStatistics` class.
import dask.array
import numpy

import pyinterp

generator = numpy.random.Generator(numpy.random.PCG64(0))
values = generator.random((2, 4, 6, 8))

ds = pyinterp.DescriptiveStatistics(values)

# %%
# Once the object is created, you can access various statistical variables,
# such as the count, mean, variance, standard deviation, skewness, kurtosis,
# minimum, maximum, and sum.
print(f'Count: {ds.count()}')
print(f'Mean: {ds.mean()}')
print(f'Variance: {ds.var()}')

# %%
# You can also get all the calculated statistical variables as a structured
# NumPy array.
stats_array = ds.array()
print('Structured array of statistics:')
print(stats_array)

# %%
# Computing Statistics Along an Axis
# ----------------------------------
#
# Similar to NumPy, you can compute statistics along a specific axis by
# providing the ``axis`` parameter.
ds_axis = pyinterp.DescriptiveStatistics(values, axis=(1, 2))
print('Mean along axis (1, 2):')
print(ds_axis.mean())

# %%
# Working with Dask Arrays
# ------------------------
#
# The :py:class:`pyinterp.DescriptiveStatistics` class also supports Dask
# arrays, allowing you to work with datasets that are larger than memory.
dask_values = dask.array.from_array(values, chunks=(2, 2, 2, 2))
ds_dask = pyinterp.DescriptiveStatistics(dask_values, axis=(1, 2))
print('Mean with Dask array:')
print(ds_dask.mean())

# %%
# Weighted Statistics
# -------------------
#
# You can also calculate weighted statistics by providing a ``weights`` array.
weights = generator.random((2, 4, 6, 8))
ds_weighted = pyinterp.DescriptiveStatistics(values,
                                             weights=weights,
                                             axis=(1, 2))
print('Weighted mean:')
print(ds_weighted.mean())
