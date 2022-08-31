"""
**********************
Descriptive Statistics
**********************

Numpy offers many statistical functions, but if you want to obtain several
statistical variables from the same array, it's necessary to process the data
several times to calculate the various parameters. This example shows how to use
the DescriptiveStatistics class to obtain several statistical variables with a
single calculation. Also, the calculation algorithm is incremental and is more
numerically stable.

.. note::

    Pébay, P., Terriberry, T.B., Kolla, H. et al.
    Numerically stable, scalable formulas for parallel and online
    computation of higher-order multivariate central moments
    with arbitrary weights.
    Comput Stat 31, 1305–1325,
    2016,
    https://doi.org/10.1007/s00180-015-0637-z
"""
# %%
import dask.array
import numpy

import pyinterp

# %%
# Create a random array
values = numpy.random.random_sample((2, 4, 6, 8))

# %%
# Create a DescriptiveStatistics object.
ds = pyinterp.DescriptiveStatistics(values)

# %%
# The constructor will calculate the statistical variables on the provided data.
# The calculated variables are stored in the instance and can be accessed using
# different methods:
#
# * mean
# * var
# * std
# * skewness
# * kurtosis
# * min
# * max
# * sum
# * sum_of_weights
# * count
ds.count()

# %%
ds.mean()

# %%
# It's possible to get a structured numpy array containing the different
# statistical variables calculated.
ds.array()

# %%
# Like numpy, it's possible to compute statistics along axis.
ds = pyinterp.DescriptiveStatistics(values, axis=(1, 2))
ds.mean()

# %%
# The class can also process a dask array. In this case, the call to the
# constructor triggers the calculation.
ds = pyinterp.DescriptiveStatistics(dask.array.from_array(values,
                                                          chunks=(2, 2, 2, 2)),
                                    axis=(1, 2))
ds.mean()

# %%
# Finally, it's possible to calculate weighted statistics.
weights = numpy.random.random_sample((2, 4, 6, 8))
ds = pyinterp.DescriptiveStatistics(values, weights=weights, axis=(1, 2))
ds.mean()
