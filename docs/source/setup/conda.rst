Setup with Anaconda / conda-forge
*********************************

`conda <https://docs.conda.io/>`_ (or the faster drop-in replacement
`mamba <https://mamba.readthedocs.io/>`_) can install pre-built binaries with
all compiled dependencies.

Install::

    conda install -c conda-forge pyinterp

(Or with mamba)::

    mamba install -c conda-forge pyinterp

Optional packages you may want::

    conda install xarray dask cartopy pandas

To keep everything current::

    conda update --all

See :doc:`build` for details if you need to compile from source instead.
