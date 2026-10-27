Setup with Anaconda / conda-forge
*********************************

`conda <https://docs.conda.io/>`_ (or the faster drop-in replacement
`mamba <https://mamba.readthedocs.io/>`_) installs pre-built binaries together
with all their compiled dependencies:

.. tab:: mamba

   .. code-block:: bash

      mamba install -c conda-forge pyinterp

.. tab:: conda

   .. code-block:: bash

      conda install -c conda-forge pyinterp

Optional packages used by the examples:

.. code-block:: bash

   mamba install -c conda-forge cartopy matplotlib pandas

To compile the library yourself instead, see :doc:`build`.
