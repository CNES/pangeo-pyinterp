Installing the library
======================

Pre-built packages are available on `PyPI <https://pypi.org/p/pyinterp>`_ and
`conda-forge <https://conda-forge.org>`_, so most users do not need to compile
anything:

.. tab:: pip

   .. code-block:: bash

      pip install pyinterp

.. tab:: conda / mamba

   .. code-block:: bash

      mamba install -c conda-forge pyinterp

:doc:`setup/pip` lists the platforms covered by the wheels. Contributors, and
users of other platforms, can build the library by following
:doc:`setup/build`.

.. toctree::
   :maxdepth: 1
   :hidden:

   setup/pip
   setup/conda
   setup/build
