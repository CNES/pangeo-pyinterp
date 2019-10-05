# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from . import version
from .core import Axis
from .grid import Grid2D, Grid3D
from .rtree import RTree
from .interpolator.bicubic import bicubic
from .interpolator.bivariate import bivariate
from .interpolator.trivariate import trivariate
__version__ = version.release()
del version
