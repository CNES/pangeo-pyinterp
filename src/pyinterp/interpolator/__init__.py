# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Interpolation functions
-----------------------
"""
from .bicubic import bicubic
from .bivariate import bivariate
from .quadrivariate import quadrivariate
from .trivariate import trivariate

__all__ = [
    'bicubic',
    'bivariate',
    'quadrivariate',
    'trivariate',
]
