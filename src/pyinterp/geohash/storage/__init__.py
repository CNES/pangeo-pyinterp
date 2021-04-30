# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Index storage support
---------------------
"""
from .abc import AbstractMutableMapping
from .file_system import FileSystem
from .unqlite import MutableMapping, UnQlite
