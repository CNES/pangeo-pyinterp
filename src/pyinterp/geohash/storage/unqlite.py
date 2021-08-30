# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
from .abc import AbstractMutableMapping
from ...core import storage


class UnQlite(storage.unqlite.Database, AbstractMutableMapping):
    """Storage class using UnQlite.
    """
    def __init__(self, name: str, **kwargs):
        """Initialize UnQlite storage.

        Args:
            name (str): Path to the database file.
            kwargs: Additional keyword arguments are passed to the underlying
                UnQlite class.
        """
        # normalize path
        if name != ':mem:':
            name = os.path.abspath(name)
        super().__init__(name, **kwargs)

    def __enter__(self) -> 'UnQlite':
        return self

    def __exit__(self, type, value, tb):
        self.commit()


class MutableMapping(UnQlite):
    """Shortcut for a memory container."""
    def __init__(self):
        super().__init__(":mem:", mode="w")
