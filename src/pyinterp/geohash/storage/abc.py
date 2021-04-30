# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Abstract index storage class
----------------------------
"""
from typing import Any, Iterable, List, Optional, Tuple
import abc


class AbstractMutableMapping:  # pragma: no cover
    """Abstract index storage class"""
    @abc.abstractmethod
    def __contains__(self, key: bytes) -> bool:
        ...

    @abc.abstractmethod
    def __delitem__(self, key: bytes) -> None:
        ...

    @abc.abstractmethod
    def __getitem__(self, key: bytes) -> list:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __setitem__(self, key: bytes, value: object) -> None:
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        ...

    @abc.abstractmethod
    def extend(self, other: Iterable[Tuple[bytes, Any]]) -> None:
        ...

    @abc.abstractmethod
    def keys(self) -> List[bytes]:
        ...

    @abc.abstractmethod
    def update(self, other: Iterable[Tuple[bytes, Any]]) -> None:
        ...

    @abc.abstractmethod
    def values(self, keys: Optional[List[bytes]] = None) -> List[Any]:
        ...

    @abc.abstractmethod
    def items(self,
              keys: Optional[List[bytes]] = None) -> List[Tuple[bytes, Any]]:
        ...

    @abc.abstractmethod
    def __enter__(self) -> Any:
        ...

    @abc.abstractmethod
    def __exit__(self, type, value, tb):
        ...

    def __iter__(self):
        return self.keys()
