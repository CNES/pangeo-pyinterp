from typing import Any, Dict, Iterable, List, Optional, Tuple


class DatabaseError(RuntimeError):
    ...


class LockError(OSError):
    ...


class OperationalError(RuntimeError):
    ...


class ProgrammingError(RuntimeError):
    ...


class CompressionType:
    none: 'CompressionType'
    snappy: 'CompressionType'


class Database:
    def __init__(
            self,
            name: str,
            mode: Optional[str] = None,
            compression_type: CompressionType = CompressionType.snappy
    ) -> None:
        ...

    def __contains__(self, key: bytes) -> bool:
        ...

    def __delitem__(self, key: bytes) -> None:
        ...

    def __getitem__(self, key: bytes) -> List[Any]:
        ...

    def __getstate__(self) -> Tuple:
        ...

    def __setstate__(self, state: Tuple) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __setitem__(self, key: bytes, value: Any) -> None:
        ...

    def clear(self) -> None:
        ...

    def commit(self) -> None:
        ...

    def error_log(self) -> str:
        ...

    def extend(self, other: Iterable[Tuple[bytes, Any]]) -> None:
        ...

    def items(self,
              keys: Optional[List[bytes]] = None) -> List[Tuple[bytes, Any]]:
        ...

    def keys(self) -> List[bytes]:
        ...

    def rollback(self) -> None:
        ...

    def update(self, other: Iterable[Tuple[bytes, Any]]) -> None:
        ...

    def values(self, keys: Optional[List[bytes]] = None) -> List[Any]:
        ...
