from . import unqlite


class Marshaller:
    def __init__(self) -> None:
        ...

    def dumps(self, obj: object) -> bytes:
        ...

    def loads(self, bytes_object: bytes) -> object:
        ...
