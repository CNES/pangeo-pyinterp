from ..array import Array1DFloat64, Array1DUInt64

def decode(hash: Array1DUInt64,
           precision: int = ...,
           round: bool = ...) -> tuple[Array1DFloat64, Array1DFloat64]:
    ...


def encode(lon: Array1DFloat64,
           lat: Array1DFloat64,
           precision: int = ...) -> Array1DUInt64:
    ...


def neighbors(hash: int, precision: int = ...) -> Array1DUInt64:
    ...
