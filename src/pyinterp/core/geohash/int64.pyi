from ...typing import NDArray1DFloat64, NDArray1DUInt64

def decode(hash: NDArray1DUInt64,
           precision: int = ...,
           round: bool = ...) -> tuple[NDArray1DFloat64, NDArray1DFloat64]:
    ...


def encode(lon: NDArray1DFloat64,
           lat: NDArray1DFloat64,
           precision: int = ...) -> NDArray1DUInt64:
    ...


def neighbors(hash: int, precision: int = ...) -> NDArray1DUInt64:
    ...
