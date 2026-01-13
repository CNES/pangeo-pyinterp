from ..type_hints import (
    NDArray1DDateTime64,
    NDArray1DInt32,
    NDArray1DTimeDelta64,
    NDArray1DUInt8,
)

def date(
    array: NDArray1DDateTime64,
) -> tuple[NDArray1DInt32, NDArray1DUInt8, NDArray1DUInt8]: ...
def isocalendar(
    array: NDArray1DDateTime64,
) -> tuple[NDArray1DInt32, NDArray1DUInt8, NDArray1DUInt8]: ...
def time(
    array: NDArray1DDateTime64,
) -> tuple[NDArray1DUInt8, NDArray1DUInt8, NDArray1DUInt8]: ...
def timedelta_since_january(
    array: NDArray1DDateTime64,
) -> NDArray1DTimeDelta64: ...
def weekday(array: NDArray1DDateTime64) -> NDArray1DUInt8: ...
