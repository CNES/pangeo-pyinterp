from typing import SupportsInt

from ..typing import NDArray, NDArrayInt64, NDArrayStructured, NDArrayTimeDelta

def date(array: NDArray) -> NDArrayStructured:
    ...


def datetime(array: NDArray) -> NDArray:
    ...

def datetime64_to_str(value: SupportsInt, resolution: str) -> str: ...


def timedelta_since_january(array: NDArray) -> NDArrayTimeDelta:
    ...


def isocalendar(array: NDArray) -> NDArrayStructured:
    ...


def time(array: NDArray) -> NDArrayStructured:
    ...


def weekday(array: NDArray) -> NDArrayInt64:
    ...
