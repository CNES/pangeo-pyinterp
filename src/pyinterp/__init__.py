import numpy as np


def _core_suffix(x: np.ndarray):
    """Get the suffix of the class handling the numpy data type.

    Args:
        x (numpy.ndarray): array to process
    Returns:
        str: the class suffix
    """
    dtype = x.dtype.type
    if dtype == np.float64:
        return 'Float64'
    if dtype == np.float32:
        return 'Float32'
    if dtype == np.int64:
        return 'Int64'
    if dtype == np.uint64:
        return 'UInt64'
    if dtype == np.int32:
        return 'Int32'
    if dtype == np.uint32:
        return 'UInt32'
    if dtype == np.int16:
        return 'Int16'
    if dtype == np.uint16:
        return 'UInt16'
    if dtype == np.int8:
        return 'Int8'
    if dtype == np.uint8:
        return 'UInt8'
    raise ValueError("Unhandled dtype: " + str(dtype))
