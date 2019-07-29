"""
Get software version information
================================
"""


def release(full: bool = False) -> str:
    """Returns the software version number"""
    # ddab30ba4cf5b5d44a049f62741801df8d44886e
    result = "0.0.2"
    if full:
        result += " (29 July 2019)"
    return result
