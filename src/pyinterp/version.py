"""
Get software version information
================================
"""


def release(full: bool = False) -> str:
    """Returns the software version number"""
    result = "0.0.6"
    if full:
        result += " (26 October 2019)"
    return result
