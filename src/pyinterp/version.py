"""
Get software version information
================================
"""


def release(full: bool = False) -> str:
    """Returns the software version number"""
    result = "0.0.5"
    if full:
        result += " (19 September 2019)"
    return result
