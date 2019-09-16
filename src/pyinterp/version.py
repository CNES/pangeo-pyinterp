"""
Get software version information
================================
"""


def release(full: bool = False) -> str:
    """Returns the software version number"""
    result = "0.0.4"
    if full:
        result += " (16 September 2019)"
    return result
