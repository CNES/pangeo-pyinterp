"""
Get software version information
================================
"""


def release(full: bool = False) -> str:
    """Returns the software version number"""
    # cd4701c9d31dd51b7d28e0a3dfcdd060bb46607c
    result = "0.0.3"
    if full:
        result += " (29 July 2019)"
    return result
