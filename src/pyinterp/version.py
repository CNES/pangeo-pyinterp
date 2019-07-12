"""
Get software version information
================================
"""


def release(full: bool = False) -> str:
    """Returns the software version number"""
    # c133d9af2aee7c2bd37212fc8cd757a695b3cc5e
    result = "0.0.2"
    if full:
        result += " (12 July 2019)"
    return result
