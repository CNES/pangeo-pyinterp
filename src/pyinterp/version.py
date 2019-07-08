"""
Get software version information
================================
"""


def release(full: bool = False) -> str:
    """Returns the software version number"""
    # 66148b7562ad646aab3d03d525e5191df0525184
    result = "0.0.1"
    if full:
        result += " (08 July 2019)"
    return result
