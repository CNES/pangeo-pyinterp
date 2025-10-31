# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Root pytest configuration file for local development."""
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options to pytest.

    This function is called early by pytest to register custom options.
    These options may also be defined in pyinterp/tests/conftest.py
    when pytest runs from the build directory.
    """
    # Only add options if they haven't been added already
    try:
        parser.addoption('--visualize', action='store_true', default=False)
        parser.addoption('--dump', action='store_true', default=False)
        parser.addoption('--measure-coverage',
                         action='store_true',
                         default=False)
    except ValueError:
        pass  # Already added
